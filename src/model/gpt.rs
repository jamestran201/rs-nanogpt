use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::block::Block;
use super::config::GptConfig;
use super::embedding::TokenEmbedding;
use super::linear::Linear;
use super::rms_norm::rms_norm;

/// The GPT model: token embedding → `n_layer` pre-norm transformer blocks →
/// final RMSNorm → untied `lm_head` unembedding, producing logits `(B,T,vocab)`.
pub struct Gpt {
    wte: TokenEmbedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    config: GptConfig,
}

impl Gpt {
    pub fn new(cfg: GptConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let wte = TokenEmbedding::new(&cfg, vb.pp("wte"))?;

        let blocks_vb = vb.pp("blocks");
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(&cfg, blocks_vb.pp(i), &device))
            .collect::<Result<Vec<_>>>()?;

        // Untied unembedding: a separate matrix from `wte`, tiny-init so the
        // initial logits are near-uniform and the loss starts at ≈ ln(vocab).
        let lm_head = Linear::normal(cfg.n_embd, cfg.vocab_size, 0.001, vb.pp("lm_head"))?;

        Ok(Self {
            wte,
            blocks,
            lm_head,
            config: cfg,
        })
    }

    /// Map token ids `idx: (B, T)` to logits `(B, T, vocab_size)`.
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        let mut x = self.wte.forward(idx)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = rms_norm(&x, self.config.norm_eps)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &GptConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Reduction, cross_entropy};
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len: 16,
            n_layer: 2,
            n_head: 2,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    fn builder(vm: &VarMap, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_varmap(vm, DType::F32, dev)
    }

    #[test]
    fn forward_returns_b_t_vocab() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let vocab = cfg.vocab_size;
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?; // (B=2, T=3)
        assert_eq!(model.forward(&idx)?.dims(), &[2, 3, vocab]);
        Ok(())
    }

    #[test]
    fn registers_full_parameter_set() -> Result<()> {
        // Locks in the parameter naming as the model grows: the embedding, one
        // `blocks.{i}.*` group per layer, and the untied `lm_head`.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let (vocab, n_embd, n_layer) = (cfg.vocab_size, cfg.n_embd, cfg.n_layer);
        let _model = Gpt::new(cfg, builder(&vm, &dev))?;

        let data = vm.data().lock().unwrap();
        let mut keys: Vec<String> = data.keys().cloned().collect();
        keys.sort();

        let mut want = vec!["lm_head.weight".to_string(), "wte.weight".to_string()];
        for i in 0..n_layer {
            for name in [
                "attn.c_k", "attn.c_proj", "attn.c_q", "attn.c_v", "mlp.c_fc", "mlp.c_proj",
            ] {
                want.push(format!("blocks.{i}.{name}.weight"));
            }
        }
        want.sort();

        assert_eq!(keys, want);
        assert_eq!(data["wte.weight"].dims(), &[vocab, n_embd]);
        assert_eq!(data["lm_head.weight"].dims(), &[vocab, n_embd]);
        Ok(())
    }

    #[test]
    fn init_logits_are_near_uniform() -> Result<()> {
        // At init every block is the identity (zero-init residual projections)
        // and `lm_head` is tiny-init, so the logits are ~0 ⇒ softmax ≈ uniform,
        // i.e. cross-entropy will start at ≈ ln(vocab).
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3, 4]], &dev)?;
        let logits = model.forward(&idx)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            logits.iter().all(|v| v.abs() < 0.05),
            "expected near-zero init logits, got max {:?}",
            logits.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()))
        );
        Ok(())
    }

    #[test]
    fn init_loss_is_near_ln_vocab() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let vocab = cfg.vocab_size;
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?; // (B=2, T=4)
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?; // (B=2, T=4)

        let logits = model.forward(&idx)?; // (2, 4, vocab)
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;

        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let expected = (vocab as f32).ln();
        assert!(
            (loss - expected).abs() < 0.1,
            "got {loss}, want ≈ {expected}"
        );
        Ok(())
    }
}
