use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::CausalSelfAttention;
use super::config::GptConfig;
use super::mlp::Mlp;
use super::rms_norm::rms_norm;

/// One pre-norm transformer block: attention sub-layer then MLP sub-layer, each
/// wrapped in a residual connection.
///
/// This is the only place the pre-norm and the residual add live — both
/// [`CausalSelfAttention`] and [`Mlp`] take an already-normed input and return
/// just their contribution, so the block normalizes the input, runs the
/// sub-layer, and adds the result back onto the untouched residual stream:
///
/// ```text
/// x = x + attn(rmsnorm(x))
/// x = x + mlp (rmsnorm(x))
/// ```
pub struct Block {
    attn: CausalSelfAttention,
    mlp: Mlp,
    norm_eps: f32,
}

impl Block {
    pub fn new(cfg: &GptConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let attn = CausalSelfAttention::new(cfg, vb.pp("attn"), device)?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            attn,
            mlp,
            norm_eps: cfg.norm_eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(&rms_norm(x, self.norm_eps)?)?)?;
        let x = (&x + self.mlp.forward(&rms_norm(&x, self.norm_eps)?)?)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len: 16,
            n_layer: 1,
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
    fn forward_preserves_b_t_c() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let block = Block::new(&cfg, builder(&vm, &dev), &dev)?;
        let x = Tensor::randn(0.0f32, 1.0, (2, 5, cfg.n_embd), &dev)?;
        assert_eq!(block.forward(&x)?.dims(), &[2, 5, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn is_identity_at_init() -> Result<()> {
        // Both residual projections are zero-init, so each sub-layer contributes
        // zeros and the block returns its input unchanged.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let block = Block::new(&cfg, builder(&vm, &dev), &dev)?;
        let x = Tensor::randn(0.0f32, 2.0, (1, 6, cfg.n_embd), &dev)?;
        let got = block.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let want = x.flatten_all()?.to_vec1::<f32>()?;
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
        Ok(())
    }

    #[test]
    fn registers_attn_and_mlp_weights() -> Result<()> {
        // Locks in the parameter naming under a block: `attn.*` and `mlp.*`.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let _block = Block::new(&cfg, builder(&vm, &dev), &dev)?;

        let data = vm.data().lock().unwrap();
        let mut keys: Vec<String> = data.keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "attn.c_k.weight",
                "attn.c_proj.weight",
                "attn.c_q.weight",
                "attn.c_v.weight",
                "mlp.c_fc.weight",
                "mlp.c_proj.weight",
            ]
        );
        Ok(())
    }
}
