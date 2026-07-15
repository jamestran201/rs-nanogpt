use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::CausalSelfAttention;
use super::config::GptConfig;
use super::mlp::Mlp;
use super::rms_norm::rms_norm;
use super::rope::Rope;

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
    /// `rope`/`causal_mask` are shared from `Gpt::new`; the clones taken here
    /// are cheap handles.
    pub fn new(
        cfg: &GptConfig,
        vb: VarBuilder,
        rope: &Rope,
        causal_mask: &Tensor,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::new(cfg, vb.pp("attn"), rope, causal_mask)?;
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
    use crate::test_support::new_block;
    use candle_core::Device;
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

    #[test]
    fn forward_preserves_b_t_c() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let block = new_block(&cfg, &vm, &dev)?;
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
        let block = new_block(&cfg, &vm, &dev)?;
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
        let _block = new_block(&cfg, &vm, &dev)?;

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
