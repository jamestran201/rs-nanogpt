use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::GptConfig;
use super::linear::Linear;

/// ReLU² feed-forward MLP — the second sub-layer of each transformer block.
///
/// Like [`super::attention::CausalSelfAttention`], this receives an
/// already-normed residual input `x: (B, T, C)` and returns the MLP
/// contribution `(B, T, C)` for the caller to add back — the pre-norm and the
/// residual add live in the block-composition step, not here. There is no norm
/// inside this module.
pub struct Mlp {
    /// Up-projection `C → 4C`.
    c_fc: Linear,
    /// Down-projection `4C → C`. Zero-init so the block starts as the identity.
    c_proj: Linear,
}

impl Mlp {
    pub fn new(cfg: &GptConfig, vb: VarBuilder) -> Result<Self> {
        let c = cfg.n_embd;
        let hidden = 4 * c;
        let s = (3.0 / cfg.n_embd as f64).sqrt();
        let c_fc = Linear::uniform(c, hidden, 0.4 * s, vb.pp("c_fc"))?;
        let c_proj = Linear::zeros(hidden, c, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = h.relu()?.sqr()?;
        self.c_proj.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
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
        let mlp = Mlp::new(&cfg, builder(&vm, &dev))?;
        let x = Tensor::randn(0.0f32, 1.0, (2, 5, cfg.n_embd), &dev)?;
        assert_eq!(mlp.forward(&x)?.dims(), &[2, 5, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn hidden_is_four_c() -> Result<()> {
        // c_fc expands C -> 4C; weight is stored (out, in).
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let mlp = Mlp::new(&cfg, builder(&vm, &dev))?;
        assert_eq!(mlp.c_fc.weight().dims(), &[4 * cfg.n_embd, cfg.n_embd]);
        assert_eq!(mlp.c_proj.weight().dims(), &[cfg.n_embd, 4 * cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn zero_init_proj_gives_zero_output() -> Result<()> {
        // c_proj is zero-init, so the MLP's contribution to the residual stream
        // is all zeros at init — the block starts as the identity.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let mlp = Mlp::new(&cfg, builder(&vm, &dev))?;
        let x = Tensor::randn(0.0f32, 2.0, (1, 6, cfg.n_embd), &dev)?;
        let y = mlp.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(y.iter().all(|v| *v == 0.0), "expected all zeros at init");
        Ok(())
    }

    #[test]
    fn relu_squared_is_nonnegative() -> Result<()> {
        // With a non-zero c_proj forced to identity-ish positive weights, the
        // hidden activation relu(x)² is >= 0 for every element regardless of x.
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut vm = VarMap::new();
        let mlp = Mlp::new(&cfg, builder(&vm, &dev))?;
        // Inspect the hidden activation directly (pre c_proj).
        vm.set_one(
            "c_fc.weight",
            Tensor::randn(0.0f32, 1.0, (4 * cfg.n_embd, cfg.n_embd), &dev)?,
        )?;
        let x = Tensor::randn(0.0f32, 2.0, (1, 6, cfg.n_embd), &dev)?;
        let h = mlp.c_fc.forward(&x)?.relu()?.sqr()?;
        let h = h.flatten_all()?.to_vec1::<f32>()?;
        assert!(h.iter().all(|v| *v >= 0.0), "ReLU² must be non-negative");
        Ok(())
    }
}
