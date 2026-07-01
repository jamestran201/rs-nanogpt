use candle_core::{Result, Tensor};
use candle_nn::{Init, Module, VarBuilder};

/// The weight has shape `(out_dim, in_dim)` to match candle's (and PyTorch's)
/// convention of storing `wᵀ`.
pub struct Linear {
    inner: candle_nn::Linear,
}

impl Linear {
    /// `Uniform(-s, s)` init — used for the input-side projections (`c_q`, `c_k`, `c_v`, `mlp.c_fc`).
    pub fn uniform(in_dim: usize, out_dim: usize, s: f64, vb: VarBuilder) -> Result<Self> {
        Self::with_init(in_dim, out_dim, Init::Uniform { lo: -s, up: s }, vb)
    }

    /// `Normal(0, std)` init — used for the `lm_head` unembedding, with a
    /// deliberately tiny `std` so the initial logits are near-uniform and the
    /// loss starts at ≈ ln(vocab).
    pub fn normal(in_dim: usize, out_dim: usize, std: f64, vb: VarBuilder) -> Result<Self> {
        Self::with_init(
            in_dim,
            out_dim,
            Init::Randn {
                mean: 0.0,
                stdev: std,
            },
            vb,
        )
    }

    /// Zero init — used for the residual output projections (`c_proj`,
    /// `mlp.c_proj`). Starting these at zero makes each block the identity at
    /// init, so the residual stream is a clean highway and early training is
    /// stable.
    pub fn zeros(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_init(in_dim, out_dim, Init::Const(0.0), vb)
    }

    fn with_init(in_dim: usize, out_dim: usize, init: Init, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", init)?;
        Ok(Self {
            inner: candle_nn::Linear::new(weight, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn builder(vm: &VarMap, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_varmap(vm, DType::F32, dev)
    }

    #[test]
    fn forward_maps_in_dim_to_out_dim() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let lin = Linear::uniform(4, 7, 0.5, builder(&vm, &dev))?;
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 4), &dev)?; // (B, T, in_dim)
        assert_eq!(lin.forward(&x)?.dims(), &[2, 3, 7]);
        Ok(())
    }

    #[test]
    fn weight_is_out_by_in_and_bias_free() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let lin = Linear::uniform(4, 7, 0.5, builder(&vm, &dev))?;
        // Stored as (out_dim, in_dim).
        assert_eq!(lin.weight().dims(), &[7, 4]);
        // Bias-free: the VarMap holds exactly the weight, nothing named "bias".
        let data = vm.data().lock().unwrap();
        assert!(data.contains_key("weight"));
        assert!(!data.keys().any(|k| k.contains("bias")));
        Ok(())
    }

    #[test]
    fn zero_init_maps_everything_to_zero() -> Result<()> {
        // The residual-identity property: a zero-init projection outputs zeros
        // regardless of input, so the block starts as the identity.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let lin = Linear::zeros(5, 5, builder(&vm, &dev))?;
        let x = Tensor::randn(0.0f32, 3.0, (2, 6, 5), &dev)?;
        let y = lin.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(y.iter().all(|v| *v == 0.0), "expected all zeros");
        Ok(())
    }
}
