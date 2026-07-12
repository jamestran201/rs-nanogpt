use candle_core::{DType, Device, Result};

/// Select the compute device based on the enabled backend feature.
///
/// Defaults to CPU. Build with `--features metal` (Apple GPU) or
/// `--features cuda` (NVIDIA, the cloud path) to target a GPU.
pub fn default_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0)
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::new_metal(0)
    }
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    {
        Ok(Device::Cpu)
    }
}

/// The dtype activations are computed in on `dev` (weights stay fp32 `Var`s;
/// each forward casts them to this dtype in-graph).
///
/// bf16 on CUDA to reduce activation memory by approx half. Other systems use fp32 because bf16 is only supported on CUDA
pub fn compute_dtype(dev: &Device) -> DType {
    if dev.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_computes_in_f32() {
        // The guardrail behind the whole bf16 plan: on CPU every dtype cast is
        // a same-dtype no-op, so the existing test suite exercises an
        // unchanged fp32 graph.
        assert_eq!(compute_dtype(&Device::Cpu), DType::F32);
    }
}
