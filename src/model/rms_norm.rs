use candle_core::{D, DType, Result, Tensor};

pub fn rms_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    // The mean-of-squares reduction runs in fp32 even for bf16 activations —
    let mean_sq = x.to_dtype(DType::F32)?.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = mean_sq.affine(1.0, eps as f64)?.sqrt()?;
    x.broadcast_div(&denom.to_dtype(x.dtype())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const EPS: f32 = 1e-6;

    fn rms(v: &[f32]) -> f32 {
        (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32).sqrt()
    }

    #[test]
    fn known_vector_value() -> Result<()> {
        let dev = Device::Cpu;
        // x = [3, 4]: mean(x²) = (9+16)/2 = 12.5, rms = 3.535534.
        let x = Tensor::new(&[3.0f32, 4.0], &dev)?;
        let y = rms_norm(&x, EPS)?.to_vec1::<f32>()?;
        assert!((y[0] - 3.0 / 3.535534).abs() < 1e-4, "got {}", y[0]);
        assert!((y[1] - 4.0 / 3.535534).abs() < 1e-4, "got {}", y[1]);
        Ok(())
    }

    #[test]
    fn zero_input_is_finite() -> Result<()> {
        // The eps in sqrt(mean(x²) + eps) exists to keep an all-zero vector
        // (rms = 0) from dividing by zero; the output must stay finite.
        let dev = Device::Cpu;
        let x = Tensor::zeros((8,), DType::F32, &dev)?;
        let y = rms_norm(&x, EPS)?.to_vec1::<f32>()?;
        assert!(y.iter().all(|v| v.is_finite()), "got {y:?}");
        Ok(())
    }

    #[test]
    fn output_rms_is_unit() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0.0f32, 5.0, (32,), &dev)?;
        let y = rms_norm(&x, EPS)?.to_vec1::<f32>()?;
        assert!((rms(&y) - 1.0).abs() < 1e-3, "output rms was {}", rms(&y));
        Ok(())
    }

    #[test]
    fn bf16_input_keeps_dtype_with_unit_rms() -> Result<()> {
        // The real CUDA compute dtype is testable here on CPU because rms_norm
        // is built from unary/reduce ops (no matmul): the reduction upcasts to
        // fp32, and the output must come back in the input's dtype.
        let dev = Device::Cpu;
        let x = Tensor::randn(0.0f32, 5.0, (32,), &dev)?.to_dtype(DType::BF16)?;
        let y = rms_norm(&x, EPS)?;
        assert_eq!(y.dtype(), DType::BF16);
        let y = y.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        // bf16's ~0.4% per-element rounding: loose tolerance vs the f32 tests.
        assert!((rms(&y) - 1.0).abs() < 0.02, "output rms was {}", rms(&y));
        Ok(())
    }

    #[test]
    fn shape_is_preserved() -> Result<()> {
        let dev = Device::Cpu;
        let x = Tensor::randn(0.0f32, 1.0, (2, 3, 5), &dev)?;
        assert_eq!(rms_norm(&x, EPS)?.dims(), &[2, 3, 5]);
        Ok(())
    }

    #[test]
    fn normalizes_only_the_last_dim() -> Result<()> {
        // Shape (B, T, n_head, head_dim): each head_dim slice should be unit-rms
        // independently; the norm must not mix across the other axes.
        let dev = Device::Cpu;
        let (b, t, h, dh) = (2usize, 3, 2, 4);
        let x = Tensor::randn(0.0f32, 3.0, (b, t, h, dh), &dev)?;
        // Flatten the leading dims so every row is one head_dim slice.
        let flat = rms_norm(&x, EPS)?
            .reshape((b * t * h, dh))?
            .to_vec2::<f32>()?;
        for slice in &flat {
            assert!((rms(slice) - 1.0).abs() < 1e-3, "slice rms {}", rms(slice));
        }
        Ok(())
    }
}
