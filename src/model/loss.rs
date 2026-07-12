use candle_core::{D, DType, Result, Tensor};
use candle_nn::ops::log_softmax;

/// How to collapse the per-position losses into the returned tensor.
pub enum Reduction {
    /// Average over the non-ignored positions → scalar `()`.
    Mean,
    /// Per-position loss, shape `(B, T)`; ignored positions are `0.0`.
    None,
}

/// Cross-entropy between next-token `logits` and the true `targets`.
/// * `logits`  — `(B, T, V)` raw (unnormalized) scores, float.
/// * `targets` — `(B, T)` class ids, dtype **I64** so `ignore_index = -1` is
///   representable.
pub fn cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: i64,
    reduction: Reduction,
) -> Result<Tensor> {
    let (b, t, v) = logits.dims3()?;

    // Flatten to the rank-2 form cross-entropy operates on: one independent
    // next-token classification per (batch, position). (B,T,V) -> (B*T, V).
    // The fp32 upcast (a no-op for fp32 logits) is required under bf16
    // compute: the softmax over the vocab is the most precision-sensitive op,
    // and it keeps the returned loss f32 for host-side reads.
    let logits2d = logits.reshape((b * t, v))?.to_dtype(DType::F32)?;
    let targets1d = targets.reshape(b * t)?;

    // Per-row log-probabilities over the vocab.
    let logp = log_softmax(&logits2d, D::Minus1)?;

    // Mask out ignored positions. `gather` must not see a negative index, so
    // clamp ignored targets to 0 before gathering and zero their loss after.
    let valid = targets1d.ne(ignore_index)?; // U8 (N,)
    let zeros = targets1d.zeros_like()?;
    let safe = valid.where_cond(&targets1d, &zeros)?; // (N,), all >= 0

    // -log p for every row: (N,1) -> (N,).
    let negative_log_likelihood = logp.gather(&safe.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?;

    let validf = valid.to_dtype(DType::F32)?;
    let negative_log_likelihood = (negative_log_likelihood * &validf)?;

    match reduction {
        Reduction::None => negative_log_likelihood.reshape((b, t)),
        Reduction::Mean => {
            // Average over valid tokens only (not B*T), matching ignore_index.
            let total = negative_log_likelihood.sum_all()?;
            let count = validf.sum_all()?;
            total.broadcast_div(&count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Uniform (all-equal) logits put probability 1/V on every token, so the
    // loss at any position is -log(1/V) = ln(V) — this is the value an untrained
    // model sits at, and it makes the expected loss exact.
    fn ln(v: usize) -> f32 {
        (v as f32).ln()
    }

    fn targets_i64(rows: &[&[i64]], dev: &Device) -> Result<Tensor> {
        let (b, t) = (rows.len(), rows[0].len());
        let flat: Vec<i64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::from_vec(flat, (b, t), dev)
    }

    #[test]
    fn uniform_logits_give_ln_vocab() -> Result<()> {
        let dev = Device::Cpu;
        let (b, t, v) = (2, 3, 7);
        let logits = Tensor::zeros((b, t, v), DType::F32, &dev)?;
        let targets = targets_i64(&[&[0, 1, 2], &[3, 4, 5]], &dev)?;
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;
        assert!((loss - ln(v)).abs() < 1e-5, "got {loss}, want {}", ln(v));
        Ok(())
    }

    #[test]
    fn bf16_logits_upcast_to_f32_loss() -> Result<()> {
        // Under bf16 compute the model hands over bf16 logits; the loss must
        // be computed and returned in f32 (uniform logits still give ln(V)
        // exactly — zeros are exact in bf16).
        let dev = Device::Cpu;
        let (b, t, v) = (2, 3, 7);
        let logits = Tensor::zeros((b, t, v), DType::BF16, &dev)?;
        let targets = targets_i64(&[&[0, 1, 2], &[3, 4, 5]], &dev)?;
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?;
        assert_eq!(loss.dtype(), DType::F32);
        let loss = loss.to_scalar::<f32>()?;
        assert!((loss - ln(v)).abs() < 1e-5, "got {loss}, want {}", ln(v));
        Ok(())
    }

    #[test]
    fn known_value_matches_manual_log_softmax() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[[2.0f32, 1.0, 0.1]]], &dev)?;
        let targets = targets_i64(&[&[0]], &dev)?;
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;

        let denom = (2.0f32.exp() + 1.0f32.exp() + 0.1f32.exp()).ln();
        let want = -(2.0 - denom);
        assert!((loss - want).abs() < 1e-5, "got {loss}, want {want}");
        Ok(())
    }

    #[test]
    fn matches_candle_library_when_no_ignores() -> Result<()> {
        // With every target valid, our mean must equal candle's own
        // cross_entropy (which is u32, mean-only, rank-2).
        let dev = Device::Cpu;
        let (n, v) = (5usize, 11usize);
        let logits2d = Tensor::randn(0.0f32, 1.0, (n, v), &dev)?;
        let ids: Vec<u32> = vec![3, 0, 10, 7, 1];

        let lib = candle_nn::loss::cross_entropy(&logits2d, &Tensor::new(ids.clone(), &dev)?)?
            .to_scalar::<f32>()?;

        let logits3d = logits2d.reshape((1, n, v))?;
        let targets = Tensor::new(ids.iter().map(|&x| x as i64).collect::<Vec<_>>(), &dev)?
            .reshape((1, n))?;
        let ours = cross_entropy(&logits3d, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;

        assert!((ours - lib).abs() < 1e-5, "ours {ours} vs candle {lib}");
        Ok(())
    }

    #[test]
    fn mean_excludes_ignored_positions() -> Result<()> {
        // Uniform logits → every valid position contributes ln(V). With two of
        // four positions ignored, the mean must still be ln(V): the denominator
        // counts valid tokens, not all B*T. (If ignored counted, it would be
        // ln(V)/2.)
        let dev = Device::Cpu;
        let v = 5;
        let logits = Tensor::zeros((1, 4, v), DType::F32, &dev)?;
        let targets = targets_i64(&[&[2, -1, 3, -1]], &dev)?;
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;
        assert!((loss - ln(v)).abs() < 1e-5, "got {loss}, want {}", ln(v));
        Ok(())
    }

    #[test]
    fn none_reduction_zeros_ignored_and_keeps_shape() -> Result<()> {
        let dev = Device::Cpu;
        let v = 6;
        let logits = Tensor::zeros((1, 4, v), DType::F32, &dev)?;
        let targets = targets_i64(&[&[5, -1, 1, -1]], &dev)?;
        let loss = cross_entropy(&logits, &targets, -1, Reduction::None)?;
        assert_eq!(loss.dims(), &[1, 4]);

        let got = loss.flatten_all()?.to_vec1::<f32>()?;
        let e = ln(v);
        let want = [e, 0.0, e, 0.0];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "got {got:?}, want {want:?}");
        }
        Ok(())
    }

    #[test]
    fn confident_correct_prediction_is_near_zero() -> Result<()> {
        // A large logit on the target token drives p_target -> 1, loss -> 0.
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[[20.0f32, 0.0, 0.0, 0.0]]], &dev)?;
        let targets = targets_i64(&[&[0]], &dev)?;
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;
        assert!(loss < 1e-6, "expected ~0 loss, got {loss}");
        Ok(())
    }
}
