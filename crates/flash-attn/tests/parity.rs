//! Parity of the vendored FA2 kernels against a naive attention reference.
//!
//! An *integration* test on purpose: it sees only the crate's public API, which
//! doubles as a check that `trainable::` is usable from outside — the one thing
//! the training-path wiring needs.
//!
//! Requires a CUDA device; the whole crate does, so there is no non-CUDA build
//! of this file to guard.
//!
//! Method: build q/k/v in f32, round them to bf16, and cast *those* back to f32
//! for the reference. Both sides then see identical inputs and the only thing
//! measured is accumulation order — not input rounding. bf16 carries ~3 decimal
//! digits, which is what sets the tolerance floor.

use candle::op::BackpropOp;
use candle::{DType, Device, Result, Shape, Storage, Tensor, Var, D};
use candle_nn::ops::softmax;

/// Mixed absolute/relative tolerance. Not pure relative: dQ entries pass
/// through zero, and dividing by ~0 turns a correct kernel into a red test.
const ATOL: f32 = 2e-2;
const RTOL: f32 = 2e-2;

fn assert_close(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
    let g = got.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let w = want.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for (i, (g, w)) in g.iter().zip(&w).enumerate() {
        assert!(
            (g - w).abs() <= ATOL + RTOL * w.abs(),
            "{what}[{i}]: {g} vs {w}"
        );
    }
    Ok(())
}

/// Additive causal mask, `0` on and below the diagonal and `-inf` above.
fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    let keep = Tensor::tril2(t, DType::U8, dev)?;
    let zeros = Tensor::zeros((t, t), DType::F32, dev)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (t, t), dev)?;
    keep.where_cond(&zeros, &neg_inf)
}

/// `softmax(scale·QKᵀ + mask)V` and the row log-sum-exp of the masked scores,
/// all in f32.
///
/// The softmax must be the *composed* `candle_nn::ops::softmax`: the fused
/// `softmax_last_dim` is `apply_op1_no_bwd` and silently severs autograd — no
/// error, just zero gradient into q and k — and this is the grad oracle.
fn naive(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Result<(Tensor, Tensor)> {
    let scores = q
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .affine(scale, 0.0)?;
    let scores = if causal {
        let (_b, _h, t, _d) = q.dims4()?;
        scores.broadcast_add(&causal_mask(t, q.device())?)?
    } else {
        scores
    };
    // logsumexp, factored the stable way; the row max is always causally
    // visible (the diagonal), so it is finite even with -inf entries.
    let m = scores.max_keepdim(D::Minus1)?;
    let lse = (scores
        .broadcast_sub(&m)?
        .exp()?
        .sum_keepdim(D::Minus1)?
        .log()?
        + &m)?
        .squeeze(D::Minus1)?;
    let o = softmax(&scores, D::Minus1)?.matmul(v)?;
    Ok((o, lse))
}

/// Wrap the forward's owned storage as a comparable tensor. `Storage` is
/// re-exported at candle's root but `BackpropOp` is not, hence `candle::op::`.
fn as_tensor(storage: candle::CudaStorage, shape: Shape) -> Tensor {
    Tensor::from_storage(Storage::Cuda(storage), shape, BackpropOp::none(), false)
}

fn case(b: usize, h: usize, t: usize, causal: bool) -> Result<()> {
    let d = 128usize;
    let dev = Device::new_cuda(0)?;
    let scale = 1.0 / (d as f64).sqrt();
    let what = |name: &str| format!("{name} b={b} h={h} t={t} causal={causal}");

    let mk = || -> Result<Tensor> {
        Tensor::randn(0f32, 1.0, (b, h, t, d), &dev)?.to_dtype(DType::BF16)
    };
    let (q, k, v, d_o) = (mk()?, mk()?, mk()?, mk()?);
    // The reference sees exactly the values the kernel sees.
    let (q32, k32, v32) = (
        q.to_dtype(DType::F32)?,
        k.to_dtype(DType::F32)?,
        v.to_dtype(DType::F32)?,
    );
    let do32 = d_o.to_dtype(DType::F32)?;

    let (qv, kv, vv) = (
        Var::from_tensor(&q32)?,
        Var::from_tensor(&k32)?,
        Var::from_tensor(&v32)?,
    );
    let (o_ref, lse_ref) = naive(
        qv.as_tensor(),
        kv.as_tensor(),
        vv.as_tensor(),
        scale,
        causal,
    )?;
    // loss = <dO, O>, whose gradient w.r.t. O is exactly dO — so the autograd
    // grads below are the same VJP the kernel computes.
    let grads_ref = (&o_ref * &do32)?.sum_all()?.backward()?;

    let (o_storage, o_shape, lse) =
        rs_flash_attn::trainable::flash_attn_fwd_lse(&q, &k, &v, scale as f32, causal)?;
    assert_eq!(o_shape.dims(), &[b, h, t, d], "{}", what("o shape"));
    assert_eq!(lse.dims(), &[b, h, t], "{}", what("lse shape"));
    assert_eq!(lse.dtype(), DType::F32, "{}", what("lse dtype"));
    let o = as_tensor(o_storage, o_shape);
    assert_close(&o, &o_ref, &what("o"))?;
    // The LSE is the one value carried between two separate kernel launches, so
    // a layout error here would otherwise surface much later as a wrong grad.
    assert_close(&lse, &lse_ref, &what("lse"))?;

    let (dq, dk, dv) =
        rs_flash_attn::trainable::flash_attn_bwd(&q, &k, &v, &o, &lse, &d_o, scale as f32, causal)?;
    for (got, var, name) in [(&dq, &qv, "dq"), (&dk, &kv, "dk"), (&dv, &vv, "dv")] {
        assert_eq!(
            got.dtype(),
            DType::BF16,
            "{}",
            what(&format!("{name} dtype"))
        );
        assert_eq!(
            got.dims(),
            &[b, h, t, d],
            "{}",
            what(&format!("{name} shape"))
        );
        assert_close(got, grads_ref.get(var.as_tensor()).unwrap(), &what(name))?;
    }
    Ok(())
}

#[test]
fn smallest_case() -> Result<()> {
    case(1, 1, 128, true)
}

#[test]
fn batch_and_head_strides() -> Result<()> {
    // b > 1 and h > 1 exercise the batch/head stride indexing — the part the
    // (B, H, T, D) permuted-stride mapping is entirely responsible for.
    case(2, 4, 256, true)
}

#[test]
fn non_causal() -> Result<()> {
    // The non-causal branch and, with it, the window normalization: forward and
    // backward normalize the window on opposite sides of the FFI and must land
    // on the same `params`.
    case(2, 4, 256, false)
}

#[test]
fn ragged_seqlen() -> Result<()> {
    // 192 % kBlockN != 0 on the >= 144 KB-smem backward path (H100/GH200/A100),
    // so this takes the predicated `is_even_MN == false` branch. On an
    // sm_86-class card kBlockN drops to 64 and this quietly stops exercising it.
    case(1, 2, 192, true)
}
