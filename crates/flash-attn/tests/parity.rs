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

/// Absolute term, as a fraction of the reference tensor's own RMS.
const ATOL: f32 = 2e-2;
/// Relative term, per element.
const RTOL: f32 = 2e-2;
/// Whole-tensor relative Frobenius error, `‖got − want‖₂ / ‖want‖₂`.
const FTOL: f32 = 2e-2;

/// Elementwise `|a − b| ≤ ATOL·rms(b) + RTOL·|b|`, then a whole-tensor relative
/// Frobenius error.
///
/// **The absolute term is scaled by the reference tensor's own RMS**, not a
/// fixed constant. It has to exist at all — dQ entries pass through zero, and a
/// pure relative bound divides by ~0 — but a *constant* one ends up set by the
/// largest tensor under test and is far too loose for the smallest. At these
/// inputs (`randn(0,1)`, d=128, scale=1/√128) dQ and dK entries have RMS ≈ 0.05
/// while O is O(1) and the LSE is O(6), so a flat `atol = 2e-2` was ~40% of a
/// typical dQ entry: multiplying the kernel's dQ by 1.3 passed every assertion.
/// The real bf16-vs-f32 disagreement on dQ is ~3e-3 relative, from two
/// comparable sources: dQ is *stored* bf16 (unit roundoff 2⁻⁸, ~2.2e-3 RMS),
/// and dS is rounded to bf16 before the dQ matmul (~2.2e-3 after 256 terms
/// accumulate in quadrature — it does not shrink with the sum, because the
/// value only grows as √n either). So the bars here keep ~6× margin while
/// catching anything systematic above a few percent. Tightening further is not
/// worth a round trip: every error this suite exists to catch — a swapped
/// stride, `unpadded_lse = 1`, a permuted return, a double-applied scale — is a
/// 10-100% error, not a 1.5% one.
///
/// The Frobenius check is the one no elementwise bound can substitute for: a
/// uniform scale error spread thinly across every entry sits just under a
/// per-element bar and shows up loudly in the norm.
fn assert_close(got: &Tensor, want: &Tensor, what: &str) -> Result<()> {
    assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
    let g = got.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let w = want.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    let sq: f32 = w.iter().map(|x| x * x).sum();
    // Guarded so an all-zero reference fails as itself rather than as a NaN
    // comparison ten lines below.
    assert!(
        sq > 0.0,
        "{what}: reference is all zeros, the check is vacuous"
    );
    let atol = ATOL * (sq / w.len() as f32).sqrt();

    for (i, (g, w)) in g.iter().zip(&w).enumerate() {
        assert!(
            (g - w).abs() <= atol + RTOL * w.abs(),
            "{what}[{i}]: {g} vs {w} (atol {atol:e})"
        );
    }

    let err: f32 = g.iter().zip(&w).map(|(g, w)| (g - w) * (g - w)).sum();
    let rel = (err / sq).sqrt();
    assert!(
        rel <= FTOL,
        "{what}: relative Frobenius error {rel:e} > {FTOL:e}"
    );
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

/// The `check_*` rejections, which nothing else in this file exercises.
///
/// They are the only thing between a caller's mistake and a silently wrong
/// result: GQA and a causal length mismatch in particular are caught nowhere
/// downstream — `run_mha_backward` aborts on the first and never compares the
/// two lengths for the second. All of them run before any launch, so a test
/// costs a few zeroed allocations.
#[test]
fn rejects_unsupported_shapes() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let scale = (1.0 / 128f64.sqrt()) as f32;
    let bf = |b, h, t, d| Tensor::zeros((b, h, t, d), DType::BF16, &dev);
    let fwd = rs_flash_attn::trainable::flash_attn_fwd_lse;
    let ok = bf(1, 2, 128, 128)?;

    // GQA/MQA. The C backward aborts the process on this, so the forward has to
    // refuse it too or the pair accepts shapes the pair cannot serve.
    assert!(
        fwd(&bf(1, 4, 128, 128)?, &ok, &ok, scale, true).is_err(),
        "GQA"
    );

    // Causal with seqlen_q != seqlen_k: FA2 >= 2.1 aligns the mask
    // bottom-right, and nothing in either ABI compares the two lengths.
    let short_q = bf(1, 2, 64, 128)?;
    assert!(
        fwd(&short_q, &ok, &ok, scale, true).is_err(),
        "causal ragged"
    );
    // ...and the guard is scoped to `causal`, where it is the only case that
    // can be wrong. Non-causal rectangular has no mask to misalign.
    //
    // This one actually launches, so read the result back: the launch is
    // asynchronous and a fault would otherwise surface at the next
    // synchronization point, failing whichever test happened to reach it first.
    // This proves the path is reachable and finite, not that it is numerically
    // right — the inputs are zeros, and its only future consumer is the
    // deferred decode path.
    let (_o, shape, lse) = fwd(&short_q, &ok, &ok, scale, false).expect("non-causal ragged");
    assert_eq!(shape.dims(), &[1, 2, 64, 128], "non-causal ragged shape");
    assert!(
        lse.sum_all()?.to_scalar::<f32>()?.is_finite(),
        "non-causal ragged lse"
    );

    // The one compiled configuration: bf16, head_dim 128, non-empty.
    let f32_q = Tensor::zeros((1, 2, 128, 128), DType::F32, &dev)?;
    assert!(fwd(&f32_q, &ok, &ok, scale, true).is_err(), "dtype");
    let hd64 = bf(1, 2, 128, 64)?;
    assert!(fwd(&hd64, &hd64, &hd64, scale, true).is_err(), "head_dim");
    // A zero-length *view*, not a zero-length allocation: `cuMemAllocAsync(0)`
    // is not documented to succeed, so `Tensor::zeros` here would risk failing
    // in the constructor rather than in the guard under test.
    let empty = ok.narrow(2, 0, 0)?;
    assert!(
        fwd(&empty, &empty, &empty, scale, true).is_err(),
        "empty seqlen"
    );

    // The backward's LSE guards. The LSE is the one buffer whose layout is
    // *hardcoded* in the kernels — there is no stride to pass — so a wrong
    // shape or dtype is read as garbage with no fault at all.
    let bwd = rs_flash_attn::trainable::flash_attn_bwd;
    let (o_storage, o_shape, good_lse) = fwd(&ok, &ok, &ok, scale, true)?;
    let o = as_tensor(o_storage, o_shape);
    assert!(
        bwd(
            &ok,
            &ok,
            &ok,
            &o,
            &good_lse.transpose(1, 2)?,
            &ok,
            scale,
            true
        )
        .is_err(),
        "lse shape"
    );
    assert!(
        bwd(
            &ok,
            &ok,
            &ok,
            &o,
            &good_lse.to_dtype(DType::BF16)?,
            &ok,
            scale,
            true
        )
        .is_err(),
        "lse dtype"
    );
    Ok(())
}

#[test]
fn ragged_query_rows() -> Result<()> {
    // The **M** axis, which none of the cases above reach: the hdim128 backward
    // runs `kBlockM = 64` (`flash_bwd_launch_template.h`), and 128/192/256 are
    // all multiples of 64, so every row guard — the LSE read predicate, the
    // `Clear_OOB_MN` copies, `convert_dQ`'s row bound, and `Clear_dQaccum`'s
    // tail — is compiled but never actually rejects a row. 100 gives a 36-row
    // final block, and is odd against the forward's `kBlockM = 128` too, so
    // `is_even_MN` is false on both axes in both directions.
    //
    // This is what makes the exact `b·h·seqlen_q` LSE allocation an assertion
    // rather than an argument: the backward's read predicate rejects rows 36-63
    // of the second block, which would otherwise land at slice indices 100-127
    // — inside the *next head's* LSE, and past the buffer for the last head.
    //
    // Two bonuses over the cases above. `seqlen_q_rounded` (128) and
    // `ceil(100/64)·64` (128) coincide, so `dq_accum` and `dsoftmax_sum` are
    // written to their exact last element — the tightest fit in the suite,
    // where 128/192/256 all leave slack. And unlike `ragged_seqlen`, this case
    // needs no smem-tier caveat: on an sm_86-class card the backward drops to
    // `kBlockN = 64` but `kBlockM` stays 64, and `100 % 64 = 36` regardless.
    case(1, 2, 100, true)
}
