//! The trainable pair: a forward that keeps the log-sum-exp, and the matching
//! backward. Ours, not upstream's — `lib.rs` is vendored candle code and stays
//! a clean diff against it (see README.md), so everything this project adds
//! lives here.
//!
//! Upstream's `FlashAttn` forward throws the LSE away, which makes it unusable
//! for training: the backward needs it to reconstruct the softmax. These two
//! functions expose the pair the training path actually needs, and nothing
//! else — no alibi, no softcap, no sliding window, no varlen, no GQA. The
//! model uses none of them and each would be a branch no test here covers.
//!
//! # Layout
//!
//! Everything is `(B, H, T, D)` — the project's convention — with a contiguous
//! last dimension. FA2 *never* requires the `(B, T, H, D)` memory order its
//! parameter names suggest: every tensor it touches is addressed through an
//! explicit stride triple, e.g. `make_shape(actual_seqlen_q, h, d)` over
//! `make_stride(params.o_row_stride, params.o_head_stride, _1{})`
//! (`kernels/flash_fwd_kernel.h`, and identically for Q/K/V/dO/dQ/dK/dV in
//! `kernels/flash_bwd_kernel.h`). So the row axis is dim 2 and the head axis is
//! dim 1, and inputs pass through as-is while outputs are allocated contiguous
//! in the same layout. No transpose and no copy, in either direction — which
//! also matters because candle seeds a `CustomOp3` gradient with the
//! *argument's* shape (`backprop.rs`, `grads.or_insert(arg)`), so dQ/dK/dV have
//! to come back `(B, H, T, D)` anyway.
//!
//! # Aborts
//!
//! The C side aborts the process on an unsupported configuration rather than
//! returning a wrong gradient (see `kernels/flash_api.cu`). Every condition it
//! aborts on is checked here first, so ordinary misuse surfaces as a candle
//! `Error`; an abort that does fire means this module and the C guards
//! disagree, which is a bug in one of them.

use core::ffi::{c_int, c_void};

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle::cuda_backend::CudaDType;
use candle::op::BackpropOp;
use candle::{CudaDevice, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::bf16;

use crate::{ffi, round_multiple};

/// The only head dimension this build instantiates — see `KERNEL_FILES` in
/// `build.rs` and the `run_mha_fwd`/`run_mha_bwd` dispatchers in
/// `kernels/flash_api.cu`, which abort on anything else.
const HEAD_DIM: usize = 128;

/// The `(batch, row, head)` stride triple FA2 addresses a `(B, H, T, D)` tensor
/// through. Dim 2 is the row (sequence) axis, dim 1 the head axis; see the
/// module docs. Strides are in elements, as both ABIs require.
///
/// Only valid on a rank-4 layout — call [`check_bhtd`] first.
fn strides_bhtd(l: &Layout) -> (u32, u32, u32) {
    let s = l.stride();
    (s[0] as u32, s[2] as u32, s[1] as u32)
}

fn cuda_storage<'a>(s: &'a Storage, name: &str) -> Result<&'a CudaStorage> {
    match s {
        Storage::Cuda(s) => Ok(s),
        _ => candle::bail!("flash-attn (trainable): {name} must be on a CUDA device"),
    }
}

/// Wrap a freshly allocated device buffer as a tensor with **no** backprop op.
///
/// Detached is what every consumer here wants: the LSE is a stash, and dQ/dK/dV
/// are gradients, so a graph edge would only pin the forward it came from inside
/// candle's `GradStore` — the exact shape of a memory regression this project
/// has already paid for once.
fn detached_tensor<T: CudaDType>(
    slice: CudaSlice<T>,
    shape: impl Into<Shape>,
    dev: &CudaDevice,
) -> Tensor {
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
        shape,
        BackpropOp::none(),
        false,
    )
}

/// Rank / dtype / contiguity checks shared by every bf16 `(B, H, T, D)`
/// operand. Returns `(b, h, t)`; `d` is checked against [`HEAD_DIM`] rather
/// than returned, since callers have nothing to do with any other value.
fn check_bhtd(l: &Layout, dtype: DType, name: &str) -> Result<(usize, usize, usize)> {
    if l.shape().rank() != 4 {
        candle::bail!(
            "flash-attn (trainable): {name} must be rank 4 (B, H, T, D), got {:?}",
            l.shape()
        )
    }
    if dtype != DType::BF16 {
        candle::bail!("flash-attn (trainable): {name} must be bf16, got {dtype:?}")
    }
    let stride = l.stride();
    if stride[3] != 1 {
        candle::bail!("flash-attn (trainable): the last dim of {name} must be contiguous, got strides {stride:?}")
    }
    let (b, h, t, d) = l.shape().dims4()?;
    if d != HEAD_DIM {
        candle::bail!(
            "flash-attn (trainable): {name} has head_dim {d}; this build compiles only {HEAD_DIM}"
        )
    }
    Ok((b, h, t))
}

/// The shape agreement between Q and K/V, shared by both directions so the
/// forward and the backward accept exactly the same inputs. Returns
/// `(b, h, seqlen_q, seqlen_k)`.
fn check_qkv(
    q_l: &Layout,
    k_l: &Layout,
    v_l: &Layout,
    q_dtype: DType,
    k_dtype: DType,
    v_dtype: DType,
    causal: bool,
) -> Result<(usize, usize, usize, usize)> {
    let (b, h, seqlen_q) = check_bhtd(q_l, q_dtype, "q")?;
    let (b_k, h_k, seqlen_k) = check_bhtd(k_l, k_dtype, "k")?;
    if check_bhtd(v_l, v_dtype, "v")? != (b_k, h_k, seqlen_k) {
        candle::bail!(
            "flash-attn (trainable): k {:?} and v {:?} must have the same shape",
            k_l.shape(),
            v_l.shape()
        )
    }
    if b != b_k {
        candle::bail!(
            "flash-attn (trainable): batch mismatch, q {:?} vs k {:?}",
            q_l.shape(),
            k_l.shape()
        )
    }
    if h != h_k {
        // GQA/MQA. Upstream's backward allocates dk/dv at `h` heads and reduces
        // them onto `h_k` after the launch; that reduction is not ported, so
        // the C side aborts (`kernels/flash_api.cu`). Refuse in the forward too
        // — the two halves must take the same shapes or the stash is a trap.
        candle::bail!(
            "flash-attn (trainable): GQA/MQA is not supported (q has {h} heads, k/v have {h_k})"
        )
    }
    if causal && seqlen_q != seqlen_k {
        // FA2 >= 2.1 aligns the causal mask **bottom-right**, via the
        // `actual_seqlen_k - actual_seqlen_q` term in the kernels' col-limit
        // arithmetic. That term vanishes only when the two lengths are equal —
        // the training path, and the only case this crate is tested for. The
        // C backward never compares the two, so without this the mismatch is
        // silent: a plausible loss curve, not a crash.
        candle::bail!(
            "flash-attn (trainable): causal attention requires seqlen_q == seqlen_k \
             (got {seqlen_q} vs {seqlen_k}); FA2's causal mask is bottom-right aligned, \
             so the rectangular (KV-cache) case needs verifying before it is allowed"
        )
    }
    Ok((b, h, seqlen_q, seqlen_k))
}

/// The `(is_causal, window_size_left, window_size_right)` triple both ABIs must
/// land on.
///
/// The forward normalizes the window in Rust (candle's wrapper, `lib.rs`) and
/// the backward normalizes it in C (`kernels/flash_api.cu`), so the same intent
/// needs two different encodings — the one thing that keeps them consistent is
/// that both end at the same `params`.
///
/// Causal, forward: candle's wrapper is fed `left = None, right = Some(0)`,
/// which becomes `left = -1, right = 0`, hence `is_causal = 1`, and then the
/// fill-in `left = seqlen_k`. Causal, backward: `(1, -1, -1)` goes into C,
/// which sets `right = 0` from `is_causal`, derives `is_causal` back from the
/// window, and applies the same fill-in. Both reach `(1, seqlen_k, 0)`.
/// Non-causal is `(0, -1, -1)` on both sides untouched.
fn window(causal: bool, seqlen_k: usize, c_normalizes: bool) -> (c_int, c_int, c_int) {
    match (causal, c_normalizes) {
        (true, false) => (1, seqlen_k as c_int, 0),
        (true, true) => (1, -1, -1),
        (false, _) => (0, -1, -1),
    }
}

/// Forward pass, keeping the softmax log-sum-exp the backward needs.
///
/// `q`, `k`, `v` are `(B, H, T, D)` bf16 on CUDA with a contiguous last dim;
/// `k` and `v` must have the same shape, and with `causal` set `T` must match
/// between q and k/v (see [`check_qkv`]).
///
/// Returns the output storage — *not* a `Tensor` — because the only consumer is
/// `CustomOp3::cuda_fwd`, which must hand candle an owned `CudaStorage`;
/// returning a tensor would force the caller into a `storage_and_layout` +
/// `try_clone` round trip, i.e. a full deep copy of O per layer per forward.
/// The shape it goes with is `(B, H, T, D)`, contiguous. The LSE comes back as
/// a detached `(B, H, T)` f32 tensor, ready to stash for the backward.
pub fn flash_attn_fwd_lse(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(CudaStorage, Shape, Tensor)> {
    let (q_storage, q_l) = q.storage_and_layout();
    let (k_storage, k_l) = k.storage_and_layout();
    let (v_storage, v_l) = v.storage_and_layout();
    let q_s = cuda_storage(&q_storage, "q")?;
    let k_s = cuda_storage(&k_storage, "k")?;
    let v_s = cuda_storage(&v_storage, "v")?;

    let (b, h, seqlen_q, seqlen_k) =
        check_qkv(q_l, k_l, v_l, q.dtype(), k.dtype(), v.dtype(), causal)?;

    let dev = q_s.device();
    let stream = dev.cuda_stream();

    let q_slice = q_s.as_cuda_slice::<bf16>()?.slice(q_l.start_offset()..);
    let k_slice = k_s.as_cuda_slice::<bf16>()?.slice(k_l.start_offset()..);
    let v_slice = v_s.as_cuda_slice::<bf16>()?.slice(v_l.start_offset()..);

    let out_shape = Shape::from((b, h, seqlen_q, HEAD_DIM));
    let out_l = Layout::contiguous(&out_shape);
    let mut dst = unsafe { dev.alloc::<bf16>(out_shape.elem_count())? };

    // Exactly `b * h * seqlen_q`, the shape the kernels index — `(b, h,
    // seqlen_q)` fp32 with stride `(h*seqlen_q, seqlen_q, 1)`, fixed rather
    // than stride-configurable (`flash_fwd_kernel.h`'s `get_lse_tile` on the
    // `!unpadded_lse` branch, and the backward reads it at the same offset).
    // Every access is predicated on the real row index, so an exact allocation
    // is neither overrun nor under-filled. Candle's own dense forward asks for
    // `b * 128 * h * seqlen_q` here, which its varlen forward does not — a
    // quirk, and an expensive one when the LSE is retained across 24 layers.
    let mut softmax_lse = dev.alloc_zeros::<f32>(b * h * seqlen_q)?;

    let (q_batch_stride, q_row_stride, q_head_stride) = strides_bhtd(q_l);
    let (k_batch_stride, k_row_stride, k_head_stride) = strides_bhtd(k_l);
    let (v_batch_stride, v_row_stride, v_head_stride) = strides_bhtd(v_l);
    let (o_batch_stride, o_row_stride, o_head_stride) = strides_bhtd(&out_l);

    let (is_causal, window_size_left, window_size_right) =
        window(causal, seqlen_k, /* c_normalizes */ false);

    let head_size_rounded = round_multiple(HEAD_DIM, 32);
    let seqlen_q_rounded = round_multiple(seqlen_q, 128);
    let seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // MUST be 0, and it is the one argument in either ABI whose wrong value is
    // silent. The varlen value (1) sends `get_lse_tile` down a branch shaped by
    // `params.total_q`, which nothing in `flash_api.cu` ever assigns — so the
    // LSE shape collapses to `(1, h, 0)` with a **zero stride on the head
    // axis** and every head aliases onto head 0. Fully in bounds, so no fault
    // and no abort: just an LSE that is right for one head per batch and
    // garbage for the rest, arriving in the backward as a wrong gradient.
    // (Note that the file this wrapper is modelled on contains both values ~470
    // lines apart, behind the same typo'd `/* upadded_lse */` comment.)
    let unpadded_lse: c_int = 0;

    unsafe {
        let (q_ptr, _guard) = q_slice.device_ptr(&stream);
        let (k_ptr, _guard) = k_slice.device_ptr(&stream);
        let (v_ptr, _guard) = v_slice.device_ptr(&stream);
        let (dst_ptr, _guard) = dst.device_ptr_mut(&stream);
        let (softmax_lse_ptr, _guard) = softmax_lse.device_ptr_mut(&stream);
        ffi::run_mha(
            q_ptr as *const c_void,
            k_ptr as *const c_void,
            v_ptr as *const c_void,
            dst_ptr as *const c_void,
            softmax_lse_ptr as *const c_void,
            /* alibi_slopes_ptr */ std::ptr::null(),
            /* cu_seqlens_q_ptr */ std::ptr::null(),
            /* cu_seqlens_k_ptr */ std::ptr::null(),
            q_batch_stride,
            k_batch_stride,
            v_batch_stride,
            o_batch_stride,
            /* alibi_slopes_batch_stride */ 0,
            q_row_stride,
            k_row_stride,
            v_row_stride,
            o_row_stride,
            q_head_stride,
            k_head_stride,
            v_head_stride,
            o_head_stride,
            /* b */ b as u32,
            /* h */ h as u32,
            /* h_k */ h as u32,
            /* d */ HEAD_DIM as u32,
            /* d_rounded */ head_size_rounded as u32,
            softmax_scale,
            /* seqlen_q */ seqlen_q as u32,
            /* seqlen_k */ seqlen_k as u32,
            /* seqlen_q_rounded */ seqlen_q_rounded as u32,
            /* seqlen_k_rounded */ seqlen_k_rounded as u32,
            /* is_bf16 */ 1,
            is_causal,
            unpadded_lse,
            window_size_left,
            window_size_right,
            /* softcap */ 0.0,
            /* stream_ptr */ stream.cu_stream() as *mut c_void,
        )
    }

    let lse = detached_tensor(softmax_lse, (b, h, seqlen_q), dev);
    Ok((
        CudaStorage::wrap_cuda_slice(dst, dev.clone()),
        out_shape,
        lse,
    ))
}

/// Backward pass. `o` and `lse` must be the outputs of the matching
/// [`flash_attn_fwd_lse`] call, and `softmax_scale`/`causal` must match it too:
/// the kernel reconstructs the softmax from `lse`, so a disagreement is not
/// caught anywhere, it just yields a wrong gradient.
///
/// Returns `(dq, dk, dv)`, each `(B, H, T, D)` bf16 contiguous and detached —
/// `BackpropOp::none()`, so nothing here pins a forward graph in the
/// `GradStore`.
///
/// Not bit-reproducible: dQ is accumulated with `atomicAdd`. Upstream's
/// deterministic path needs a differently-shaped, zeroed `dq_accum` plus a
/// split stride, and the C side refuses it.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    lse: &Tensor,
    d_o: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (q_storage, q_l) = q.storage_and_layout();
    let (k_storage, k_l) = k.storage_and_layout();
    let (v_storage, v_l) = v.storage_and_layout();
    let (o_storage, o_l) = o.storage_and_layout();
    let (lse_storage, lse_l) = lse.storage_and_layout();
    let (do_storage, do_l) = d_o.storage_and_layout();
    let q_s = cuda_storage(&q_storage, "q")?;
    let k_s = cuda_storage(&k_storage, "k")?;
    let v_s = cuda_storage(&v_storage, "v")?;
    let o_s = cuda_storage(&o_storage, "o")?;
    let lse_s = cuda_storage(&lse_storage, "lse")?;
    let do_s = cuda_storage(&do_storage, "d_o")?;

    let (b, h, seqlen_q, seqlen_k) =
        check_qkv(q_l, k_l, v_l, q.dtype(), k.dtype(), v.dtype(), causal)?;
    if check_bhtd(o_l, o.dtype(), "o")? != (b, h, seqlen_q) {
        candle::bail!(
            "flash-attn (trainable): o {:?} must match q {:?}",
            o_l.shape(),
            q_l.shape()
        )
    }
    if check_bhtd(do_l, d_o.dtype(), "d_o")? != (b, h, seqlen_q) {
        candle::bail!(
            "flash-attn (trainable): d_o {:?} must match q {:?}",
            do_l.shape(),
            q_l.shape()
        )
    }
    if lse.dtype() != DType::F32 {
        candle::bail!(
            "flash-attn (trainable): lse must be f32, got {:?}",
            lse.dtype()
        )
    }
    // Shape *and* contiguity, because the kernels' LSE addressing is hardcoded
    // to `(h*seqlen_q, seqlen_q, 1)` — there is no stride to pass.
    if lse_l.shape().rank() != 3
        || lse_l.shape().dims3()? != (b, h, seqlen_q)
        || !lse_l.is_contiguous()
    {
        candle::bail!(
            "flash-attn (trainable): lse must be a contiguous ({b}, {h}, {seqlen_q}) f32 tensor, got {:?} strides {:?}",
            lse_l.shape(),
            lse_l.stride()
        )
    }

    let dev = q_s.device();
    let stream = dev.cuda_stream();

    let q_slice = q_s.as_cuda_slice::<bf16>()?.slice(q_l.start_offset()..);
    let k_slice = k_s.as_cuda_slice::<bf16>()?.slice(k_l.start_offset()..);
    let v_slice = v_s.as_cuda_slice::<bf16>()?.slice(v_l.start_offset()..);
    let o_slice = o_s.as_cuda_slice::<bf16>()?.slice(o_l.start_offset()..);
    let do_slice = do_s.as_cuda_slice::<bf16>()?.slice(do_l.start_offset()..);
    let lse_slice = lse_s.as_cuda_slice::<f32>()?.slice(lse_l.start_offset()..);

    let head_size_rounded = round_multiple(HEAD_DIM, 32);
    let seqlen_q_rounded = round_multiple(seqlen_q, 128);
    let seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Gradients, contiguous in the input layout: candle seeds a `CustomOp3`
    // gradient with `grads.or_insert(arg)`, i.e. a zeros tensor of the
    // *argument's* shape and dtype, so these must come back exactly as q/k/v
    // are. Written in full by the kernel, so no pre-zeroing (upstream uses
    // `torch::empty_like`).
    let dq_shape = Shape::from((b, h, seqlen_q, HEAD_DIM));
    let dkv_shape = Shape::from((b, h, seqlen_k, HEAD_DIM));
    let dq_l = Layout::contiguous(&dq_shape);
    let dkv_l = Layout::contiguous(&dkv_shape);
    let mut dq = unsafe { dev.alloc::<bf16>(dq_shape.elem_count())? };
    let mut dk = unsafe { dev.alloc::<bf16>(dkv_shape.elem_count())? };
    let mut dv = unsafe { dev.alloc::<bf16>(dkv_shape.elem_count())? };

    // Scratch, shapes verbatim from the contract on `run_mha_backward`.
    // `dq_accum` is fp32 `(b, seqlen_q_rounded, h, d_rounded)` and is
    // deliberately *uninitialized*: on the non-deterministic path the
    // preprocess kernel runs with `Clear_dQaccum = true` and zeroes it, which
    // is why upstream allocates it with `torch::empty`. Both layouts are fixed,
    // not stride-configurable. Freed when this call returns.
    let mut dq_accum = unsafe { dev.alloc::<f32>(b * seqlen_q_rounded * h * head_size_rounded)? };
    let mut dsoftmax_sum = unsafe { dev.alloc::<f32>(b * h * seqlen_q_rounded)? };
    // Mandatory even with dropout off: the backward dereferences
    // `params.rng_state[0..2]` unconditionally (the forward's matching write is
    // guarded, which is why only this side needs it). Zeroed; never read for
    // anything that reaches the result.
    let mut rng_state = dev.alloc_zeros::<u64>(2)?;

    let (q_batch_stride, q_row_stride, q_head_stride) = strides_bhtd(q_l);
    let (k_batch_stride, k_row_stride, k_head_stride) = strides_bhtd(k_l);
    let (v_batch_stride, v_row_stride, v_head_stride) = strides_bhtd(v_l);
    let (o_batch_stride, o_row_stride, o_head_stride) = strides_bhtd(o_l);
    let (do_batch_stride, do_row_stride, do_head_stride) = strides_bhtd(do_l);
    let (dq_batch_stride, dq_row_stride, dq_head_stride) = strides_bhtd(&dq_l);
    let (dkv_batch_stride, dkv_row_stride, dkv_head_stride) = strides_bhtd(&dkv_l);

    let (is_causal, window_size_left, window_size_right) =
        window(causal, seqlen_k, /* c_normalizes */ true);

    unsafe {
        let (q_ptr, _guard) = q_slice.device_ptr(&stream);
        let (k_ptr, _guard) = k_slice.device_ptr(&stream);
        let (v_ptr, _guard) = v_slice.device_ptr(&stream);
        let (o_ptr, _guard) = o_slice.device_ptr(&stream);
        let (lse_ptr, _guard) = lse_slice.device_ptr(&stream);
        let (do_ptr, _guard) = do_slice.device_ptr(&stream);
        let (dq_ptr, _guard) = dq.device_ptr_mut(&stream);
        let (dk_ptr, _guard) = dk.device_ptr_mut(&stream);
        let (dv_ptr, _guard) = dv.device_ptr_mut(&stream);
        let (dq_accum_ptr, _guard) = dq_accum.device_ptr_mut(&stream);
        let (dsoftmax_sum_ptr, _guard) = dsoftmax_sum.device_ptr_mut(&stream);
        let (rng_state_ptr, _guard) = rng_state.device_ptr_mut(&stream);
        ffi::run_mha_backward(
            q_ptr as *const c_void,
            k_ptr as *const c_void,
            v_ptr as *const c_void,
            o_ptr as *const c_void,
            lse_ptr as *const c_void,
            do_ptr as *const c_void,
            /* alibi_slopes_ptr */ std::ptr::null(),
            dq_ptr as *const c_void,
            dk_ptr as *const c_void,
            dv_ptr as *const c_void,
            dq_accum_ptr as *const c_void,
            dsoftmax_sum_ptr as *const c_void,
            rng_state_ptr as *const c_void,
            q_batch_stride,
            k_batch_stride,
            v_batch_stride,
            o_batch_stride,
            do_batch_stride,
            dq_batch_stride,
            /* dk_batch_stride */ dkv_batch_stride,
            /* dv_batch_stride */ dkv_batch_stride,
            /* alibi_slopes_batch_stride */ 0,
            q_row_stride,
            k_row_stride,
            v_row_stride,
            o_row_stride,
            do_row_stride,
            dq_row_stride,
            /* dk_row_stride */ dkv_row_stride,
            /* dv_row_stride */ dkv_row_stride,
            q_head_stride,
            k_head_stride,
            v_head_stride,
            o_head_stride,
            do_head_stride,
            dq_head_stride,
            /* dk_head_stride */ dkv_head_stride,
            /* dv_head_stride */ dkv_head_stride,
            /* b */ b as u32,
            /* h */ h as u32,
            /* h_k */ h as u32,
            /* d */ HEAD_DIM as u32,
            /* d_rounded */ head_size_rounded as u32,
            softmax_scale,
            /* seqlen_q */ seqlen_q as u32,
            /* seqlen_k */ seqlen_k as u32,
            /* seqlen_q_rounded */ seqlen_q_rounded as u32,
            /* seqlen_k_rounded */ seqlen_k_rounded as u32,
            /* is_bf16 */ 1,
            is_causal,
            /* deterministic */ 0,
            window_size_left,
            window_size_right,
            /* softcap */ 0.0,
            /* stream_ptr */ stream.cu_stream() as *mut c_void,
        )
    }

    Ok((
        detached_tensor(dq, &dq_shape, dev),
        detached_tensor(dk, &dkv_shape, dev),
        detached_tensor(dv, &dkv_shape, dev),
    ))
}
