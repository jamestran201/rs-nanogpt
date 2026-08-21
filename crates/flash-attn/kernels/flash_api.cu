#include <cstdio>
#include <cstdlib>

#include "kernels.h"
#include "kernel_helpers.h"
#include "flash_fwd_launch_template.h"
#include "flash_bwd_launch_template.h"

// NARROWED (rs-nanogpt): only head_dim 128 / bf16 is instantiated in this build
// (see KERNEL_FILES in build.rs). The upstream FP16_SWITCH/HEADDIM_SWITCH nest
// expands to every combination, so leaving it would reference templates that no
// longer have a translation unit — an undefined-symbol link error. Aborting
// rather than falling back: there is no other kernel here to fall back to, and a
// silent wrong-shape dispatch would be far worse than a loud stop.
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  if (params.d != 128 || !params.is_bf16) {
    fprintf(stderr,
            "rs-flash-attn: this build compiles only head_dim 128 / bf16 "
            "(got head_dim %d, %s). Restore the instantiations in build.rs "
            "and this dispatcher to widen it.\n",
            params.d, params.is_bf16 ? "bf16" : "fp16");
    abort();
  }
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      run_mha_fwd_<cutlass::bfloat16_t, 128, Is_causal>(params, stream);
  });
}

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int is_bf16,
    int is_causal,
    int unpadded_lse,

    int window_size_left,
    int window_size_right,

    float softcap,
    void *stream_ptr
) {
    Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr; // used for `return_softmax`.
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.num_splits = 1;
    params.unpadded_lse = unpadded_lse;

    // Launch on the caller's stream rather than the compiled-in default.
    //
    // This is NOT a race fix: candle's device stream is `per_thread_stream()`
    // (CU_STREAM_PER_THREAD, cuda_backend/device.rs `BackendDevice::new`), and
    // cudaforge compiles every file here with `--default-stream per-thread`,
    // so the upstream `cudaStream_t stream = 0` already resolved to that same
    // stream. It was correct, but only by the coincidence of a build-tool flag.
    // Passing the stream explicitly makes the ordering a property of the code:
    // it survives a cudaforge flag change and stays correct if the device is
    // ever built with `CudaDevice::new_with_stream`, which uses a genuinely
    // separate CU_STREAM_NON_BLOCKING stream.
    run_mha_fwd(params, (cudaStream_t)stream_ptr);
}

// NARROWED, exactly as `run_mha_fwd` above: only head_dim 128 / bf16 is
// instantiated (see KERNEL_FILES in build.rs).
void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  if (params.d != 128 || !params.is_bf16) {
    fprintf(stderr,
            "rs-flash-attn: this build compiles only head_dim 128 / bf16 "
            "(got head_dim %d, %s). Restore the instantiations in build.rs "
            "and this dispatcher to widen it.\n",
            params.d, params.is_bf16 ? "bf16" : "fp16");
    abort();
  }
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
      run_mha_bwd_<cutlass::bfloat16_t, 128, Is_causal>(params, stream);
  });
}

// Flat-C mirror of upstream `mha_bwd` (flash_api.cpp), minus the PyTorch
// tensor plumbing: the caller owns every allocation and passes raw pointers.
//
// Caller-allocated workspaces, shapes taken verbatim from upstream:
//   dq_accum_ptr    fp32, (b, seqlen_q_rounded, h, d_rounded), UNINITIALIZED —
//                   the preprocess kernel clears it on the non-deterministic
//                   path (`flash_bwd_dot_do_o_kernel<true, ...>`), which is why
//                   upstream allocates it with `torch::empty`, not `zeros`.
//   dsoftmax_sum_ptr fp32, (b, h, seqlen_q_rounded), uninitialized.
//   rng_state_ptr   2 x uint64, device memory, MUST be non-null and should be
//                   zeroed. The backward dereferences `params.rng_state[0..2]`
//                   *unconditionally* (flash_bwd_kernel.h, the `flash::Dropout
//                   dropout(...)` construction) — unlike the forward, where the
//                   corresponding write is guarded by `if (Is_dropout && ...)`.
//                   Dropout is off here so the values are never used, but the
//                   load still happens and a null pointer faults. Upstream
//                   likewise always passes a real buffer: `mha_fwd` allocates
//                   `torch::empty({2}, kInt64)` and threads it into `mha_bwd`.
// where seqlen_q_rounded = round_up(seqlen_q, 128),
//       seqlen_k_rounded = round_up(seqlen_k, 128),
//       d_rounded        = d <= 192 ? round_up(d, 32) : 256   (note: the
//       forward uses round_up(d, 32) unconditionally; the two agree at d = 128).
// These three are validated against that rule below, since they index the
// caller's workspaces.
//
// Two further contracts on the inputs: `softmax_lse_ptr` is (b, h, seqlen_q)
// fp32 — seqlen_q, NOT seqlen_q_rounded (flash_bwd_kernel.h, the
// non-`unpadded_lse` branch) — and q/k/v/o/dout/dq/dk/dv are all bf16 with a
// contiguous last dimension.
//
// dq/dk/dv are written in full and need no pre-zeroing (upstream uses
// `torch::empty_like`). All strides are in ELEMENTS, not bytes.
extern "C" void run_mha_backward(
    // inputs from the forward
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *dout_ptr,
    void *alibi_slopes_ptr,

    // outputs
    void *dq_ptr,
    void *dk_ptr,
    void *dv_ptr,

    // caller-allocated scratch (see above)
    void *dq_accum_ptr,
    void *dsoftmax_sum_ptr,
    void *rng_state_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t do_batch_stride,
    uint32_t dq_batch_stride,
    uint32_t dk_batch_stride,
    uint32_t dv_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    uint32_t do_row_stride,
    uint32_t dq_row_stride,
    uint32_t dk_row_stride,
    uint32_t dv_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    uint32_t do_head_stride,
    uint32_t dq_head_stride,
    uint32_t dk_head_stride,
    uint32_t dv_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int is_bf16,
    int is_causal,
    int deterministic,

    int window_size_left,
    int window_size_right,

    float softcap,
    void *stream_ptr
) {
    // Unsupported configurations abort rather than degrade: each would produce
    // a silently wrong gradient, which is the worst failure mode there is.
    if (h_k == 0) {
        fprintf(stderr, "rs-flash-attn bwd: h_k must be > 0\n");
        abort();
    }
    if (h % h_k != 0) {
        fprintf(stderr, "rs-flash-attn bwd: h (%u) must be divisible by h_k (%u)\n", h, h_k);
        abort();
    }
    if (h != h_k) {
        // Upstream allocates dk_expanded/dv_expanded at h heads and reduces
        // them onto h_k with a sum over the group axis after the launch. That
        // reduction is not ported: the model this build serves is pure MHA.
        fprintf(stderr, "rs-flash-attn bwd: GQA/MQA (h %u != h_k %u) is not supported\n", h, h_k);
        abort();
    }
    if (deterministic) {
        // Needs an (nsplits, b, seqlen_q_rounded, h, d_rounded) *zeroed*
        // dq_accum plus dq_accum_split_stride = its dim-0 stride, where
        // nsplits = ceil(num_sm / (b * h)). Untested here, so refused.
        fprintf(stderr, "rs-flash-attn bwd: deterministic mode is not implemented\n");
        abort();
    }
    if (rng_state_ptr == nullptr) {
        // See the workspace contract above: dereferenced unconditionally.
        fprintf(stderr, "rs-flash-attn bwd: rng_state_ptr must be non-null\n");
        abort();
    }
    // The kernels index the caller's workspaces with these, so a caller that
    // rounds differently writes out of bounds into whatever candle allocated
    // next. Cheap to check, and the failure it prevents is unfindable.
    {
        auto round_up = [](uint32_t x, uint32_t m) { return (x + m - 1) / m * m; };
        uint32_t want_d = d <= 192 ? round_up(d, 32) : 256;
        if (d_rounded != want_d || seqlen_q_rounded != round_up(seqlen_q, 128)
            || seqlen_k_rounded != round_up(seqlen_k, 128)) {
            fprintf(stderr,
                    "rs-flash-attn bwd: rounded dims disagree with upstream's rule "
                    "(got d_rounded %u seqlen_q_rounded %u seqlen_k_rounded %u; "
                    "want %u %u %u)\n",
                    d_rounded, seqlen_q_rounded, seqlen_k_rounded,
                    want_d, round_up(seqlen_q, 128), round_up(seqlen_k, 128));
            abort();
        }
    }
    if (seqlen_q == 0) {
        // Upstream skips the launch and zeroes dk/dv/softmax_d instead. The
        // training path never produces an empty span, so refuse rather than
        // carry an untested branch.
        fprintf(stderr, "rs-flash-attn bwd: seqlen_q must be > 0\n");
        abort();
    }

    Flash_bwd_params params;
    memset(&params, 0, sizeof(params));

    // --- the set_params_fprop half -------------------------------------
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.; // probability to keep; < 1 would enable Is_dropout
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.p_ptr = nullptr; // used for `return_softmax`
    params.seqused_k = nullptr;

    // Causal/window normalization, in upstream's exact order: the first three
    // lines are `mha_bwd`'s, the rest is `set_params_fprop`'s. Note is_causal is
    // *derived* from the window, not taken from the argument — and the two
    // fill-in lines matter: without them a local/sliding-window request leaves
    // one side at -1 while LOCAL_SWITCH turns Is_local on, and the kernel then
    // computes its block bounds from that -1 and drops dK/dV contributions.
    if (is_causal) { window_size_right = 0; }
    if (window_size_left >= (int)seqlen_k) { window_size_left = -1; }
    if (window_size_right >= (int)seqlen_k) { window_size_right = -1; }
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = (int)seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = (int)seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.num_splits = 1;
    params.unpadded_lse = false;

    // --- the set_params_dgrad half -------------------------------------
    params.do_ptr = dout_ptr;
    params.do_batch_stride = do_batch_stride;
    params.do_row_stride = do_row_stride;
    params.do_head_stride = do_head_stride;

    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;
    params.dq_batch_stride = dq_batch_stride;
    params.dk_batch_stride = dk_batch_stride;
    params.dv_batch_stride = dv_batch_stride;
    params.dq_row_stride = dq_row_stride;
    params.dk_row_stride = dk_row_stride;
    params.dv_row_stride = dv_row_stride;
    params.dq_head_stride = dq_head_stride;
    params.dk_head_stride = dk_head_stride;
    params.dv_head_stride = dv_head_stride;

    params.dq_accum_ptr = dq_accum_ptr;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;
    params.dsoftmax_sum = dsoftmax_sum_ptr;

    params.deterministic = false;
    // Non-deterministic only; upstream sets this to dq_accum.stride(0) when
    // deterministic, which is refused above.
    params.dq_accum_split_stride = 0;

    // Dereferenced unconditionally by the backward even with dropout off —
    // see the workspace contract above.
    params.rng_state = (uint64_t *)rng_state_ptr;

    run_mha_bwd(params, (cudaStream_t)stream_ptr);

    // Backstop only. Launch-time failures are caught inside the launch
    // template by C10_CUDA_CHECK / C10_CUDA_KERNEL_LAUNCH_CHECK (see error.h,
    // which aborts) — and they must be, because those macros call
    // cudaGetLastError(), which resets the last non-sticky error, so by the
    // time control returns here it is already gone. What can still surface at
    // this point is a sticky/asynchronous error from earlier work on the
    // stream. Cheap, so worth keeping; just not the thing standing between a
    // failed launch and garbage gradients.
    cudaError_t pending_err = cudaGetLastError();
    if (pending_err != cudaSuccess) {
        fprintf(stderr, "rs-flash-attn bwd: pending CUDA error: %s\n",
                cudaGetErrorString(pending_err));
        abort();
    }
}
