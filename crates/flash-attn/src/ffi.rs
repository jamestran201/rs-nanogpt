use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_mha(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        cu_seqlens_q_ptr: *const i32,
        cu_seqlens_k_ptr: *const i32,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,
        unpadded_lse: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,
        // The CUDA stream to launch on. Not a race fix — candle's stream is
        // `per_thread_stream()` and cudaforge builds these kernels with
        // `--default-stream per-thread`, so the previous hardcoded 0 already
        // resolved to the same stream. Passing it explicitly makes the
        // ordering a property of the code rather than of a build-tool flag.
        stream_ptr: *mut c_void,
    );

    /// Backward pass. Mirrors upstream `mha_bwd`; see the contract on
    /// `run_mha_backward` in `kernels/flash_api.cu` for the workspace shapes
    /// (`dq_accum_ptr`, `dsoftmax_sum_ptr`) the caller must allocate.
    ///
    /// `rng_state_ptr` must point at 2 zeroed `u64`s in device memory: the
    /// backward dereferences it unconditionally even with dropout off (the
    /// forward's matching write *is* guarded, which is why only this side needs
    /// it). Aborts the process on an unsupported configuration (head_dim != 128,
    /// non-bf16, GQA, `deterministic`, empty query span, a null `rng_state_ptr`,
    /// rounded dims that disagree with upstream's rule) rather than returning a
    /// silently wrong gradient. A failed kernel launch also aborts, from the
    /// `C10_CUDA_CHECK` sites inside the launch template (see `kernels/error.h`)
    /// — it cannot be caught after the fact, because those macros reset the
    /// error via `cudaGetLastError()`.
    pub(crate) fn run_mha_backward(
        q_ptr: *const c_void,
        k_ptr: *const c_void,
        v_ptr: *const c_void,
        o_ptr: *const c_void,
        softmax_lse_ptr: *const c_void,
        dout_ptr: *const c_void,
        alibi_slopes_ptr: *const c_void,

        dq_ptr: *const c_void,
        dk_ptr: *const c_void,
        dv_ptr: *const c_void,

        dq_accum_ptr: *const c_void,
        dsoftmax_sum_ptr: *const c_void,
        rng_state_ptr: *const c_void,

        q_batch_stride: u32,
        k_batch_stride: u32,
        v_batch_stride: u32,
        o_batch_stride: u32,
        do_batch_stride: u32,
        dq_batch_stride: u32,
        dk_batch_stride: u32,
        dv_batch_stride: u32,
        alibi_slopes_batch_stride: u32,

        q_row_stride: u32,
        k_row_stride: u32,
        v_row_stride: u32,
        o_row_stride: u32,
        do_row_stride: u32,
        dq_row_stride: u32,
        dk_row_stride: u32,
        dv_row_stride: u32,

        q_head_stride: u32,
        k_head_stride: u32,
        v_head_stride: u32,
        o_head_stride: u32,
        do_head_stride: u32,
        dq_head_stride: u32,
        dk_head_stride: u32,
        dv_head_stride: u32,

        b: u32,
        h: u32,
        h_k: u32,
        d: u32,
        d_rounded: u32,
        softmax_scale: f32,

        seqlen_q: u32,
        seqlen_k: u32,
        seqlen_q_rounded: u32,
        seqlen_k_rounded: u32,

        is_bf16: c_int,
        is_causal: c_int,
        deterministic: c_int,

        window_size_left: c_int,
        window_size_right: c_int,

        softcap: f32,
        stream_ptr: *mut c_void,
    );
}
