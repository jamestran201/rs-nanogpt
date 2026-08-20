# rs-flash-attn

A vendored copy of **`candle-flash-attn`**, taken verbatim from candle git rev
`39355c6c9187747e360a2d6ec9d67a2a501b2552` (the rev the root crate pins).

## Why it is vendored rather than depended on

Candle ships flash attention **forward-only** — there is no `*bwd*` kernel in
the crate on that rev or on upstream `main` (checked 2026-08-19), and every
fused op in `candle-nn` is `*_no_bwd`, so none of them can appear in a training
graph. This copy exists so the FlashAttention-2 **backward** kernels can be
added to it; see `writeups/flash-attn-vendoring-plan.md` for the plan and
`writeups/mfu-investigation.md` for why it matters.

Vendoring (rather than forking candle) keeps the root crate's `rev` pin
untouched and means there is no fork of a 20-crate workspace to maintain.

## What was changed from upstream

Phase 1 (this commit) changed **only `Cargo.toml`**:

- package renamed `candle-flash-attn` → `rs-flash-attn`, `publish = false`;
- `candle-core` / `candle-nn` path dependencies repointed at the pinned git
  rev, so they resolve to the same `candle-core` instance as the root crate;
- version reset to `0.1.0`.

Then, to make the build tractable on a rented box:

- **`build.rs` — `KERNEL_FILES` trimmed 37 → 3.** Upstream compiles 9 head dims
  × 2 dtypes × 2 causal = 36 forward instantiations. d24 runs exactly one
  configuration: head_dim 128 (`n_embd 1536 / n_head 12`) and bf16
  (`compute_dtype` returns BF16 on every CUDA device). The other 34 are ~18× of
  nvcc time for kernels this project never calls. **The `.cu` files are still on
  disk** — widening is a matter of putting names back in the list.
- **`kernels/flash_api.cu` — `run_mha_fwd` narrowed to match.** Upstream's
  `FP16_SWITCH`/`HEADDIM_SWITCH` nest expands to every combination, so it would
  reference templates that no longer have a translation unit (undefined-symbol
  link error). Now: a hard `abort()` for anything but head_dim 128 / bf16, then
  `BOOL_SWITCH` on causal. Restoring instantiations means restoring this too.
- **`tests/` removed.** Upstream's tests exercise head_dim 8 and fp16, neither of
  which is compiled here, so they no longer link. Phase 4 of the plan adds a
  parity test at the shape this project actually runs.

### Phase 3 — the FlashAttention-2 backward

Added from upstream **v2.7.3** (the tag Phase 2 pinned by SHA-1 diffing all 14
shared headers; `hardware_info.h`, present only from v2.7.2, is the
discriminator):

- `kernels/flash_bwd_preprocess_kernel.h`, `kernels/flash_bwd_kernel.h` —
  **verbatim**, no torch-isms at all.
- `kernels/flash_bwd_launch_template.h` — verbatim **except one line**:
  `#include <c10/cuda/CUDAException.h>` → `#include "error.h"`, which already
  shims `C10_CUDA_CHECK` / `C10_CUDA_KERNEL_LAUNCH_CHECK`, so all 12 call sites
  compile untouched.
- `kernels/flash_bwd_hdim128_bf16{,_causal}_sm80.cu` — **verbatim**.
- `kernels/flash.h` — uncommented the `run_mha_bwd_` declaration candle had
  commented out.
- `build.rs` — `KERNEL_FILES` 3 → 5.

Written here (host-side glue, no CUDA):

- `kernels/flash_api.cu` — `run_mha_bwd` (narrowed to hdim128/bf16 like
  `run_mha_fwd`) and `extern "C" run_mha_backward`, a flat-C mirror of
  upstream's `mha_bwd` minus the PyTorch tensor plumbing. Unsupported
  configurations (GQA, `deterministic`, empty query span, wrong head dim/dtype)
  **abort** rather than return a silently wrong gradient.
- `src/ffi.rs` — the declaration.

**Explicit stream (applies to the forward too).** Upstream's `run_mha`
hardcoded `cudaStream_t stream = 0`. That was *correct*, but only by
coincidence: candle's device stream is `per_thread_stream()`
(`CU_STREAM_PER_THREAD`, set in `BackendDevice::new`), and cudaforge compiles
every file here with `--default-stream per-thread`, so the literal `0` resolved
to the same stream. Both entry points now take a `stream_ptr` and `src/lib.rs`
passes `stream.cu_stream()`, which makes the ordering a property of this code
rather than of a build-tool flag — and keeps it correct if the device is ever
built with `CudaDevice::new_with_stream`, which *does* use a separate
`CU_STREAM_NON_BLOCKING` stream.

**`rng_state` is required, not optional.** The backward dereferences
`params.rng_state[0..2]` unconditionally, even with dropout off — unlike the
forward, whose matching *write* is guarded by `if (Is_dropout && ...)`. Upstream
hides this because `mha_fwd` always allocates the buffer and threads it into
`mha_bwd`. `run_mha_backward` therefore takes a `rng_state_ptr` (2 zeroed `u64`s
in device memory) and refuses null.

**Unsupported configurations abort** rather than degrade: head_dim != 128,
non-bf16, GQA, `deterministic`, `seqlen_q == 0`, null `rng_state_ptr`, rounded
dims that disagree with upstream's rule, and a failed kernel launch.

**`kernels/error.h` now aborts instead of swallowing.** It previously assigned
the status to an unused variable. That mattered more than it looks:
`C10_CUDA_KERNEL_LAUNCH_CHECK()` calls `cudaGetLastError()`, which *returns and
resets* the last non-sticky error — so discarding the value there destroys it,
and a check placed after the launch sequence returns can only ever see
`cudaSuccess`. A failed `cudaFuncSetAttribute` for the ≥144 KB of dynamic shared
memory the hdim128 backward requests would have reached candle as success, with
`dq/dk/dv` left holding uninitialized memory. `error.h` is candle's own de-torch
shim — there is no `csrc/flash_attn/src/error.h` upstream at v2.7.3 (HTTP 404) —
so this edit does not affect the byte-for-byte-upstream claim above. It fixes
the forward as well.

Everything else under `kernels/` and `src/`, plus the rest of `build.rs`, is
byte-for-byte upstream. Later phases — record each change here.

## Licensing

candle is MIT OR Apache-2.0 (both texts included). The CUDA sources under
`kernels/` are FlashAttention-2, © Tri Dao, BSD-3-Clause; their copyright
headers are intact.

## Building

CUDA-only, and slow — nvcc compiles 37 translation units against CUTLASS
(auto-fetched by `cudaforge` at a pinned commit). Set
`CANDLE_FLASH_ATTN_BUILD_DIR` to a persistent path on the box so
`libflashattention.a` caches between builds; without it every `cargo build`
recompiles the kernels.

    make build-flash-attn
