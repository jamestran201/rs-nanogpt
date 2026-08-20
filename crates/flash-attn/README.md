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

Everything else under `kernels/` and `src/`, plus the rest of `build.rs`, is
byte-for-byte upstream. Later phases add backward kernels — record each change here.

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
