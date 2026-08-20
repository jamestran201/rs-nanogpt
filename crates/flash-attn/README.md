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

Everything under `kernels/`, `src/`, and `tests/`, plus `build.rs`, is byte-for-byte
upstream. Later phases add backward kernels — record each change here.

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
