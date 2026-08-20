ARG ?= default-value
# Persistent nvcc/CUTLASS artifact cache for the vendored flash-attn kernels.
# Deliberately outside target/ so `cargo clean` does not throw away an hour of nvcc.
FLASH_BUILD_DIR ?= $(CURDIR)/.flash-attn-build

.PHONY: bootstrap build-cpu build-cuda build-nccl build-flash-attn test-flash-attn build-metal covhtml covlcov lint bench memprofile

# Provision a fresh Lambda Labs GPU box: system deps, Rust, and CUDA env vars.
# Idempotent; run `source ~/.bashrc` afterward to pick up the CUDA env.
bootstrap:
	./scripts/bootstrap.sh

# Release binary for CPU (default backend). Works everywhere; slow for real runs.
build-cpu:
	cargo build --release

# Release binary for NVIDIA GPUs (CUDA) — the remote single-GPU box.
# Requires the CUDA toolkit (nvcc) on PATH; if candle can't autodetect the
# GPU arch, set CUDA_COMPUTE_CAP (e.g. 80=A100, 89=L4/4090, 90=H100).
build-cuda:
	cargo build --release --features cuda

# CUDA build with multi-GPU data parallelism (pretrain --gpus N).
# Additionally needs libnccl at build and run time (bootstrap.sh checks).
build-nccl:
	cargo build --release --features nccl

# CUDA build with the vendored FlashAttention-2 kernels (crates/flash-attn).
# Compiles 5 CUDA translation units against CUTLASS, so the first build is
# slow; CANDLE_FLASH_ATTN_BUILD_DIR caches libflashattention.a between builds.
build-flash-attn:
	mkdir -p $(FLASH_BUILD_DIR)
	CANDLE_FLASH_ATTN_BUILD_DIR=$(FLASH_BUILD_DIR) cargo build --release --features flash-attn

# Everything that exercises the vendored kernels: the crate's own parity tests,
# then the root crate with the feature on. Only runnable on a CUDA box.
#
# CANDLE_FLASH_ATTN_BUILD_DIR is not optional on any of these. Unset, build.rs
# falls back to OUT_DIR, which is profile-specific — so `cargo clippy` (debug)
# would re-run nvcc over all 5 CUTLASS translation units even though the release
# build already compiled them. One shared cache dir is safe across profiles: the
# nvcc flags build.rs passes are profile-independent.
test-flash-attn:
	mkdir -p $(FLASH_BUILD_DIR)
	CANDLE_FLASH_ATTN_BUILD_DIR=$(FLASH_BUILD_DIR) cargo test -p rs-flash-attn --release
	CANDLE_FLASH_ATTN_BUILD_DIR=$(FLASH_BUILD_DIR) cargo clippy --all-targets -p rs-flash-attn
	CANDLE_FLASH_ATTN_BUILD_DIR=$(FLASH_BUILD_DIR) cargo test --features flash-attn
	CANDLE_FLASH_ATTN_BUILD_DIR=$(FLASH_BUILD_DIR) cargo clippy --all-targets --features flash-attn

# Release binary for Apple GPUs (Metal), for Mac dev/debug runs.
build-metal:
	cargo build --release --features metal

covhtml:
	cargo llvm-cov --html --open

covlcov:
	cargo llvm-cov --lcov --output-path target/llvm-cov/lcov.info

lint:
	cargo fmt
	cargo clippy --fix --allow-dirty

bench:
	cargo bench

memprofile:
	cargo test --features dhat-heap --release --test $(ARG)