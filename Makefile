ARG ?= default-value

.PHONY: build-cpu build-cuda build-metal covhtml covlcov lint bench memprofile

# Release binary for CPU (default backend). Works everywhere; slow for real runs.
build-cpu:
	cargo build --release

# Release binary for NVIDIA GPUs (CUDA) — the remote single-GPU box.
# Requires the CUDA toolkit (nvcc) on PATH; if candle can't autodetect the
# GPU arch, set CUDA_COMPUTE_CAP (e.g. 80=A100, 89=L4/4090, 90=H100).
build-cuda:
	cargo build --release --features cuda

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