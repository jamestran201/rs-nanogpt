ARG ?= default-value

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