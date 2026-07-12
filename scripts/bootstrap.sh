#!/usr/bin/env bash
#
# Bootstrap a fresh Lambda Labs GPU instance for building rs-nanogpt.
#
# Installs the system build toolchain, the Rust toolchain (via rustup), and
# appends the CUDA environment variables to ~/.bashrc. Idempotent: safe to
# re-run — already-installed pieces are skipped and the ~/.bashrc block is
# only written once.
#
# Usage:  ./scripts/bootstrap.sh   (or: make bootstrap)
#
# NOTE: a script cannot mutate the environment of the shell that launched it.
# After it finishes, run `source ~/.bashrc` (or open a new shell) so the CUDA
# vars take effect in your current session.

set -euo pipefail

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }

# --- System build dependencies ------------------------------------------------
# candle's CUDA + tokenizer builds need a C/C++ toolchain, OpenSSL headers,
# cmake, and libclang (for bindgen).
log "Installing system build dependencies (apt)"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  pkg-config \
  libssl-dev \
  cmake \
  git \
  curl \
  clang \
  libclang-dev

# --- Rust toolchain -----------------------------------------------------------
if command -v cargo >/dev/null 2>&1; then
  log "Rust already installed ($(cargo --version)); skipping rustup"
else
  log "Installing Rust toolchain (rustup, stable)"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# Make cargo available to the rest of this script.
# shellcheck disable=SC1091
source "${CARGO_HOME:-$HOME/.cargo}/env"

# --- CUDA environment variables -----------------------------------------------
# Put nvcc on PATH and the CUDA runtime libs on the loader path so candle can
# find the toolkit at build/run time. Written once, guarded by a marker.
BASHRC="$HOME/.bashrc"
MARKER="# >>> rs-nanogpt cuda env >>>"
if grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
  log "CUDA env vars already present in $BASHRC; skipping"
else
  log "Adding CUDA env vars to $BASHRC"
  cat >> "$BASHRC" <<'EOF'

# >>> rs-nanogpt cuda env >>>
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# <<< rs-nanogpt cuda env <<<
EOF
fi

log "Bootstrap complete."
printf '    Run \033[1msource ~/.bashrc\033[0m (or open a new shell), then \033[1mmake build-cuda\033[0m.\n'
