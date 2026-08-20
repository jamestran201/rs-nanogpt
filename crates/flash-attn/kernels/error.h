#pragma once

#include <cstdio>
#include <cstdlib>

// Candle's stand-in for PyTorch's <c10/cuda/CUDAException.h>, so the vendored
// FlashAttention sources compile unmodified. NOT an upstream FA2 file (there is
// no csrc/flash_attn/src/error.h at v2.7.3) — editing it is a candle deviation,
// not a divergence from FlashAttention.
//
// This must abort rather than swallow. `C10_CUDA_KERNEL_LAUNCH_CHECK()` calls
// `cudaGetLastError()`, which *returns and resets* the last non-sticky error —
// so if the value is discarded here, the error is gone: a caller checking
// `cudaGetLastError()` after the launch sequence returns will see cudaSuccess,
// and candle will see success too. That is exactly what happens with a failed
// `cudaFuncSetAttribute` for the >=144 KB of dynamic shared memory the hdim128
// backward requests: the launch fails, dq/dk/dv keep whatever uninitialized
// memory they were allocated with, and training silently continues on garbage
// gradients. These call sites are the only place that value is ever visible.
#define C10_CUDA_CHECK(EXPR)                                             \
  do {                                                                   \
    const cudaError_t __err = (EXPR);                                    \
    if (__err != cudaSuccess) {                                          \
      fprintf(stderr, "rs-flash-attn: CUDA error at %s:%d: %s\n",        \
              __FILE__, __LINE__, cudaGetErrorString(__err));            \
      abort();                                                           \
    }                                                                    \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
