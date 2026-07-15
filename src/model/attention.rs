use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{VarBuilder, ops::softmax};

use super::config::GptConfig;
use super::flash_attention::flash_attention;
use super::linear::Linear;
use super::rms_norm::rms_norm;
use super::rope::Rope;

/// Multi-head causal self-attention
///
/// Receives an already-normed residual input `x: (B, T, C)` and returns the
/// attention contribution `(B, T, C)` to be added back by the caller — the
/// pre-norm and the residual add live in the block-composition step, not here.
/// The only norm inside this module is QK-norm.
pub struct CausalSelfAttention {
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    /// RoPE tables, shared from `Gpt::new` (a clone is a cheap handle).
    rope: Rope,
    /// Additive causal mask `(seq_len, seq_len)`: 0 on/below the diagonal,
    /// `-inf` above. Shared from `Gpt::new`; stored fp32, narrowed to `(T, T)`
    /// per call (the flash path does its masked-softmax math in fp32).
    causal_mask: Tensor,
    n_head: usize,
    head_dim: usize,
    norm_eps: f32,
}

impl CausalSelfAttention {
    pub fn new(
        cfg: &GptConfig,
        vb: VarBuilder,
        rope: &Rope,
        causal_mask: &Tensor,
    ) -> Result<Self> {
        let c = cfg.n_embd;
        let s = (3.0 / cfg.n_embd as f64).sqrt();
        let c_q = Linear::uniform(c, c, s, vb.pp("c_q"))?;
        let c_k = Linear::uniform(c, c, s, vb.pp("c_k"))?;
        let c_v = Linear::uniform(c, c, s, vb.pp("c_v"))?;
        // Residual output projection: zero-init so the block starts as identity.
        let c_proj = Linear::zeros(c, c, vb.pp("c_proj"))?;

        Ok(Self {
            c_q,
            c_k,
            c_v,
            c_proj,
            rope: rope.clone(),
            causal_mask: causal_mask.clone(),
            n_head: cfg.n_head,
            head_dim: cfg.head_dim(),
            norm_eps: cfg.norm_eps,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let (nh, hd) = (self.n_head, self.head_dim);

        // Project and split into heads: (B,T,C) -> (B, n_head, T, head_dim).
        let to_heads = |t_in: Tensor| -> Result<Tensor> {
            t_in.reshape((b, t, nh, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = to_heads(self.c_q.forward(x)?)?;
        let k = to_heads(self.c_k.forward(x)?)?;
        let v = to_heads(self.c_v.forward(x)?)?;

        // RoPE then QK-norm (over head_dim), on Q and K only. nanochat norms
        // *before* RoPE; the orders are equivalent because RoPE rotates
        // disjoint 2-D pairs — preserving Σx² and hence the rms — and is
        // linear, so rms_norm(rope(x)) == rope(rms_norm(x)) up to float
        // rounding. Holds only while rms_norm is scale-free; a learnable gain
        // would break it (pinned by `qk_norm_order_is_equivalent` in rope.rs).
        let q = rms_norm(&self.rope.apply(&q)?, self.norm_eps)?;
        let k = rms_norm(&self.rope.apply(&k)?, self.norm_eps)?;

        // Chunked flash attention: same math as `naive_attention` below, but
        // nothing T² is retained in the autograd graph — see
        // `flash_attention.rs` / `writeups/flash-attention-plan.md`.
        let scale = 1.0 / (hd as f64).sqrt();
        let mask = self.causal_mask.i((..t, ..t))?;
        let y = flash_attention(&q, &k, &v, &mask, scale)?;

        // Re-assemble heads, project back: (B,T,C).
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.c_proj.forward(&y)
    }
}

/// Materialized softmax(QKᵀ·scale + mask)V — the reference implementation.
///
/// `q`/`k`/`v` are `(B, n_head, T, head_dim)`; `mask` is the additive fp32
/// causal mask slice `(T, T)`, cast here to the scores' dtype (no-op on fp32;
/// `-inf` is representable in bf16). Retains `(B, n_head, T, T)` tensors in
/// the autograd graph, so it is memory-bound in B·T² — the flash path exists
/// to avoid exactly that; this stays as the parity oracle for its tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn naive_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    // Scaled dot-product scores: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T)
    // Output shape: (B, n_head, T, T)
    let scores = q
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .affine(scale, 0.0)?;

    // Causal mask (-inf above the diagonal) then softmax over keys. This must
    // be the *composed* `candle_nn::ops::softmax` (max/exp/sum primitives):
    // the fused `softmax_last_dim` is `apply_op1_no_bwd` and silently severs
    // the autograd graph — no error, just zero gradient into Q and K — and as
    // the grad-parity oracle for the flash tests this path has to actually
    // differentiate.
    let mask = mask.to_dtype(scores.dtype())?;
    let att = softmax(&scores.broadcast_add(&mask)?, D::Minus1)?;

    // Mix values: (B, n_head, T, head_dim).
    att.matmul(v)
}

pub(crate) fn build_causal_mask(n: usize, device: &Device) -> Result<Tensor> {
    let keep = Tensor::tril2(n, DType::U8, device)?;
    let zeros = Tensor::zeros((n, n), DType::F32, device)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (n, n), device)?;
    keep.where_cond(&zeros, &neg_inf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::new_attn;
    use candle_core::{DType, IndexOp};
    use candle_nn::VarMap;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len: 16,
            n_layer: 1,
            n_head: 2,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    #[test]
    fn forward_preserves_b_t_c() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let attn = new_attn(&cfg, &vm, &dev)?;
        let x = Tensor::randn(0.0f32, 1.0, (2, 5, cfg.n_embd), &dev)?;
        assert_eq!(attn.forward(&x)?.dims(), &[2, 5, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn forward_follows_input_dtype() -> Result<()> {
        // End-to-end low-precision pass through projections, RoPE, QK-norm,
        // mask, and softmax. f16 stands in for the CUDA bf16 path (the CPU
        // backend has an f16 matmul but no bf16 one).
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let attn = new_attn(&cfg, &vm, &dev)?;
        let x = Tensor::randn(0.0f32, 1.0, (2, 5, cfg.n_embd), &dev)?.to_dtype(DType::F16)?;
        let y = attn.forward(&x)?;
        assert_eq!(y.dtype(), DType::F16);
        assert_eq!(y.dims(), &[2, 5, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn zero_init_proj_gives_zero_output() -> Result<()> {
        // c_proj is zero-init, so the whole block is the identity at init:
        // its contribution to the residual stream is all zeros.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let attn = new_attn(&cfg, &vm, &dev)?;
        let x = Tensor::randn(0.0f32, 2.0, (1, 6, cfg.n_embd), &dev)?;
        let y = attn.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(y.iter().all(|v| *v == 0.0), "expected all zeros at init");
        Ok(())
    }

    #[test]
    fn is_causal() -> Result<()> {
        // Perturbing a future token must not change an earlier position's
        // output. Needs a non-zero c_proj, else the output is trivially zero.
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut vm = VarMap::new();
        let attn = new_attn(&cfg, &vm, &dev)?;
        let c = cfg.n_embd;
        vm.set_one("c_proj.weight", Tensor::randn(0.0f32, 0.5, (c, c), &dev)?)?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 6, c), &dev)?;
        let y = attn.forward(&x)?;

        // Replace position 4 (a future token relative to positions 0..=3).
        let mut rows = x.to_vec3::<f32>()?;
        rows[0][4] = (0..c).map(|i| i as f32 * 0.3 - 1.0).collect();
        let x2 = Tensor::new(rows, &dev)?;
        let y2 = attn.forward(&x2)?;

        let pos_i = 3usize;
        let a = y.i((0, pos_i))?.to_vec1::<f32>()?;
        let b = y2.i((0, pos_i))?.to_vec1::<f32>()?;
        for (va, vb) in a.iter().zip(&b) {
            assert!(
                (va - vb).abs() < 1e-5,
                "position {pos_i} changed: {va} vs {vb}"
            );
        }
        Ok(())
    }

    #[test]
    fn backprops_into_q_k_v_projections() -> Result<()> {
        // Regression guard: the fused `softmax_last_dim` this module once used
        // is `apply_op1_no_bwd` — it silently cut the graph, so c_q/c_k never
        // received gradients and attention patterns stayed frozen at init.
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut vm = VarMap::new();
        let attn = new_attn(&cfg, &vm, &dev)?;
        let c = cfg.n_embd;
        vm.set_one("c_proj.weight", Tensor::randn(0.0f32, 0.5, (c, c), &dev)?)?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 6, c), &dev)?;
        let loss = attn.forward(&x)?.sqr()?.sum_all()?;
        let grads = loss.backward()?;

        let vars = vm.data().lock().unwrap();
        for name in ["c_q.weight", "c_k.weight", "c_v.weight"] {
            let var = vars
                .get(name)
                .unwrap_or_else(|| panic!("missing var {name}"));
            let g = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("no grad reached {name}"));
            let sumsq = g.sqr()?.sum_all()?.to_scalar::<f32>()?;
            assert!(sumsq > 0.0, "{name} gradient is identically zero");
        }
        Ok(())
    }

    #[test]
    fn single_token_reduces_to_proj_of_value() -> Result<()> {
        // With T=1: softmax over a single key is 1, and RoPE at position 0 is
        // the identity, so attention is a no-op and forward(x) == c_proj(c_v(x)).
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut vm = VarMap::new();
        let attn = new_attn(&cfg, &vm, &dev)?;
        let c = cfg.n_embd;
        vm.set_one("c_proj.weight", Tensor::randn(0.0f32, 0.5, (c, c), &dev)?)?;

        let x = Tensor::randn(0.0f32, 1.0, (1, 1, c), &dev)?;
        let got = attn.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = attn
            .c_proj
            .forward(&attn.c_v.forward(&x)?)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (g, e) in got.iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "{g} vs {e}");
        }
        Ok(())
    }
}
