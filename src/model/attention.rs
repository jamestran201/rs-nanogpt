use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{VarBuilder, ops::softmax_last_dim};

use super::config::GptConfig;
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
    rope: Rope,
    /// Additive causal mask of shape `(seq_len, seq_len)`: 0 on/below the
    /// diagonal, `-inf` above. Stored fp32; narrowed to `(T, T)` and cast to
    /// the scores' dtype per call.
    causal_mask: Tensor,
    n_head: usize,
    head_dim: usize,
    norm_eps: f32,
}

impl CausalSelfAttention {
    pub fn new(cfg: &GptConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let c = cfg.n_embd;
        let s = (3.0 / cfg.n_embd as f64).sqrt();
        let c_q = Linear::uniform(c, c, s, vb.pp("c_q"))?;
        let c_k = Linear::uniform(c, c, s, vb.pp("c_k"))?;
        let c_v = Linear::uniform(c, c, s, vb.pp("c_v"))?;
        // Residual output projection: zero-init so the block starts as identity.
        let c_proj = Linear::zeros(c, c, vb.pp("c_proj"))?;

        let rope = Rope::from_config(cfg, device)?;
        let causal_mask = build_causal_mask(cfg.sequence_len, device)?;

        Ok(Self {
            c_q,
            c_k,
            c_v,
            c_proj,
            rope,
            causal_mask,
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

        // RoPE then QK-norm (over head_dim), on Q and K only.
        let q = rms_norm(&self.rope.apply(&q)?, self.norm_eps)?;
        let k = rms_norm(&self.rope.apply(&k)?, self.norm_eps)?;

        // Scaled dot-product scores: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T)
        // Output shape: (B, n_head, T, T)
        let scale = 1.0 / (hd as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;

        // Causal mask (-inf above the diagonal) then softmax over keys. The
        // mask is stored fp32 and follows the scores' dtype (no-op on fp32;
        // -inf is representable in bf16).
        let mask = self.causal_mask.i((..t, ..t))?.to_dtype(scores.dtype())?;
        let att = softmax_last_dim(&scores.broadcast_add(&mask)?)?;

        // Mix values, re-assemble heads, project back: (B,T,C).
        let y = att.matmul(&v)?; // (B, n_head, T, head_dim)
        let y = y.transpose(1, 2)?.contiguous()?.reshape((b, t, c))?;
        self.c_proj.forward(&y)
    }
}

fn build_causal_mask(n: usize, device: &Device) -> Result<Tensor> {
    let keep = Tensor::tril2(n, DType::U8, device)?;
    let zeros = Tensor::zeros((n, n), DType::F32, device)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, (n, n), device)?;
    keep.where_cond(&zeros, &neg_inf)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn builder(vm: &VarMap, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_varmap(vm, DType::F32, dev)
    }

    #[test]
    fn forward_preserves_b_t_c() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let attn = CausalSelfAttention::new(&cfg, builder(&vm, &dev), &dev)?;
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
        let attn = CausalSelfAttention::new(&cfg, builder(&vm, &dev), &dev)?;
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
        let attn = CausalSelfAttention::new(&cfg, builder(&vm, &dev), &dev)?;
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
        let attn = CausalSelfAttention::new(&cfg, builder(&vm, &dev), &dev)?;
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
    fn single_token_reduces_to_proj_of_value() -> Result<()> {
        // With T=1: softmax over a single key is 1, and RoPE at position 0 is
        // the identity, so attention is a no-op and forward(x) == c_proj(c_v(x)).
        let dev = Device::Cpu;
        let cfg = tiny_cfg();
        let mut vm = VarMap::new();
        let attn = CausalSelfAttention::new(&cfg, builder(&vm, &dev), &dev)?;
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
