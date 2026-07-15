//! Chunked ("flash") attention with a hand-written backward.
//!
//! The naive path retains ~3-4 `(B, n_head, T, T)` tensors per layer in the
//! autograd graph until backward — the d24 memory ceiling. This module
//! computes the same math in query-row chunks so nothing T² is ever retained:
//! the forward keeps only the output `O` and a per-row log-sum-exp `LSE`, and
//! the backward reconstructs each chunk's softmax exactly from `Q`, `K`, and
//! `LSE` (`P = exp(S − LSE)`), applying the standard flash-attention gradient
//! formulas. Same result as the naive path up to float rounding — a memory
//! optimization, not an approximation.
//!
//! Everything here runs on *detached* tensors (`track_op() == false`), so no
//! graph is built and each chunk's transients free as the loop advances. The
//! softmax math runs in fp32 regardless of the compute dtype; the matmuls run
//! in the input dtype (tensor cores on CUDA).
//!
//! See `writeups/flash-attention-plan.md` for the full design.

use std::sync::Mutex;

use candle_core::{
    CpuStorage, CudaStorage, CustomOp3, D, DType, Layout, MetalStorage, Result, Shape, Storage,
    Tensor, bail,
};

/// Query rows per chunk. Transient memory per chunk is `~4 × B·H·CHUNK·T` fp32
/// in the backward; at d24 (H=12, T=2048, B=16) that is ~0.8 GB. Raising this
/// trades memory for fewer kernel launches.
pub(crate) const FLASH_CHUNK: usize = 128;

/// Forward pass, chunked over query rows.
///
/// `q`/`k`/`v`: `(B, n_head, T, head_dim)`, `mask`: additive causal mask slice
/// `(T, T)` (`0` on/below the diagonal, `-inf` above; any dtype — upcast to
/// fp32 here). Returns `O (B, n_head, T, head_dim)` in the input dtype and
/// `LSE (B, n_head, T)` fp32, the per-row softmax log-normalizer the backward
/// needs to reconstruct the softmax without storing it.
fn flash_attn_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
    chunk: usize,
) -> Result<(Tensor, Tensor)> {
    let (_b, _h, t, _hd) = q.dims4()?;
    let dtype = q.dtype();
    // Strided views only, no .contiguous() copies: on every backend candle's
    // matmul accepts transposed and row-narrowed operands directly (uniform
    // batch stride, clean inner 2-D), so materializing them would only add
    // device write traffic.
    let k_t = k.transpose(2, 3)?; // (B,H,hd,T) view
    let mask = mask.to_dtype(DType::F32)?;

    let n_chunks = t.div_ceil(chunk);
    let mut o_chunks = Vec::with_capacity(n_chunks);
    let mut lse_chunks = Vec::with_capacity(n_chunks);
    let mut c0 = 0;
    while c0 < t {
        let len = chunk.min(t - c0);
        let q_c = q.narrow(2, c0, len)?; // (B,H,len,hd)

        // Masked scores for this row block, fp32: (B,H,len,T). Full key width
        // for simplicity — masked columns softmax to exact zeros. Narrowing K
        // to the causal prefix per chunk (as the fused kernels do) would
        // halve the score compute and is a possible follow-up.
        let s = q_c.matmul(&k_t)?.to_dtype(DType::F32)?.affine(scale, 0.0)?;
        let s = s.broadcast_add(&mask.narrow(0, c0, len)?)?;

        // Row-wise softmax via the log-normalizer, so it can be stashed for
        // the backward: P = exp(S − LSE).
        let lse = s.log_sum_exp(D::Minus1)?; // (B,H,len)
        let p = s.broadcast_sub(&lse.unsqueeze(D::Minus1)?)?.exp()?;

        o_chunks.push(p.to_dtype(dtype)?.matmul(v)?); // (B,H,len,hd)
        lse_chunks.push(lse);
        c0 += len;
    }
    Ok((Tensor::cat(&o_chunks, 2)?, Tensor::cat(&lse_chunks, 2)?))
}

/// Backward pass: the standard flash-attention gradients, chunked like the
/// forward, reconstructing each chunk's softmax from the stashed `lse`.
///
/// Per chunk `c`: `D_c = rowsum(dO_c ∘ O_c)` (the softmax-VJP correction),
/// `dV += P_cᵀ dO_c`, `dP = dO_c Vᵀ`, `dS = P_c ∘ (dP − D_c)` and, with the
/// score scale folded in once, `dQ_c = dS K` and `dK += dSᵀ q_c`. `dK`/`dV`
/// accumulate in fp32 across chunks; all grads return in the input dtype.
#[allow(clippy::too_many_arguments)]
fn flash_attn_bwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    lse: &Tensor,
    d_o: &Tensor,
    mask: &Tensor,
    scale: f64,
    chunk: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, h, t, hd) = q.dims4()?;
    let dtype = q.dtype();
    // Strided views only — see the matching note in `flash_attn_fwd`.
    let k_t = k.transpose(2, 3)?; // (B,H,hd,T) view
    let v_t = v.transpose(2, 3)?; // (B,H,hd,T) view
    let mask = mask.to_dtype(DType::F32)?;

    let mut dq_chunks = Vec::with_capacity(t.div_ceil(chunk));
    let mut dk = Tensor::zeros((b, h, t, hd), DType::F32, q.device())?;
    let mut dv = Tensor::zeros((b, h, t, hd), DType::F32, q.device())?;
    let mut c0 = 0;
    while c0 < t {
        let len = chunk.min(t - c0);
        let q_c = q.narrow(2, c0, len)?; // (B,H,len,hd)
        let o_c = o.narrow(2, c0, len)?;
        let do_c = d_o.narrow(2, c0, len)?;
        let lse_c = lse.narrow(2, c0, len)?;

        // Reconstruct the chunk's softmax exactly as the forward computed it.
        let s = q_c.matmul(&k_t)?.to_dtype(DType::F32)?.affine(scale, 0.0)?;
        let s = s.broadcast_add(&mask.narrow(0, c0, len)?)?;
        let p = s.broadcast_sub(&lse_c.unsqueeze(D::Minus1)?)?.exp()?; // (B,H,len,T) fp32

        // dV += Pᵀ dO: (B,H,T,len) @ (B,H,len,hd).
        let p_dt = p.to_dtype(dtype)?;
        let dv_c = p_dt.transpose(2, 3)?.matmul(&do_c)?;
        dv = (dv + dv_c.to_dtype(DType::F32)?)?;

        // dS = P ∘ (dP − D), fp32; masked columns have P = 0 so their
        // gradient is exactly zero. Fold the score scale in here once — it
        // then covers both dQ and dK.
        let dp = do_c.matmul(&v_t)?.to_dtype(DType::F32)?; // (B,H,len,T)
        let d_c =
            (do_c.to_dtype(DType::F32)? * o_c.to_dtype(DType::F32)?)?.sum_keepdim(D::Minus1)?; // (B,H,len,1)
        let ds = ((p * dp.broadcast_sub(&d_c)?)?).affine(scale, 0.0)?;
        let ds_dt = ds.to_dtype(dtype)?;

        // dQ_c = dS K: (B,H,len,T) @ (B,H,T,hd).
        dq_chunks.push(ds_dt.matmul(k)?);
        // dK += dSᵀ q_c: (B,H,T,len) @ (B,H,len,hd).
        let dk_c = ds_dt.transpose(2, 3)?.matmul(&q_c)?;
        dk = (dk + dk_c.to_dtype(DType::F32)?)?;
        c0 += len;
    }
    Ok((
        Tensor::cat(&dq_chunks, 2)?,
        dk.to_dtype(dtype)?,
        dv.to_dtype(dtype)?,
    ))
}

/// Flash attention as a single autograd node: `(q, k, v) → O` with the
/// hand-written backward above. Drop-in for `naive_attention` (same inputs,
/// same output, same dtype behavior) minus the retained T² tensors.
///
/// `q`/`k`/`v`: `(B, n_head, T, head_dim)` with RoPE/QK-norm already applied;
/// `mask`: additive causal `(T, T)` slice; `scale = 1/sqrt(head_dim)`.
pub(crate) fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    flash_attention_chunked(q, k, v, mask, scale, FLASH_CHUNK)
}

fn flash_attention_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
    chunk: usize,
) -> Result<Tensor> {
    let op = FlashAttnOp {
        q: q.detach(),
        k: k.detach(),
        v: v.detach(),
        mask: mask.detach(),
        scale,
        chunk,
        lse: Mutex::new(None),
    };
    q.apply_op3(k, v, op)
}

/// The `CustomOp3` bridge. Candle's custom-op forward API is storage-level
/// (per-backend `*_fwd` receiving raw `&CpuStorage`/`&CudaStorage`/…), which
/// would block the device-agnostic tensor code above — so the op instead
/// stashes *detached clones of the tensor handles* at construction (cheap Arc
/// clones of the same storage the graph tensors use) and every backend hook
/// runs the one shared tensor-level forward on those, ignoring the passed-in
/// storages. Only the output makes one storage round-trip
/// (`storage_and_layout` + `try_clone`, a deep copy of O alone).
///
/// The stash is also what carries `LSE` from forward to backward:
/// `apply_op3` stores this op as an `Arc` in the graph node and backprop
/// calls `bwd` on the *same instance*. Each forward builds a fresh op, so
/// nothing is shared across micro-batches; eval graphs that never run
/// backward drop the op (and stash) when the graph drops.
struct FlashAttnOp {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor,
    scale: f64,
    chunk: usize,
    lse: Mutex<Option<Tensor>>,
}

impl FlashAttnOp {
    fn fwd(&self) -> Result<(Storage, Shape)> {
        let (o, lse) = flash_attn_fwd(
            &self.q, &self.k, &self.v, &self.mask, self.scale, self.chunk,
        )?;
        *self.lse.lock().expect("flash-attn lse mutex poisoned") = Some(lse);
        let o = o.contiguous()?;
        let (storage, layout) = o.storage_and_layout();
        Ok((storage.try_clone(layout)?, o.shape().clone()))
    }
}

impl CustomOp3 for FlashAttnOp {
    fn name(&self) -> &'static str {
        "flash-attn"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        match self.fwd()? {
            (Storage::Cpu(s), shape) => Ok((s, shape)),
            _ => bail!("flash-attn cpu_fwd produced a non-cpu tensor"),
        }
    }

    fn cuda_fwd(
        &self,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
        _: &CudaStorage,
        _: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match self.fwd()? {
            (Storage::Cuda(s), shape) => Ok((s, shape)),
            _ => bail!("flash-attn cuda_fwd produced a non-cuda tensor"),
        }
    }

    fn metal_fwd(
        &self,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
        _: &MetalStorage,
        _: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        match self.fwd()? {
            (Storage::Metal(s), shape) => Ok((s, shape)),
            _ => bail!("flash-attn metal_fwd produced a non-metal tensor"),
        }
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        d_o: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        // Detach *all five* inputs: they arrive tracked, and any op touching
        // a tracked tensor would rebuild graph nodes here — the returned
        // grads would then pin every chunk intermediate inside the GradStore
        // until the optimizer step, defeating the whole point.
        let lse = self
            .lse
            .lock()
            .expect("flash-attn lse mutex poisoned")
            .clone();
        let Some(lse) = lse else {
            bail!("flash-attn backward called before forward")
        };
        let (dq, dk, dv) = flash_attn_bwd(
            &q.detach(),
            &k.detach(),
            &v.detach(),
            &o.detach(),
            &lse,
            &d_o.detach(),
            &self.mask,
            self.scale,
            self.chunk,
        )?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::attention::{build_causal_mask, naive_attention};
    use crate::test_support::assert_close;
    use candle_core::{Device, Var};

    fn qkv_mask(
        b: usize,
        h: usize,
        t: usize,
        hd: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;
        let mask = build_causal_mask(t, &dev)?;
        Ok((q, k, v, mask))
    }

    #[test]
    fn fwd_matches_naive_across_shapes() -> Result<()> {
        // (b, h, t, hd, chunk): divisible, ragged, t < chunk, t = 1, chunk = 1.
        for (b, h, t, hd, chunk) in [
            (2, 3, 16, 4, 8),
            (1, 2, 7, 4, 5),
            (1, 1, 3, 4, 128),
            (1, 1, 1, 4, 128),
            (1, 2, 4, 4, 1),
        ] {
            let (q, k, v, mask) = qkv_mask(b, h, t, hd)?;
            let scale = 1.0 / (hd as f64).sqrt();
            let want = naive_attention(&q, &k, &v, &mask, scale)?;
            let (got, lse) = flash_attn_fwd(&q, &k, &v, &mask, scale, chunk)?;
            assert_close(&got, &want, 1e-5, &format!("fwd t={t} chunk={chunk}"))?;
            assert_eq!(lse.dims(), &[b, h, t]);
            assert_eq!(lse.dtype(), DType::F32);
        }
        Ok(())
    }

    #[test]
    fn op_grads_match_naive_autograd() -> Result<()> {
        // Full integration: gradients through apply_op3 + backward() must
        // match autograd through the naive path — same Vars, two graphs.
        let (b, h, t, hd, chunk) = (2, 2, 7, 4, 3);
        let dev = Device::Cpu;
        let (q0, k0, v0, mask) = qkv_mask(b, h, t, hd)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let w = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;

        let q = Var::from_tensor(&q0)?;
        let k = Var::from_tensor(&k0)?;
        let v = Var::from_tensor(&v0)?;

        let out_ref = naive_attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), &mask, scale)?;
        let grads_ref = (&out_ref * &w)?.sum_all()?.backward()?;

        let out = flash_attention_chunked(
            q.as_tensor(),
            k.as_tensor(),
            v.as_tensor(),
            &mask,
            scale,
            chunk,
        )?;
        assert_close(&out, &out_ref, 1e-5, "op fwd")?;
        let grads = (&out * &w)?.sum_all()?.backward()?;

        for (var, name) in [(&q, "dq"), (&k, "dk"), (&v, "dv")] {
            let got = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("no {name}"));
            let want = grads_ref.get(var.as_tensor()).unwrap();
            assert_eq!(got.dtype(), var.as_tensor().dtype(), "{name} dtype");
            assert_close(got, want, 1e-5, name)?;
        }
        Ok(())
    }

    #[test]
    fn op_follows_f16_dtype() -> Result<()> {
        // f16 is the CPU stand-in for the CUDA bf16 path: output and grads
        // must come back f16 (the fp32 softmax island stays internal), and
        // stay close to the fp32 reference within f16 rounding.
        let (b, h, t, hd) = (1, 2, 6, 4);
        let dev = Device::Cpu;
        let (q0, k0, v0, mask) = qkv_mask(b, h, t, hd)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let w32 = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;

        let out_ref = naive_attention(&q0, &k0, &v0, &mask, scale)?;

        let q = Var::from_tensor(&q0.to_dtype(DType::F16)?)?;
        let k = Var::from_tensor(&k0.to_dtype(DType::F16)?)?;
        let v = Var::from_tensor(&v0.to_dtype(DType::F16)?)?;
        let out =
            flash_attention_chunked(q.as_tensor(), k.as_tensor(), v.as_tensor(), &mask, scale, 4)?;
        assert_eq!(out.dtype(), DType::F16);
        assert_close(&out, &out_ref, 2e-2, "f16 fwd vs f32 naive")?;

        let w = w32.to_dtype(DType::F16)?;
        let grads = (&out * &w)?.sum_all()?.backward()?;
        for (var, name) in [(&q, "dq"), (&k, "dk"), (&v, "dv")] {
            let g = grads
                .get(var.as_tensor())
                .unwrap_or_else(|| panic!("no {name}"));
            assert_eq!(g.dtype(), DType::F16, "{name} dtype");
            assert_eq!(g.dims(), &[b, h, t, hd], "{name} shape");
        }
        Ok(())
    }

    #[test]
    fn op_instances_are_independent_across_graphs() -> Result<()> {
        // Grad-accumulation shape: each forward builds a fresh op (fresh LSE
        // stash); two forward/backward rounds on different data must not
        // interfere.
        let (b, h, t, hd, chunk) = (1, 2, 8, 4, 3);
        let scale = 1.0 / (hd as f64).sqrt();
        let mut grads = Vec::new();
        let mut refs = Vec::new();
        for _ in 0..2 {
            let (q0, k0, v0, mask) = qkv_mask(b, h, t, hd)?;
            let q = Var::from_tensor(&q0)?;
            let out = flash_attention_chunked(q.as_tensor(), &k0, &v0, &mask, scale, chunk)?;
            grads.push((
                out.sum_all()?
                    .backward()?
                    .get(q.as_tensor())
                    .unwrap()
                    .clone(),
                q,
            ));
            let out_ref = naive_attention(&q0, &k0, &v0, &mask, scale)?;
            refs.push((out_ref, k0, v0, mask, q0));
        }
        // Both rounds got correct, round-specific gradients.
        for (i, ((dq, _q), (_out_ref, k0, v0, mask, q0))) in grads.iter().zip(&refs).enumerate() {
            let qv = Var::from_tensor(q0)?;
            let out = naive_attention(qv.as_tensor(), k0, v0, mask, scale)?;
            let want = out.sum_all()?.backward()?;
            assert_close(
                dq,
                want.get(qv.as_tensor()).unwrap(),
                1e-5,
                &format!("round {i} dq"),
            )?;
        }
        Ok(())
    }

    #[test]
    fn bwd_matches_autograd_through_naive() -> Result<()> {
        // Reference grads: autograd through the naive path with
        // loss = sum(out ∘ w), whose gradient w.r.t. out is exactly w.
        for (b, h, t, hd, chunk) in [(2, 2, 8, 4, 3), (1, 2, 7, 4, 5), (1, 1, 1, 4, 2)] {
            let dev = Device::Cpu;
            let (q0, k0, v0, mask) = qkv_mask(b, h, t, hd)?;
            let scale = 1.0 / (hd as f64).sqrt();
            let w = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;

            let q = Var::from_tensor(&q0)?;
            let k = Var::from_tensor(&k0)?;
            let v = Var::from_tensor(&v0)?;
            let out = naive_attention(q.as_tensor(), k.as_tensor(), v.as_tensor(), &mask, scale)?;
            let loss = (&out * &w)?.sum_all()?;
            let grads = loss.backward()?;

            let (o, lse) = flash_attn_fwd(&q0, &k0, &v0, &mask, scale, chunk)?;
            let (dq, dk, dv) = flash_attn_bwd(&q0, &k0, &v0, &o, &lse, &w, &mask, scale, chunk)?;

            let what = format!("t={t} chunk={chunk}");
            assert_close(
                &dq,
                grads.get(q.as_tensor()).unwrap(),
                1e-5,
                &format!("dq {what}"),
            )?;
            assert_close(
                &dk,
                grads.get(k.as_tensor()).unwrap(),
                1e-5,
                &format!("dk {what}"),
            )?;
            assert_close(
                &dv,
                grads.get(v.as_tensor()).unwrap(),
                1e-5,
                &format!("dv {what}"),
            )?;
        }
        Ok(())
    }
}
