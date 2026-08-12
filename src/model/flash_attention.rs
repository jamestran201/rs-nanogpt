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

use std::sync::{Mutex, OnceLock};

use candle_core::{
    CpuStorage, CudaStorage, CustomOp3, D, DType, Layout, MetalStorage, Result, Shape, Storage,
    Tensor, bail,
};

/// Query rows per chunk. Backward transient per chunk is ~4 concurrent
/// `B·H·CHUNK·W` fp32 tensors, `W ≤ T` the chunk's causal prefix width; the
/// last (widest) chunk at d24 (H=12, T=2048, B=16) peaks around ~0.8 GB.
/// Raising this trades memory for fewer, larger kernel launches; the
/// `FLASH_CHUNK` env var overrides it at startup for benchmarking (see
/// `flash_chunk`, which resolves it for both the training and the cached
/// entry point).
pub(crate) const FLASH_CHUNK: usize = 128;

/// Forward pass, chunked over query rows.
///
/// `q`: `(B, n_head, T_q, head_dim)`; `k`/`v`: `(B, n_head, T_kv, head_dim)`
/// with `T_kv == kv_offset + T_q`; `mask`: additive causal mask slice
/// `(T_q, T_kv)` (`0` on/below the diagonal, `-inf` above; any dtype — upcast
/// to fp32 here). Returns `O (B, n_head, T_q, head_dim)` in the input dtype and
/// `LSE (B, n_head, T_q)` fp32, the per-row softmax log-normalizer the backward
/// needs to reconstruct the softmax without storing it.
///
/// `kv_offset` is the absolute position of `q`'s first row: on the KV-cache
/// decode path the keys are the whole cached prefix while the queries are only
/// the new span, so a query row's causal prefix is `kv_offset` keys wider than
/// its index within `q`. It is `0` on the training path, where the two coincide.
fn flash_attn_fwd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
    kv_offset: usize,
    chunk: usize,
) -> Result<(Tensor, Tensor)> {
    let (_b, _h, t, _hd) = q.dims4()?;
    let dtype = q.dtype();
    // Strided views only, no .contiguous() copies: on every backend candle's
    // matmul accepts transposed and narrowed operands directly (uniform
    // batch stride, clean inner 2-D) — including the per-chunk causal-prefix
    // narrows below — so materializing them would only add device traffic.
    //
    // Score scale folded into Q once, outside the loop: S = (scale·Q)Kᵀ + M.
    // One pass over the small Q replaces a full S-sized affine per chunk;
    // the backward folds identically, keeping its reconstruction of S
    // bit-for-bit aligned with this forward.
    let q = q.affine(scale, 0.0)?;
    let k_t = k.transpose(2, 3)?; // (B,H,hd,T) view
    let mask = mask.to_dtype(DType::F32)?;

    let n_chunks = t.div_ceil(chunk);
    let mut o_chunks = Vec::with_capacity(n_chunks);
    let mut lse_chunks = Vec::with_capacity(n_chunks);
    let mut c0 = 0;
    while c0 < t {
        let len = chunk.min(t - c0);
        // Rows [c0, c0+len) sit at absolute positions [kv_offset+c0, kv_offset+c0+len),
        // so the widest key any of them may see is kv_offset+c0+len-1.
        let w = kv_offset + c0 + len;
        let q_c = q.narrow(2, c0, len)?; // (B,H,len,hd)

        // Masked scores for this row block, fp32: (B,H,len,w) — narrowed to
        // the causal key prefix. Keys ≥ w would be masked to exp(-inf) = 0
        // anyway, so skipping them changes nothing: the row max and sum-exp
        // (hence LSE and P) are bit-identical to the full-width versions,
        // while every matmul and elementwise pass shrinks ~2× on average.
        // One chain so the pre-mask scores drop at statement end instead of
        // living (shadowed) to the end of the iteration.
        let s = q_c
            .matmul(&k_t.narrow(3, 0, w)?)?
            .to_dtype(DType::F32)?
            .broadcast_add(&mask.narrow(0, c0, len)?.narrow(1, 0, w)?)?;

        // Row-wise softmax factored so the exponentials are computed once:
        // E = exp(S − max) is the unnormalized softmax, LSE = log(ΣE) + max
        // (value-identical to `log_sum_exp`, which is composed of these same
        // ops but discards E, forcing a second sub + exp for P), and the ÷ΣE
        // normalization lands on the (len, hd) output instead of the
        // (len, w) score block. ΣE ≥ 1 — the row max is causally visible and
        // contributes exp(0) — so the division is safe; it runs in fp32
        // before the downcast. The backward is untouched: it reconstructs
        // P = exp(S − LSE) from the stash, which is already one sub + exp.
        let m = s.max_keepdim(D::Minus1)?; // (B,H,len,1)
        let e = s.broadcast_sub(&m)?.exp()?; // (B,H,len,w)
        drop(s); // dead once E exists; frees an S-sized chunk pre-matmul
        let se = e.sum_keepdim(D::Minus1)?; // (B,H,len,1)
        lse_chunks.push((se.log()? + &m)?.squeeze(D::Minus1)?); // (B,H,len)

        let o_raw = e.to_dtype(dtype)?.matmul(&v.narrow(2, 0, w)?)?; // (B,H,len,hd)
        o_chunks.push(
            o_raw
                .to_dtype(DType::F32)?
                .broadcast_div(&se)?
                .to_dtype(dtype)?,
        );
        c0 += len;
    }
    Ok((Tensor::cat(&o_chunks, 2)?, Tensor::cat(&lse_chunks, 2)?))
}

/// Backward pass: the standard flash-attention gradients, chunked like the
/// forward, reconstructing each chunk's softmax from the stashed `lse`.
///
/// Per chunk `c`: `D_c = rowsum(dO_c ∘ O_c)` (the softmax-VJP correction),
/// `dV += P_cᵀ dO_c`, `dP = dO_c Vᵀ`, `dS = P_c ∘ (dP − D_c)` — unscaled;
/// the score scale rides in the pre-scaled operands instead of on any
/// S-sized tensor: `dQ_c = dS (scale·K)` and `dK += dSᵀ (scale·Q)_c`.
/// Everything key-sided is narrowed to the chunk's causal prefix, like the
/// forward: beyond it `P = 0` hence `dS = 0`, so the skipped `dK`/`dV`
/// columns are exact zeros. `dK`/`dV` accumulate in fp32 prefix accumulators
/// that grow with the chunks (see `accum_prefix`); all grads return in the
/// input dtype.
///
/// Deliberately takes no `kv_offset`: training never uses a KV cache and the
/// cached path never backprops, so this only ever runs the `kv_offset == 0`
/// case — which keeps its prefix width `w = c0 + len` bit-for-bit aligned with
/// the forward whose softmax it reconstructs.
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
    let (_b, _h, t, _hd) = q.dims4()?;
    let dtype = q.dtype();
    // Strided views only — see the matching note in `flash_attn_fwd`.
    //
    // Scale folding, mirroring the forward: S is rebuilt from q_s = scale·Q
    // (bit-for-bit the forward's S), so dK = dSᵀ·q_s carries the scale via
    // q_s and dQ = dS·k_s carries it via k_s = scale·K — dS itself stays
    // unscaled, which is what deletes the per-chunk S-sized affine passes.
    let q_s = q.affine(scale, 0.0)?;
    let k_s = k.affine(scale, 0.0)?;
    let k_t = k.transpose(2, 3)?; // (B,H,hd,T) view (unscaled; scale is in q_s)
    let v_t = v.transpose(2, 3)?; // (B,H,hd,T) view
    let mask = mask.to_dtype(DType::F32)?;

    let mut dq_chunks = Vec::with_capacity(t.div_ceil(chunk));
    let mut dk: Option<Tensor> = None; // fp32, width = the last chunk's prefix
    let mut dv: Option<Tensor> = None;
    let mut c0 = 0;
    while c0 < t {
        let len = chunk.min(t - c0);
        let w = c0 + len; // causal prefix width, as in the forward
        let q_c = q_s.narrow(2, c0, len)?; // (B,H,len,hd), pre-scaled
        let o_c = o.narrow(2, c0, len)?;
        let do_c = d_o.narrow(2, c0, len)?;
        let lse_c = lse.narrow(2, c0, len)?;

        // Reconstruct the chunk's softmax exactly as the forward computed it
        // (same causal-prefix narrowing, same op sequence). One chain, and
        // explicit drops below: several S-sized fp32 tensors coexist in this
        // loop body, so freeing each at last use instead of iteration end
        // trims the transient peak by a couple of ~chunk-sized tensors.
        let s = q_c
            .matmul(&k_t.narrow(3, 0, w)?)?
            .to_dtype(DType::F32)?
            .broadcast_add(&mask.narrow(0, c0, len)?.narrow(1, 0, w)?)?;
        let p = s.broadcast_sub(&lse_c.unsqueeze(D::Minus1)?)?.exp()?; // (B,H,len,w) fp32
        drop(s);

        // dV_c = Pᵀ dO: (B,H,w,len) @ (B,H,len,hd).
        let p_dt = p.to_dtype(dtype)?;
        let dv_c = p_dt.transpose(2, 3)?.matmul(&do_c)?.to_dtype(DType::F32)?;
        drop(p_dt);
        dv = Some(accum_prefix(dv, dv_c, c0, len)?);

        // dS = P ∘ (dP − D), fp32, unscaled (see the folding note above);
        // masked columns have P = 0 so their gradient is exactly zero.
        let dp = do_c.matmul(&v_t.narrow(3, 0, w)?)?.to_dtype(DType::F32)?; // (B,H,len,w)
        let d_c =
            (do_c.to_dtype(DType::F32)? * o_c.to_dtype(DType::F32)?)?.sum_keepdim(D::Minus1)?; // (B,H,len,1)
        let ds = (p * dp.broadcast_sub(&d_c)?)?; // consumes p
        drop(dp);
        let ds_dt = ds.to_dtype(dtype)?;

        // dQ_c = dS (scale·K): (B,H,len,w) @ (B,H,w,hd).
        dq_chunks.push(ds_dt.matmul(&k_s.narrow(2, 0, w)?)?);
        // dK_c = dSᵀ (scale·Q)_c: (B,H,w,len) @ (B,H,len,hd).
        let dk_c = ds_dt.transpose(2, 3)?.matmul(&q_c)?.to_dtype(DType::F32)?;
        dk = Some(accum_prefix(dk, dk_c, c0, len)?);
        c0 += len;
    }
    let dk = dk.expect("t >= 1 guarantees at least one chunk");
    let dv = dv.expect("t >= 1 guarantees at least one chunk");
    Ok((
        Tensor::cat(&dq_chunks, 2)?,
        dk.to_dtype(dtype)?,
        dv.to_dtype(dtype)?,
    ))
}

/// Fold one chunk's fp32 `(B,H,w,hd)` dK/dV contribution into the growing
/// prefix accumulator. Chunks arrive in increasing width and the previous
/// accumulator's width is always exactly `c0` (chunks advance by `len`), so
/// the contribution splits into an overlap `[0, c0)` added into the prefix
/// and the new diagonal-block columns `[c0, c0+len)` appended after it.
fn accum_prefix(acc: Option<Tensor>, contrib: Tensor, c0: usize, len: usize) -> Result<Tensor> {
    match acc {
        None => Ok(contrib),
        Some(prev) => Tensor::cat(
            &[
                (prev + contrib.narrow(2, 0, c0)?)?,
                contrib.narrow(2, c0, len)?,
            ],
            2,
        ),
    }
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
    flash_attention_chunked(q, k, v, mask, scale, flash_chunk())
}

/// The chunk size both entry points run at. Read once per process: lets
/// chunk-size benchmarks run without a rebuild (`FLASH_CHUNK=256 ... pretrain
/// ...`). Absent, unparsable, or zero values fall back to the compiled default.
fn flash_chunk() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("FLASH_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&c| c > 0)
            .unwrap_or(FLASH_CHUNK)
    })
}

/// Inference attention over a KV cache: same math as [`flash_attention`], but
/// with no autograd node, no LSE stash, and query rows starting at absolute
/// position `kv_offset`.
///
/// `q` is the new span `(B, n_head, T_q, head_dim)`; `k`/`v` are the whole
/// cached prefix `(B, n_head, kv_offset + T_q, head_dim)`; `mask` is the
/// `(T_q, T_kv)` rectangle of the causal mask starting at row `kv_offset`.
///
/// The `CustomOp3` wrapper is bypassed entirely — there is no backward to
/// serve, so there is nothing to stash and no `storage.try_clone` round-trip on
/// the output. `detach` shares storage and only clears the graph identity, so it
/// allocates nothing; what it buys is that the `T_q × T_kv` chunk transients
/// inside the kernel are untracked and free as the loop advances, instead of
/// being retained by an autograd graph that would only ever be dropped.
pub(crate) fn flash_attention_no_grad(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
    kv_offset: usize,
) -> Result<Tensor> {
    // K/V must be exactly the causal prefix of q's rows: the kernel derives each
    // chunk's key width from `kv_offset` alone, so a mismatch would silently
    // attend over the wrong prefix rather than fail a shape check.
    let t_q = q.dim(2)?;
    let want = kv_offset + t_q;
    if k.dim(2)? != want || v.dim(2)? != want {
        bail!(
            "flash-attn no-grad: k/v length {}/{} must equal kv_offset ({kv_offset}) + T_q ({t_q}) = {want}",
            k.dim(2)?,
            v.dim(2)?
        );
    }
    let (o, _lse) = flash_attn_fwd(
        &q.detach(),
        &k.detach(),
        &v.detach(),
        mask,
        scale,
        kv_offset,
        flash_chunk(),
    )?;
    Ok(o)
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
        // kv_offset 0: this is the training path, which never caches — keeping
        // it explicit so the backward's matching omission reads as a decision.
        let (o, lse) = flash_attn_fwd(
            &self.q, &self.k, &self.v, &self.mask, self.scale, 0, self.chunk,
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
    use candle_core::{Device, IndexOp, Var};

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
        // (b, h, t, hd, chunk): divisible, ragged, t < chunk, t = 1,
        // chunk = 1, many chunks with a ragged tail.
        for (b, h, t, hd, chunk) in [
            (2, 3, 16, 4, 8),
            (1, 2, 7, 4, 5),
            (1, 1, 3, 4, 128),
            (1, 1, 1, 4, 128),
            (1, 2, 4, 4, 1),
            (2, 2, 19, 4, 4),
        ] {
            let (q, k, v, mask) = qkv_mask(b, h, t, hd)?;
            let scale = 1.0 / (hd as f64).sqrt();
            let want = naive_attention(&q, &k, &v, &mask, scale)?;
            let (got, lse) = flash_attn_fwd(&q, &k, &v, &mask, scale, 0, chunk)?;
            assert_close(&got, &want, 1e-5, &format!("fwd t={t} chunk={chunk}"))?;
            assert_eq!(lse.dims(), &[b, h, t]);
            assert_eq!(lse.dtype(), DType::F32);
        }
        Ok(())
    }

    #[test]
    fn fwd_with_kv_offset_matches_naive() -> Result<()> {
        // The prefix-width arithmetic is the one place an off-by-one hides, so
        // sweep ragged (t0, t_q) against chunk sizes above, below and equal to
        // t_q. Queries are the span at [t0, t0+t_q); keys/values are the whole
        // prefix [0, t0+t_q); the mask is the same rectangle attention slices.
        let dev = Device::Cpu;
        let (b, h, hd) = (2usize, 3, 4);
        let scale = 1.0 / (hd as f64).sqrt();
        for (t0, t_q) in [(0usize, 4usize), (3, 1), (5, 3), (7, 1)] {
            let t_kv = t0 + t_q;
            let q = Tensor::randn(0f32, 1.0, (b, h, t_q, hd), &dev)?;
            let k = Tensor::randn(0f32, 1.0, (b, h, t_kv, hd), &dev)?;
            let v = Tensor::randn(0f32, 1.0, (b, h, t_kv, hd), &dev)?;
            let mask = build_causal_mask(t_kv, &dev)?.i((t0..t_kv, ..t_kv))?;
            let want = naive_attention(&q, &k, &v, &mask, scale)?;
            for chunk in [1usize, 2, 128] {
                let what = format!("t0={t0} t_q={t_q} chunk={chunk}");
                let (got, lse) = flash_attn_fwd(&q, &k, &v, &mask, scale, t0, chunk)?;
                assert_close(&got, &want, 1e-5, &what)?;
                assert_eq!(lse.dims(), &[b, h, t_q], "lse dims {what}");
            }
        }
        Ok(())
    }

    #[test]
    fn no_grad_entry_rejects_a_kv_length_mismatch() -> Result<()> {
        // A wrong K/V length must bail rather than silently attend over the
        // wrong prefix — the kernel takes the width from kv_offset, not from K.
        let dev = Device::Cpu;
        let (b, h, hd, t0, t_q) = (1usize, 2, 4, 3, 2);
        let scale = 1.0 / (hd as f64).sqrt();
        let q = Tensor::randn(0f32, 1.0, (b, h, t_q, hd), &dev)?;
        let mask = build_causal_mask(t0 + t_q, &dev)?.i((t0..t0 + t_q, ..t0 + t_q))?;
        let ok = Tensor::randn(0f32, 1.0, (b, h, t0 + t_q, hd), &dev)?;
        let short = Tensor::randn(0f32, 1.0, (b, h, t0 + t_q - 1, hd), &dev)?;

        assert!(flash_attention_no_grad(&q, &ok, &ok, &mask, scale, t0).is_ok());
        assert!(flash_attention_no_grad(&q, &short, &ok, &mask, scale, t0).is_err());
        assert!(flash_attention_no_grad(&q, &ok, &short, &mask, scale, t0).is_err());
        Ok(())
    }

    #[test]
    fn no_grad_output_carries_no_gradient() -> Result<()> {
        // Pins the `detach`, which is the memory property: the T² interior must
        // not be retained by a graph the inference path will never back through.
        let dev = Device::Cpu;
        let (b, h, t, hd) = (1usize, 2, 4, 4);
        let scale = 1.0 / (hd as f64).sqrt();
        let q = Var::from_tensor(&Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, t, hd), &dev)?;
        let mask = build_causal_mask(t, &dev)?;

        let out = flash_attention_no_grad(q.as_tensor(), &k, &v, &mask, scale, 0)?;
        let grads = out.sum_all()?.backward()?;
        assert!(grads.get(q.as_tensor()).is_none(), "q kept a gradient");
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
        for (b, h, t, hd, chunk) in [
            (2, 2, 8, 4, 3),
            (1, 2, 7, 4, 5),
            (1, 1, 1, 4, 2),
            (2, 2, 19, 4, 4),
        ] {
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

            let (o, lse) = flash_attn_fwd(&q0, &k0, &v0, &mask, scale, 0, chunk)?;
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
