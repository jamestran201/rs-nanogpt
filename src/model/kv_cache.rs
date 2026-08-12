//! Per-layer key/value cache for incremental decoding.
//!
//! Without a cache, generating token `i` re-forwards the whole context, so an
//! `N`-token completion from a `P`-token prompt costs `≈ N·P + N²/2`
//! token-forwards; with one it costs `P + N`. The cache holds each layer's
//! *post-RoPE, post-QK-norm* keys and the values, so a decode step does zero
//! re-work on the tokens it has already seen — it projects one token, appends
//! its K/V, and attends over the running prefix.
//!
//! Buffers are preallocated to `capacity` and filled left to right by an
//! in-place `slice_set`, so a decode step allocates nothing per layer and the
//! cache never holds autograd graph nodes.
//!
//! Shaped after `candle_nn::kv_cache`, but with two deliberate differences:
//! overflowing `capacity` is an error rather than silent growth (RoPE's tables
//! and the causal mask are both sized to `sequence_len`, so growing past it only
//! moves the failure somewhere more confusing), and [`KvCache::reset`] rewinds
//! the length while *keeping* the allocation — the crop path in `generate_ids`
//! resets once per token, and reallocating there would cost a ~300 MB
//! alloc/free per token at d24.
//!
//! See `writeups/kv-cache-plan.md`.

use candle_core::{Result, Tensor, bail};

use super::config::GptConfig;

/// One layer's K or V buffer: `(B, n_head, capacity, head_dim)`, filled left to
/// right.
///
/// Allocated on the first append — the batch size and dtype are not known before
/// then (dtype follows the device: bf16 on CUDA, fp32 elsewhere). Holds no
/// length of its own: the position is a single counter on [`KvCache`], so a
/// forward that fails partway through the block loop cannot leave the layers
/// disagreeing about where the next span goes.
struct Cache {
    buf: Option<Tensor>,
    capacity: usize,
}

impl Cache {
    fn new(capacity: usize) -> Self {
        Self {
            buf: None,
            capacity,
        }
    }

    /// Write `src` `(B, n_head, T, head_dim)` at time index `at`, and return the
    /// running prefix `(B, n_head, at + T, head_dim)`.
    ///
    /// The returned tensor is a *view into the buffer* (at `at + T == capacity`
    /// it is the buffer itself, since a full-range `narrow` clones the handle),
    /// so it has to be consumed before the next append — attention uses it
    /// immediately.
    ///
    /// No bounds check and no shape/dtype cross-check against the allocated
    /// buffer: [`KvCache::check_room`] runs once per forward before any layer
    /// writes, and `slice_set` itself bails on a dtype, batch, head or head_dim
    /// mismatch — all of which are unreachable anyway, since the shapes come
    /// from one model on one device within a single generation call.
    fn append(&mut self, src: &Tensor, at: usize) -> Result<Tensor> {
        let (b, h, t, hd) = src.dims4()?;
        if self.buf.is_none() {
            let shape = (b, h, self.capacity, hd);
            self.buf = Some(Tensor::zeros(shape, src.dtype(), src.device())?);
        }
        let buf = self.buf.as_ref().expect("just allocated");
        // In place, and deliberately not backprop-safe: the cached path never
        // differentiates. `slice_set` needs both sides contiguous; `src` already
        // is (K is a fresh `rms_norm(rope(k))`, V comes out of a `.contiguous()`),
        // so this call is free when it is already true.
        buf.slice_set(&src.contiguous()?, 2, at)?;
        buf.narrow(2, 0, at + t)
    }
}

/// One layer's K/V pair. Opaque: it travels as a `&mut` from [`KvCache`] into
/// the attention module and has no public surface of its own.
pub(crate) struct LayerKv {
    k: Cache,
    v: Cache,
}

impl LayerKv {
    /// Append this span's keys and values at time index `at`, returning the
    /// running `(K, V)` prefixes to attend over.
    #[cfg_attr(not(test), allow(dead_code))] // consumed by Part D (attention).
    pub(crate) fn append(&mut self, k: &Tensor, v: &Tensor, at: usize) -> Result<(Tensor, Tensor)> {
        Ok((self.k.append(k, at)?, self.v.append(v, at)?))
    }
}

/// The whole model's cache: one [`LayerKv`] per block plus the single position
/// counter they all share.
///
/// That counter is the reason there is no length per layer. `Gpt::forward` reads
/// it once as `t0`, threads it to every layer, and advances it *after* the block
/// loop, so a failure partway through (an OOM on a first `Tensor::zeros`, say)
/// leaves `pos` untouched and the next forward simply overwrites the partial
/// writes at the same offsets. With a length per layer, the same failure would
/// silently rotate every later key in the lagging layers to the wrong position.
pub struct KvCache {
    layers: Vec<LayerKv>,
    len: usize,
    capacity: usize,
}

impl KvCache {
    /// `capacity` is the longest context this cache will hold. It may not exceed
    /// `cfg.sequence_len`, which is what the RoPE tables and the causal mask are
    /// sized to.
    pub fn new(cfg: &GptConfig, capacity: usize) -> Result<Self> {
        if capacity > cfg.sequence_len {
            bail!(
                "kv-cache capacity ({capacity}) exceeds sequence_len ({}): RoPE tables and the causal mask are sized to sequence_len",
                cfg.sequence_len
            );
        }
        let layers = (0..cfg.n_layer)
            .map(|_| LayerKv {
                k: Cache::new(capacity),
                v: Cache::new(capacity),
            })
            .collect();
        Ok(Self {
            layers,
            len: 0,
            capacity,
        })
    }

    /// Number of tokens cached — the absolute position the next span starts at.
    pub fn pos(&self) -> usize {
        self.len
    }

    /// Rewind to empty, keeping the allocation. The stale rows are never read:
    /// every read narrows to `pos + T`.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Bail unless `t` more tokens fit. Called once per forward, before any
    /// layer writes, so a rejected call leaves every buffer untouched.
    #[cfg_attr(not(test), allow(dead_code))] // consumed by Part D (Gpt::forward_inner).
    pub(crate) fn check_room(&self, t: usize) -> Result<()> {
        if self.len + t > self.capacity {
            bail!(
                "kv-cache overflow: {} cached + {t} new exceeds capacity {}",
                self.len,
                self.capacity
            );
        }
        Ok(())
    }

    /// Called once, after every layer has written.
    #[cfg_attr(not(test), allow(dead_code))] // consumed by Part D (Gpt::forward_inner).
    pub(crate) fn advance(&mut self, t: usize) {
        self.len += t;
    }

    #[cfg_attr(not(test), allow(dead_code))] // consumed by Part D (Gpt::forward_inner).
    pub(crate) fn layer_mut(&mut self, i: usize) -> &mut LayerKv {
        &mut self.layers[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::assert_close;
    use candle_core::Device;

    fn cfg(n_layer: usize, sequence_len: usize) -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len,
            n_layer,
            n_head: 2,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    /// `(1, 2, t, 4)` filled with `fill`, so a row's value identifies it.
    fn span(t: usize, fill: f32, dev: &Device) -> Result<Tensor> {
        Tensor::full(fill, (1, 2, t, 4), dev)
    }

    #[test]
    fn append_returns_the_running_prefix() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = KvCache::new(&cfg(1, 16), 8)?;
        let layer = cache.layer_mut(0);

        // K and V carry *different* fills, so a body that appended both into one
        // buffer — or swapped the pair — fails here rather than passing silently.
        let spans = [(3usize, 1.0f32, 0usize), (1, 2.0, 3), (2, 3.0, 4)];
        let (mut k_written, mut v_written) = (Vec::new(), Vec::new());
        for (t, fill, at) in spans {
            let src_k = span(t, fill, &dev)?;
            let src_v = span(t, -fill, &dev)?;
            let (k, v) = layer.append(&src_k, &src_v, at)?;
            k_written.push(src_k);
            v_written.push(src_v);
            let want_k = Tensor::cat(&k_written, 2)?;
            assert_eq!(k.dims(), want_k.dims(), "prefix shape after append at {at}");
            assert_close(&k, &want_k, 0.0, "k prefix")?;
            assert_close(&v, &Tensor::cat(&v_written, 2)?, 0.0, "v prefix")?;
        }
        Ok(())
    }

    #[test]
    fn reset_rewinds_without_reallocating() -> Result<()> {
        // "Same allocation" is not directly assertable (`Tensor::same_storage`
        // is pub(crate) in candle and there is no data pointer), so assert it
        // behaviorally: fill to capacity with a distinctive pattern, reset, then
        // append one row and check the untouched rows still hold the pattern.
        let dev = Device::Cpu;
        let capacity = 4;
        let mut cache = KvCache::new(&cfg(1, 16), capacity)?;

        // Write then advance, the order `Gpt::forward` uses.
        let full = span(capacity, 7.0, &dev)?;
        cache.layer_mut(0).append(&full, &full, 0)?;
        cache.advance(capacity);
        assert_eq!(cache.pos(), capacity);

        cache.reset();
        assert_eq!(cache.pos(), 0);

        let layer = cache.layer_mut(0);
        let one = span(1, 9.0, &dev)?;
        layer.append(&one, &one, 0)?;

        // The whole buffer, both heads: row 0 rewritten, rows 1.. still holding
        // the pre-reset pattern — which they cannot if `reset` reallocated.
        let buf = layer.k.buf.as_ref().expect("buffer allocated");
        let want = Tensor::cat(&[one, span(capacity - 1, 7.0, &dev)?], 2)?;
        assert_close(buf, &want, 0.0, "buffer after reset + append")?;
        Ok(())
    }

    #[test]
    fn check_room_rejects_a_span_past_capacity() -> Result<()> {
        let mut cache = KvCache::new(&cfg(1, 16), 4)?;
        assert!(cache.check_room(4).is_ok());
        assert!(cache.check_room(5).is_err());
        cache.advance(3);
        assert!(cache.check_room(1).is_ok());
        assert!(cache.check_room(2).is_err());
        Ok(())
    }

    #[test]
    fn new_rejects_capacity_beyond_sequence_len() {
        // Unreachable from `generate_ids`, which caps capacity at seq_len; kept
        // as a guard on a public constructor.
        let cfg = cfg(2, 16);
        assert!(KvCache::new(&cfg, 16).is_ok());
        assert!(KvCache::new(&cfg, 17).is_err());
    }

    #[test]
    fn layers_are_independent() -> Result<()> {
        // `layer_mut` ignoring its index would make every block share layer 0's
        // K/V — silent, and catastrophic — so write to one layer and check the
        // others are still untouched.
        let dev = Device::Cpu;
        let mut cache = KvCache::new(&cfg(3, 16), 4)?;
        assert_eq!(cache.layers.len(), 3);

        // Write through the accessor, read the field directly — reading through
        // `layer_mut` too would hide any *consistent* permutation of its index.
        let one = span(1, 1.0, &dev)?;
        cache.layer_mut(1).append(&one, &one, 0)?;
        assert!(cache.layers[1].k.buf.is_some(), "layer 1 never written");
        for i in [0usize, 2] {
            assert!(cache.layers[i].k.buf.is_none(), "layer {i} aliases layer 1");
        }
        Ok(())
    }
}
