use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::build_causal_mask;
use super::block::Block;
use super::config::GptConfig;
use super::device::compute_dtype;
use super::embedding::TokenEmbedding;
use super::kv_cache::KvCache;
use super::linear::Linear;
use super::rms_norm::rms_norm;
use super::rope::Rope;

/// The GPT model: token embedding → `n_layer` pre-norm transformer blocks →
/// final RMSNorm → untied `lm_head` unembedding, producing logits `(B,T,vocab)`.
pub struct Gpt {
    wte: TokenEmbedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    config: GptConfig,
    /// Activation dtype, derived from the device (bf16 on CUDA, else fp32).
    /// Applied once at the embedding boundary; every downstream module
    /// follows its input's dtype.
    compute_dtype: DType,
}

impl Gpt {
    pub fn new(cfg: GptConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let wte = TokenEmbedding::new(&cfg, vb.pp("wte"))?;

        // One set of RoPE tables and one causal mask, shared by every block:
        // the clones each block takes are handles onto the same device
        // buffers. Building these per layer would duplicate O(seq_len²)
        // memory for the mask alone — ~400 MB at 24 layers / T=2048.
        let rope = Rope::from_config(&cfg, &device)?;
        let causal_mask = build_causal_mask(cfg.sequence_len, &device)?;

        let blocks_vb = vb.pp("blocks");
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(&cfg, blocks_vb.pp(i), &rope, &causal_mask))
            .collect::<Result<Vec<_>>>()?;

        // Untied unembedding: a separate matrix from `wte`, tiny-init so the
        // initial logits are near-uniform and the loss starts at ≈ ln(vocab).
        let lm_head = Linear::normal(cfg.n_embd, cfg.vocab_size, 0.001, vb.pp("lm_head"))?;

        Ok(Self {
            wte,
            blocks,
            lm_head,
            config: cfg,
            compute_dtype: compute_dtype(vb.device()),
        })
    }

    /// Map token ids `idx: (B, T)` to logits `(B, T, vocab_size)`.
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        self.forward_inner(idx, None)
    }

    /// Incremental forward over a KV cache: `idx` is the **new span only**, and
    /// the returned logits `(B, T_new, vocab_size)` cover only that span. The
    /// span is taken to start at `cache.pos()`, so the caller feeds the prompt
    /// once and then one token per step.
    ///
    /// The cache lives with the caller rather than inside the model: `forward`
    /// takes `&self` and is called from eight places, none of which want a
    /// cache, so owning one here would mean either `&mut self` everywhere or a
    /// mutex bought for nothing.
    pub fn forward_with_cache(&self, idx: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        self.forward_inner(idx, Some(cache))
    }

    fn forward_inner(&self, idx: &Tensor, mut cache: Option<&mut KvCache>) -> Result<Tensor> {
        // One read of T for both cache calls: they sit on opposite sides of the
        // block loop and `advance` cannot fail, so disagreeing values would not
        // surface until a much later `slice_set` bail.
        let (_b, t) = idx.dims2()?;
        // The span's absolute start, read once and threaded to every layer —
        // *not* a length per layer. `check_room` runs before any layer writes,
        // and `advance` after every layer has, so a failure mid-loop leaves the
        // position untouched and the next forward simply rewrites the same
        // offsets. Per-layer lengths would instead desync silently, rotating
        // every later key in the lagging layers to the wrong position.
        let t0 = match cache.as_deref() {
            Some(c) => {
                c.check_room(t)?;
                c.pos()
            }
            None => 0,
        };

        let mut x = self.wte.forward(idx)?.to_dtype(self.compute_dtype)?;
        for (i, block) in self.blocks.iter().enumerate() {
            // `as_deref_mut` reborrows the option each iteration; moving it in
            // would consume it on the first block.
            let kv = cache.as_deref_mut().map(|c| c.layer_mut(i));
            x = block.forward_inner(&x, kv, t0)?;
        }
        // Moves the option — this is its last use, and the block loop is done.
        if let Some(c) = cache {
            c.advance(t);
        }

        let x = rms_norm(&x, self.config.norm_eps)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &GptConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Reduction, cross_entropy};
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len: 16,
            n_layer: 2,
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
    fn forward_returns_b_t_vocab() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let vocab = cfg.vocab_size;
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?; // (B=2, T=3)
        assert_eq!(model.forward(&idx)?.dims(), &[2, 3, vocab]);
        Ok(())
    }

    #[test]
    fn cpu_forward_stays_f32() -> Result<()> {
        // Device-gated compute dtype: on CPU the boundary cast and every
        // downstream weight cast are same-dtype no-ops, so the whole fp32 test
        // suite exercises an unchanged graph.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), builder(&vm, &dev))?;
        assert_eq!(model.compute_dtype, DType::F32);

        let idx = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        assert_eq!(model.forward(&idx)?.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn registers_full_parameter_set() -> Result<()> {
        // Locks in the parameter naming as the model grows: the embedding, one
        // `blocks.{i}.*` group per layer, and the untied `lm_head`.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let (vocab, n_embd, n_layer) = (cfg.vocab_size, cfg.n_embd, cfg.n_layer);
        let _model = Gpt::new(cfg, builder(&vm, &dev))?;

        let data = vm.data().lock().unwrap();
        let mut keys: Vec<String> = data.keys().cloned().collect();
        keys.sort();

        let mut want = vec!["lm_head.weight".to_string(), "wte.weight".to_string()];
        for i in 0..n_layer {
            for name in [
                "attn.c_k",
                "attn.c_proj",
                "attn.c_q",
                "attn.c_v",
                "mlp.c_fc",
                "mlp.c_proj",
            ] {
                want.push(format!("blocks.{i}.{name}.weight"));
            }
        }
        want.sort();

        assert_eq!(keys, want);
        assert_eq!(data["wte.weight"].dims(), &[vocab, n_embd]);
        assert_eq!(data["lm_head.weight"].dims(), &[vocab, n_embd]);
        Ok(())
    }

    #[test]
    fn init_logits_are_near_uniform() -> Result<()> {
        // At init every block is the identity (zero-init residual projections)
        // and `lm_head` is tiny-init, so the logits are ~0 ⇒ softmax ≈ uniform,
        // i.e. cross-entropy will start at ≈ ln(vocab).
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3, 4]], &dev)?;
        let logits = model.forward(&idx)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            logits.iter().all(|v| v.abs() < 0.05),
            "expected near-zero init logits, got max {:?}",
            logits.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()))
        );
        Ok(())
    }

    #[test]
    fn forward_with_cache_matches_forward() -> Result<()> {
        // The whole-model parity gate: any split of a sequence into cached
        // spans must reproduce the one-shot forward's logits for those rows.
        // Two layers, because a 1-layer model cannot catch per-layer cache
        // mis-indexing (`test_support::tiny_gpt` is 1 layer, hence the local
        // config). `detie` first, or every block is the identity and the logits
        // are position-independent — a broken cache would pass.
        use crate::model::KvCache;
        use crate::test_support::{assert_close, detie};
        let dev = Device::Cpu;
        let mut vm = VarMap::new();
        let cfg = tiny_cfg();
        assert_eq!(cfg.n_layer, 2, "parity needs more than one layer");
        let model = Gpt::new(cfg, builder(&vm, &dev))?;
        detie(&mut vm, &cfg, &dev)?;

        let ids = [3u32, 1, 4, 1, 5, 9, 2, 6];
        let idx = Tensor::from_slice(&ids, (1, ids.len()), &dev)?;
        let want = model.forward(&idx)?;

        for spans in [vec![ids.len()], vec![5usize, 3], vec![1; ids.len()]] {
            let mut cache = KvCache::new(&cfg, ids.len())?;
            let mut t0 = 0;
            let mut got = Vec::new();
            for len in &spans {
                let span = idx.narrow(1, t0, *len)?;
                got.push(model.forward_with_cache(&span, &mut cache)?);
                assert_eq!(cache.pos(), t0 + len, "pos after span at {t0}");
                t0 += len;
            }
            let got = Tensor::cat(&got, 1)?;
            assert_close(&got, &want, 1e-5, &format!("spans {spans:?}"))?;
        }
        Ok(())
    }

    #[test]
    fn forward_with_cache_rejects_overflow() -> Result<()> {
        // The bail must land before any layer writes, so a rejected call leaves
        // both the buffers and the position untouched and the next call still
        // sees a consistent cache.
        use crate::model::KvCache;
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let model = Gpt::new(cfg, builder(&vm, &dev))?;
        let mut cache = KvCache::new(&cfg, 4)?;

        let idx = Tensor::new(&[[1u32, 2, 3]], &dev)?;
        model.forward_with_cache(&idx, &mut cache)?;
        assert_eq!(cache.pos(), 3);

        // 3 cached + 3 new > capacity 4. Assert on the *message*, not just
        // `is_err`: without `check_room` the first layer's `slice_set` bails on
        // its own, which is also an error that leaves `pos` at 3 — so error-ness
        // alone would not distinguish the cache's guard from candle's, and the
        // guard exists precisely because candle's message never mentions a cache.
        let err = model
            .forward_with_cache(&idx, &mut cache)
            .unwrap_err()
            .to_string();
        assert!(err.contains("kv-cache overflow"), "wrong error: {err}");
        assert_eq!(cache.pos(), 3, "a rejected call must not advance");

        // Still usable: one more token fits.
        let one = Tensor::new(&[[7u32]], &dev)?;
        model.forward_with_cache(&one, &mut cache)?;
        assert_eq!(cache.pos(), 4);
        Ok(())
    }

    #[test]
    fn init_loss_is_near_ln_vocab() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let vocab = cfg.vocab_size;
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?; // (B=2, T=4)
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?; // (B=2, T=4)

        let logits = model.forward(&idx)?; // (2, 4, vocab)
        let loss = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()?;

        assert!(loss.is_finite(), "loss must be finite, got {loss}");
        let expected = (vocab as f32).ln();
        assert!(
            (loss - expected).abs() < 0.1,
            "got {loss}, want ≈ {expected}"
        );
        Ok(())
    }
}
