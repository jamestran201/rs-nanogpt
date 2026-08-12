use candle_core::{DType, Device, IndexOp, Result, Tensor};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::model::{Gpt, KvCache};
use crate::tokenizer::TokenId;

#[derive(Debug, Clone, Copy)]
pub struct SampleOptions {
    pub max_tokens: usize,
    /// `<= 0` is greedy (argmax); otherwise `softmax(logits / temperature)`.
    /// Every CLI surface rejects negatives, so `<= 0` means exactly `0` in
    /// practice.
    pub temperature: f64,
    /// Sample only from the `k` highest-logit tokens. `0` disables the
    /// restriction; ignored when greedy. nanochat's chat default is 50
    /// (`chat_cli.py:19`).
    pub top_k: usize,
    pub seed: u64,
}

/// Continue `prefix` (a non-empty, already-rendered token sequence), one token
/// at a time. An empty `prefix` is an `Err`, not a panic. Batch size 1, over a
/// KV cache: the prefix is forwarded once and each step afterwards forwards
/// only the token it just sampled — until the history outgrows the model's
/// `sequence_len`, past which the window slides and each step re-forwards the
/// whole cropped window.
///
/// Returns the *generated* ids only (never the prefix). Generation ends at
/// `max_tokens` or on the first id in `stop`; a stop id **is** in the return
/// value — history must stay well-formed — but is **not** passed to
/// `on_token`, so a streaming caller never prints `<|assistant_end|>`.
pub fn generate_ids(
    model: &Gpt,
    prefix: &[TokenId],
    opts: SampleOptions,
    stop: &[TokenId],
    device: &Device,
    mut on_token: impl FnMut(TokenId) -> Result<()>,
) -> Result<Vec<TokenId>> {
    // The doc'd precondition, enforced: an empty prefix would underflow
    // `ctx.len() - 1` below rather than returning the `Result` we promise.
    if prefix.is_empty() {
        candle_core::bail!("generate_ids: prefix must be non-empty");
    }
    let seq_len = model.config().sequence_len;

    let mut ids = Vec::with_capacity(prefix.len() + opts.max_tokens);
    ids.extend_from_slice(prefix);

    // One allocation for the whole call, not one per token — sized to the
    // longest context this call can actually reach. `reset` below keeps it.
    let capacity = seq_len.min(prefix.len() + opts.max_tokens);
    let mut cache = KvCache::new(model.config(), capacity)?;

    let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
    for _ in 0..opts.max_tokens {
        // The cache holds `ids[..cache.pos()]`. Normally that is everything but
        // the token just sampled, so exactly one token is fed. Once the history
        // outgrows the context the window slides, every cached key's RoPE angle
        // shifts, and the cache has to be rebuilt from the cropped window —
        // which is the same full re-forward the pre-cache loop did every step,
        // so no step gets asymptotically slower and the logits stay comparable.
        let start = ids.len().saturating_sub(seq_len);
        if start > 0 {
            cache.reset();
        }
        // Covers both branches: after a reset `pos == 0` so `start` wins; with
        // no crop `start == 0` so `pos` wins. Never empty — with no crop
        // `ids.len() == pos + 1` by induction (`start` never decreases, since
        // `ids` only grows, so a crop step is never *followed* by a non-crop
        // one and the invariant is never re-entered after a reset breaks it),
        // or `prefix.len()` on the first step; and after a crop
        // `start < ids.len()` since `seq_len >= 1`. That non-emptiness is what
        // protects the `n - 1` below.
        let feed = &ids[start.max(cache.pos())..];

        // `n` before the tensor: `feed` borrows `ids`, which `ids.push(next)`
        // needs back. NLL ends the borrow here either way, but binding the
        // length keeps a later reorder from breaking the build obscurely.
        let n = feed.len();
        let input = Tensor::from_vec(feed.to_vec(), (1, n), device)?;
        let logits = model.forward_with_cache(&input, &mut cache)?;
        // Upcast before the host read: `to_vec1::<f32>` requires an exact
        // dtype match and errors on the bf16 logits a CUDA forward produces.
        let last = logits
            .i((0, n - 1))?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        let next = if opts.temperature <= 0.0 {
            argmax(&last)
        } else {
            sample(&last, opts.temperature, opts.top_k, &mut rng)
        } as TokenId;
        ids.push(next);
        if stop.contains(&next) {
            break;
        }
        on_token(next)?;
    }

    Ok(ids.split_off(prefix.len()))
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("logits over a non-empty vocab")
}

/// Sample an index from `softmax(logits / temperature)`, restricted to the
/// `top_k` highest logits when `top_k` is in `1..vocab_size`.
///
/// The unrestricted branch is *required*, not an optimization:
/// `select_nth_unstable_by` panics at `index == len`.
fn sample(logits: &[f32], temperature: f64, top_k: usize, rng: &mut ChaCha8Rng) -> usize {
    if top_k == 0 || top_k >= logits.len() {
        return draw(&softmax(logits.iter().copied(), temperature), rng);
    }
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    // Descending on logit, so the k survivors are the k largest.
    idx.select_nth_unstable_by(top_k, |&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize])
    });
    idx.truncate(top_k);
    let weights = softmax(idx.iter().map(|&i| logits[i as usize]), temperature);
    idx[draw(&weights, rng)] as usize
}

/// The *unnormalized* softmax `exp((l - max) / temperature)`, in `f64` with the
/// max subtracted for stability. Normalizing is left to [`draw`], which divides
/// nothing and compares against the total instead.
fn softmax(logits: impl Iterator<Item = f32> + Clone, temperature: f64) -> Vec<f64> {
    let max = logits.clone().fold(f32::NEG_INFINITY, f32::max) as f64;
    logits
        .map(|l| ((l as f64 - max) / temperature).exp())
        .collect()
}

/// Inverse-CDF draw: a uniform over the unnormalized total, walked against the
/// running sum.
fn draw(weights: &[f64], rng: &mut ChaCha8Rng) -> usize {
    let total: f64 = weights.iter().sum();
    let threshold = rng.random::<f64>() * total;
    let mut acc = 0.0;
    for (i, w) in weights.iter().enumerate() {
        acc += w;
        if acc > threshold {
            return i;
        }
    }
    weights.len() - 1 // unreachable barring f64 rounding at the tail
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{byte_tokenizer, detie, tiny_gpt};
    use crate::tokenizer::BpeTokenizer;

    /// The pre-KV-cache loop, kept as the parity oracle for the cached one:
    /// re-forward the whole cropped context every step. Deliberately a copy
    /// rather than a refactor — an oracle sharing code with what it checks
    /// would move along with it. No stop handling; the callers pass none.
    fn generate_ids_uncached(
        model: &Gpt,
        prefix: &[TokenId],
        opts: SampleOptions,
        device: &Device,
    ) -> Result<Vec<TokenId>> {
        let seq_len = model.config().sequence_len;
        let mut ids = Vec::with_capacity(prefix.len() + opts.max_tokens);
        ids.extend_from_slice(prefix);

        let mut rng = ChaCha8Rng::seed_from_u64(opts.seed);
        for _ in 0..opts.max_tokens {
            let start = ids.len().saturating_sub(seq_len);
            let ctx = &ids[start..];
            let input = Tensor::from_vec(ctx.to_vec(), (1, ctx.len()), device)?;
            let logits = model.forward(&input)?;
            let last = logits
                .i((0, ctx.len() - 1))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let next = if opts.temperature <= 0.0 {
                argmax(&last)
            } else {
                sample(&last, opts.temperature, opts.top_k, &mut rng)
            } as TokenId;
            ids.push(next);
        }
        Ok(ids.split_off(prefix.len()))
    }

    /// `tiny_gpt`, de-tied. Both are needed for a token-equality test: at init
    /// every block is the identity, so the logits are position-independent and
    /// a cache that ignored its own contents would pass; and `lm_head` is
    /// tiny-init, so top-2 logit gaps are the same order as the
    /// cached-vs-uncached numerical difference and argmax would be a coin flip.
    fn detied_gpt(vocab: usize, seq_len: usize) -> Gpt {
        let (mut vm, model) = tiny_gpt(vocab, seq_len);
        detie(&mut vm, model.config(), &Device::Cpu).unwrap();
        model
    }

    /// The base-format prefix a pretraining sample conditions on.
    fn prefix(tok: &BpeTokenizer, text: &str) -> Vec<TokenId> {
        let mut ids = vec![tok.bos_id()];
        ids.extend(tok.encode(text));
        ids
    }

    fn opts(max_tokens: usize, temperature: f64, seed: u64) -> SampleOptions {
        SampleOptions {
            max_tokens,
            temperature,
            top_k: 0,
            seed,
        }
    }

    /// `tiny_gpt` builds from an unseeded `VarMap::new()`, so no test here may
    /// assume *which* ids come out — only relationships between runs.
    #[test]
    fn greedy_is_deterministic_and_seed_independent() {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 64);
        let p = prefix(&tok, "hello");

        // Greedy is fully determined by the weights, so the seed cannot change it.
        let a = generate_ids(&model, &p, opts(12, 0.0, 42), &[], &dev, |_| Ok(())).unwrap();
        let b = generate_ids(&model, &p, opts(12, 0.0, 7), &[], &dev, |_| Ok(())).unwrap();
        assert_eq!(a, b);
        // The continuation only: the prefix is never in the return value.
        assert_eq!(a.len(), 12);
    }

    #[test]
    fn temperature_sampling_is_reproducible_per_seed() {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 64);
        let p = prefix(&tok, "hi");

        let a = generate_ids(&model, &p, opts(16, 0.8, 123), &[], &dev, |_| Ok(())).unwrap();
        let b = generate_ids(&model, &p, opts(16, 0.8, 123), &[], &dev, |_| Ok(())).unwrap();
        assert_eq!(a, b, "same seed must reproduce the same sample");
    }

    #[test]
    fn generation_past_seq_len_crops_without_panicking() {
        // seq_len is 8 but we generate well past it: the context must be cropped
        // to the last seq_len tokens (RoPE/mask are sized to seq_len), not error.
        // Chat closes this path with its own budget guard; pretrain sampling
        // still runs it.
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 8);
        let p = prefix(&tok, "context");

        let out = generate_ids(&model, &p, opts(30, 1.0, 5), &[], &dev, |_| Ok(())).unwrap();
        assert_eq!(out.len(), 30);
    }

    /// **The Part E gate.** Everything below this exercises the sampling
    /// contract; this is the only test that exercises what the cache added —
    /// the `start` / `cache.pos()` / `reset` / `capacity` bookkeeping at the
    /// loop head. A hand-rolled cache loop in the test would re-test the model
    /// instead, which `gpt.rs` already covers.
    ///
    /// It pins *what* the loop computes, not *how much* it forwards. A
    /// regression to re-prefilling the whole context every step — an
    /// unconditional `cache.reset()`, say — is still correct and still fits
    /// `capacity`, so it would pass here — and no *value-level* test can catch
    /// it: the returned ids are identical, and `capacity` is derived from
    /// exactly the quantities a re-prefill is bounded by, so it cannot fire
    /// either. Only cost is observable, which is why the one-token-per-step
    /// property is gated by the timed `chat --max-tokens 200` run in the plan's
    /// Verification section rather than here.
    #[test]
    fn cached_generation_matches_the_uncached_reference() {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let model = detied_gpt(tok.vocab_size(), 64);
        let p = prefix(&tok, "hello");
        let o = opts(20, 0.0, 0);

        let cached = generate_ids(&model, &p, o, &[], &dev, |_| Ok(())).unwrap();
        let uncached = generate_ids_uncached(&model, &p, o, &dev).unwrap();
        assert_eq!(cached.len(), 20);
        assert_eq!(cached, uncached);
    }

    #[test]
    fn generation_past_seq_len_matches_the_uncached_reference() {
        // The crop/reset branch, which the config above never enters: 8-token
        // context, 12 generated, so every step past the 8th rebuilds the cache
        // from the slid window.
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let model = detied_gpt(tok.vocab_size(), 8);
        let p = prefix(&tok, "context");
        let o = opts(12, 0.0, 0);

        let cached = generate_ids(&model, &p, o, &[], &dev, |_| Ok(())).unwrap();
        let uncached = generate_ids_uncached(&model, &p, o, &dev).unwrap();
        assert_eq!(cached.len(), 12);
        assert_eq!(cached, uncached);
    }

    /// Self-calibrating: the stop id is taken from an unrestricted run, so the
    /// test cannot depend on the random init.
    #[test]
    fn stop_token_ends_generation_early() {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 64);
        let p = prefix(&tok, "hi");
        let o = opts(16, 0.0, 1);

        let full = generate_ids(&model, &p, o, &[], &dev, |_| Ok(())).unwrap();
        let stop = full[2];
        let first = full.iter().position(|&id| id == stop).unwrap();

        let mut streamed = Vec::new();
        let out = generate_ids(&model, &p, o, &[stop], &dev, |id| {
            streamed.push(id);
            Ok(())
        })
        .unwrap();

        assert_eq!(out, full[..=first], "generation stops on the stop id");
        assert_eq!(
            out.last(),
            Some(&stop),
            "the stop id stays in the returned ids, so history is well-formed"
        );
        assert_eq!(streamed, full[..first], "but is never handed to the caller");
    }

    /// Restricting to the single highest logit is argmax, at any temperature.
    /// A property of every model, so no golden vector is needed.
    #[test]
    fn top_k_one_is_greedy() {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 64);
        let p = prefix(&tok, "hi");

        let greedy = generate_ids(&model, &p, opts(10, 0.0, 3), &[], &dev, |_| Ok(())).unwrap();
        let k1 = SampleOptions {
            top_k: 1,
            ..opts(10, 0.9, 3)
        };
        let restricted = generate_ids(&model, &p, k1, &[], &dev, |_| Ok(())).unwrap();
        assert_eq!(greedy, restricted);
    }

    #[test]
    fn max_tokens_zero_generates_nothing() {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 64);
        let p = prefix(&tok, "abc");

        let mut fired = false;
        let out = generate_ids(&model, &p, opts(0, 0.0, 0), &[], &dev, |_| {
            fired = true;
            Ok(())
        })
        .unwrap();
        assert!(out.is_empty());
        assert!(
            !fired,
            "the callback must not fire when nothing is generated"
        );
    }

    #[test]
    fn empty_prefix_is_an_error_not_an_underflow() {
        let (_vm, model) = tiny_gpt(byte_tokenizer().vocab_size(), 64);
        let e = generate_ids(&model, &[], opts(4, 0.0, 0), &[], &Device::Cpu, |_| Ok(()))
            .expect_err("an empty prefix has no last position to read logits from");
        assert!(e.to_string().contains("must be non-empty"), "{e}");
    }

    /// `top_k_one_is_greedy` pins the comparator's direction but can only catch
    /// an off-by-one in `truncate` probabilistically. Crafted logits make the
    /// boundary exact: with `k == 2`, ids 0 and 3 must never be drawn.
    ///
    /// The four logits are deliberately *near-competitive*. Spreading them out
    /// would make the test vacuous against the very mutant it advertises: at
    /// `[0, 10, 9, -5]` a `truncate(top_k + 1)` bug leaks id 0 with probability
    /// 3e-5, so 200 draws would miss it ~99% of the time. Within half a nat,
    /// the leak is 0.24 per draw and cannot hide.
    #[test]
    fn top_k_restricts_to_exactly_the_k_largest() {
        // Descending: 1 (10.0), 2 (9.9), then 0 (9.5), 3 (9.4).
        let logits = [9.5f32, 10.0, 9.9, 9.4];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut seen = [false; 4];
        for _ in 0..200 {
            seen[sample(&logits, 1.0, 2, &mut rng)] = true;
        }
        assert_eq!(
            seen,
            [false, true, true, false],
            "only the top 2 logits are reachable, and both are"
        );
    }
}
