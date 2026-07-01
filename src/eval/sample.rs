use candle_core::{Device, IndexOp, Result, Tensor};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::model::Gpt;
use crate::tokenizer::{BpeTokenizer, TokenId};

/// Autoregressive generation for a single prompt (batch size 1). `temperature
/// <= 0` is greedy (argmax); otherwise the next token is drawn from
/// `softmax(logits / temperature)` with a seeded PRNG. The context is cropped to
/// the model's `sequence_len`, and the output is decoded with `decode_lossy` so a
/// sequence cut mid-character never panics. Returns the BOS-prefixed prompt plus
/// the continuation.
pub fn generate(
    model: &Gpt,
    tok: &BpeTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    seed: u64,
    device: &Device,
) -> Result<String> {
    let seq_len = model.config().sequence_len;

    let mut ids = vec![tok.bos_id()];
    ids.extend(tok.encode(prompt));
    ids.reserve(max_tokens);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..max_tokens {
        // No KV-cache: re-forward the whole (cropped) context each step.
        let start = ids.len().saturating_sub(seq_len);
        let ctx = &ids[start..];
        let input = Tensor::from_vec(ctx.to_vec(), (1, ctx.len()), device)?;
        let logits = model.forward(&input)?;
        let last = logits.i((0, ctx.len() - 1))?.to_vec1::<f32>()?;

        let next = if temperature <= 0.0 {
            argmax(&last)
        } else {
            sample(&last, temperature, &mut rng)
        };
        ids.push(next as TokenId);
    }

    Ok(tok.decode_lossy(&ids))
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .expect("logits over a non-empty vocab")
}

/// Sample an index from `softmax(logits / temperature)` by inverse-CDF. The
/// softmax is computed in `f64` with the max subtracted for stability; we draw a
/// uniform over the unnormalized total rather than normalizing first.
fn sample(logits: &[f32], temperature: f64, rng: &mut ChaCha8Rng) -> usize {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let weights: Vec<f64> = logits
        .iter()
        .map(|&l| ((l as f64 - max) / temperature).exp())
        .collect();
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
    use crate::model::GptConfig;
    use crate::test_support::byte_tokenizer;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    fn tiny_model(tok: &BpeTokenizer, seq_len: usize, dev: &Device) -> (Gpt, VarMap) {
        let cfg = GptConfig {
            vocab_size: tok.vocab_size(),
            sequence_len: seq_len,
            n_layer: 1,
            n_head: 1,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        };
        let vm = VarMap::new();
        let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, dev)).unwrap();
        (model, vm)
    }

    #[test]
    fn greedy_is_deterministic_and_seed_independent() {
        let dev = Device::Cpu;
        let mut vf = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vf);
        let (model, _vm) = tiny_model(&tok, 64, &dev);

        // Greedy is fully determined by the weights, so the seed cannot change it.
        let a = generate(&model, &tok, "hello", 12, 0.0, 42, &dev).unwrap();
        let b = generate(&model, &tok, "hello", 12, 0.0, 7, &dev).unwrap();
        assert_eq!(a, b);
        assert!(a.starts_with("<|bos|>hello"), "got {a:?}");

        // max_tokens == 0 returns the BOS-prefixed prompt unchanged.
        let none = generate(&model, &tok, "abc", 0, 0.0, 0, &dev).unwrap();
        assert_eq!(none, "<|bos|>abc");
    }

    #[test]
    fn temperature_sampling_is_reproducible_per_seed() {
        let dev = Device::Cpu;
        let mut vf = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vf);
        let (model, _vm) = tiny_model(&tok, 64, &dev);

        let a = generate(&model, &tok, "hi", 16, 0.8, 123, &dev).unwrap();
        let b = generate(&model, &tok, "hi", 16, 0.8, 123, &dev).unwrap();
        assert_eq!(a, b, "same seed must reproduce the same sample");
    }

    #[test]
    fn generation_past_seq_len_crops_without_panicking() {
        // seq_len is 8 but we generate well past it: the context must be cropped
        // to the last seq_len tokens (RoPE/mask are sized to seq_len), not error.
        let dev = Device::Cpu;
        let mut vf = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vf);
        let (model, _vm) = tiny_model(&tok, 8, &dev);

        let out = generate(&model, &tok, "context", 30, 1.0, 5, &dev).unwrap();
        assert!(out.starts_with("<|bos|>context"), "got {out:?}");
    }
}
