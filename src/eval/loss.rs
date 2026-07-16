use std::f64::consts::LN_2;

use candle_core::{Result, Tensor};

use crate::data::Batch;
use crate::model::{Gpt, Reduction, cross_entropy};

pub struct EvalMetrics {
    pub val_loss: f64,
    pub bpb: f64,
}

#[derive(Default)]
pub struct BpbAccumulator {
    nats_loss: f64,
    valid_tokens: u64,
    nats_bytes: f64,
    bytes: u64,
}

impl BpbAccumulator {
    pub fn add(&mut self, loss2d: &Tensor, targets: &Tensor, token_bytes: &[u32]) -> Result<()> {
        let loss = loss2d.flatten_all()?.to_vec1::<f32>()?;
        let tgts = targets.flatten_all()?.to_vec1::<i64>()?;
        debug_assert_eq!(loss.len(), tgts.len(), "loss/targets length mismatch");
        for (&l, &y) in loss.iter().zip(&tgts) {
            if y < 0 {
                continue; // ignore_index: counted by neither metric
            }
            self.nats_loss += l as f64;
            self.valid_tokens += 1;
            let b = token_bytes[y as usize];
            if b > 0 {
                // A real (non-special) token: contributes to the byte-normalized bpb.
                self.nats_bytes += l as f64;
                self.bytes += b as u64;
            }
        }
        Ok(())
    }

    /// The four running sums as `[nats_loss, valid_tokens, nats_bytes, bytes]`
    /// — the shape a cross-rank `Sum` all-reduce composes over. The two `u64`
    /// counters are deliberately cast to f64: integers are exact in f64 up to
    /// 2⁵³, far beyond any realistic token/byte count, and one uniform dtype
    /// lets all four ride a single reduce.
    pub fn sums(&self) -> [f64; 4] {
        [
            self.nats_loss,
            self.valid_tokens as f64,
            self.nats_bytes,
            self.bytes as f64,
        ]
    }

    /// Metrics from (possibly cross-rank-reduced) [`sums`](Self::sums).
    pub fn metrics_from_sums(sums: [f64; 4]) -> EvalMetrics {
        let [nats_loss, valid_tokens, nats_bytes, bytes] = sums;
        let val_loss = if valid_tokens == 0.0 {
            f64::INFINITY
        } else {
            nats_loss / valid_tokens
        };
        let bpb = if bytes == 0.0 {
            f64::INFINITY
        } else {
            nats_bytes / (LN_2 * bytes)
        };
        EvalMetrics { val_loss, bpb }
    }

    pub fn metrics(&self) -> EvalMetrics {
        Self::metrics_from_sums(self.sums())
    }
}

/// Score only the batches of one shard — `batches[i]` with `i % num_shards ==
/// shard_index` — and return the accumulator's raw sums. Data-parallel eval
/// splits the val set this way (every rank holds the same snapshot, scores its
/// slice, and the sums are `Sum`-reduced before turning into metrics).
pub fn evaluate_shard_sums(
    model: &Gpt,
    batches: &[Batch],
    token_bytes: &[u32],
    shard_index: usize,
    num_shards: usize,
) -> Result<[f64; 4]> {
    assert!(
        num_shards >= 1 && shard_index < num_shards,
        "invalid shard {shard_index} / {num_shards}"
    );
    let mut acc = BpbAccumulator::default();
    for batch in batches.iter().skip(shard_index).step_by(num_shards) {
        let logits = model.forward(&batch.inputs)?;
        let loss2d = cross_entropy(&logits, &batch.targets, -1, Reduction::None)?;
        acc.add(&loss2d, &batch.targets, token_bytes)?;
    }
    Ok(acc.sums())
}

pub fn evaluate(model: &Gpt, batches: &[Batch], token_bytes: &[u32]) -> Result<EvalMetrics> {
    let sums = evaluate_shard_sums(model, batches, token_bytes, 0, 1)?;
    Ok(BpbAccumulator::metrics_from_sums(sums))
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_core::Device;

    use crate::data::DataLoader;
    use crate::test_support::{byte_tokenizer, tiny_gpt, two_shard_corpus};

    fn t2<D: candle_core::WithDType>(data: &[D], b: usize, t: usize) -> Tensor {
        Tensor::from_vec(data.to_vec(), (b, t), &Device::Cpu).unwrap()
    }

    #[test]
    fn masks_specials_from_bpb_and_ignored_from_both() {
        // Distinct per-position losses pin down exactly which positions each
        // metric summed.   p0 normal   p1 normal   p2 special(0B)   p3 ignored(-1)
        let loss = t2(&[1.0f32, 2.0, 4.0, 8.0], 1, 4);
        let tgt = t2(&[0i64, 1, 2, -1], 1, 4);
        let token_bytes = vec![1u32, 3, 0, 5]; // id2 is a special (0 bytes)

        let mut acc = BpbAccumulator::default();
        acc.add(&loss, &tgt, &token_bytes).unwrap();
        let m = acc.metrics();

        // val loss drops only p3 (-1); the special p2 still counts → (1+2+4)/3.
        assert!(
            (m.val_loss - (1.0 + 2.0 + 4.0) / 3.0).abs() < 1e-6,
            "{}",
            m.val_loss
        );
        // bpb drops p3 (ignored) and p2 (0 bytes): nats 1+2 over bytes 1+3.
        let want = (1.0 + 2.0) / (LN_2 * (1.0 + 3.0));
        assert!((m.bpb - want).abs() < 1e-6, "{}", m.bpb);
    }

    #[test]
    fn no_scored_bytes_yields_infinite_bpb() {
        // Every target is a special (0 bytes): val loss is finite (positions are
        // not ignored) but bpb has no bytes to normalize by → infinite.
        let loss = t2(&[1.0f32, 2.0], 1, 2);
        let tgt = t2(&[0i64, 0], 1, 2);
        let token_bytes = vec![0u32]; // id 0 is a special

        let mut acc = BpbAccumulator::default();
        acc.add(&loss, &tgt, &token_bytes).unwrap();
        let m = acc.metrics();
        assert!(m.val_loss.is_finite());
        assert!(m.bpb.is_infinite());

        // An empty accumulator has no valid tokens and no bytes → both infinite.
        let empty = BpbAccumulator::default().metrics();
        assert!(empty.val_loss.is_infinite() && empty.bpb.is_infinite());
    }

    #[test]
    fn add_accumulates_across_batches() {
        // Two batches must compose into the same sums as one combined batch.
        let token_bytes = vec![1u32, 2, 1];
        let mut split = BpbAccumulator::default();
        split
            .add(
                &t2(&[1.0f32, 2.0], 1, 2),
                &t2(&[0i64, 1], 1, 2),
                &token_bytes,
            )
            .unwrap();
        split
            .add(
                &t2(&[3.0f32, 4.0], 1, 2),
                &t2(&[2i64, 0], 1, 2),
                &token_bytes,
            )
            .unwrap();

        let mut combined = BpbAccumulator::default();
        combined
            .add(
                &t2(&[1.0f32, 2.0, 3.0, 4.0], 1, 4),
                &t2(&[0i64, 1, 2, 0], 1, 4),
                &token_bytes,
            )
            .unwrap();

        let (a, b) = (split.metrics(), combined.metrics());
        assert!((a.val_loss - b.val_loss).abs() < 1e-9);
        assert!((a.bpb - b.bpb).abs() < 1e-9);
    }

    /// Sharded eval must partition the batch list: per-shard sums add up to
    /// the unsharded run's sums (counts exactly — they're integers in f64;
    /// nats to f64 round-off, since only the addition order differs), and a
    /// single shard is the identity.
    #[test]
    fn shard_sums_partition_the_batches() -> Result<()> {
        use crate::data::{DataLoader, Split};
        use crate::test_support::{byte_tokenizer, tiny_gpt, two_shard_corpus};

        let dir = two_shard_corpus();
        let tok = byte_tokenizer();
        let token_bytes = tok.token_byte_lengths();
        let dev = Device::Cpu;
        let (b, t) = (2usize, 8usize);
        let mut loader =
            DataLoader::open_with_buffer_size(dir.path(), Split::Val, &tok, b, t, 4).unwrap();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), t);
        let batches = loader.take_batches(5, &dev)?; // odd count: uneven shards

        let full = evaluate_shard_sums(&model, &batches, &token_bytes, 0, 1)?;

        for num_shards in [1usize, 2, 3] {
            let mut merged = [0.0f64; 4];
            for shard in 0..num_shards {
                let s = evaluate_shard_sums(&model, &batches, &token_bytes, shard, num_shards)?;
                for (m, x) in merged.iter_mut().zip(s) {
                    *m += x;
                }
            }
            // Integer-valued counts must match exactly.
            assert_eq!(merged[1], full[1], "valid_tokens, {num_shards} shards");
            assert_eq!(merged[3], full[3], "bytes, {num_shards} shards");
            // Nats sums only reassociate f64 additions.
            assert!((merged[0] - full[0]).abs() <= 1e-9 * (1.0 + full[0].abs()));
            assert!((merged[2] - full[2]).abs() <= 1e-9 * (1.0 + full[2].abs()));
        }
        Ok(())
    }

    /// `metrics_from_sums` over merged shard sums is how the distributed path
    /// computes val metrics; it must agree with the plain accumulator.
    #[test]
    fn metrics_from_sums_matches_metrics() {
        let token_bytes = vec![1u32, 2, 1];
        let mut acc = BpbAccumulator::default();
        acc.add(
            &t2(&[1.0f32, 2.0, 3.0], 1, 3),
            &t2(&[0i64, 1, 2], 1, 3),
            &token_bytes,
        )
        .unwrap();

        let direct = acc.metrics();
        let via_sums = BpbAccumulator::metrics_from_sums(acc.sums());
        assert_eq!(direct.val_loss, via_sums.val_loss);
        assert_eq!(direct.bpb, via_sums.bpb);

        // Zero sums (an empty accumulator) keep the infinite-metric contract.
        let empty = BpbAccumulator::metrics_from_sums([0.0; 4]);
        assert!(empty.val_loss.is_infinite() && empty.bpb.is_infinite());
    }

    /// The full loader → forward → cross_entropy → accumulate loop.
    /// An untrained model has near-uniform logits regardless of input, so both
    /// metrics land near their ln(vocab) baseline.
    #[test]
    fn evaluate_untrained_model_is_near_ln_vocab() -> Result<()> {
        use crate::data::Split;

        let dir = two_shard_corpus();
        let tok = byte_tokenizer();
        let token_bytes = tok.token_byte_lengths();

        let dev = Device::Cpu;
        let (b, t) = (2usize, 8usize);
        let mut loader =
            DataLoader::open_with_buffer_size(dir.path(), Split::Val, &tok, b, t, 4).unwrap();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), t);

        let batches = loader.take_batches(2, &dev)?;
        let m = evaluate(&model, &batches, &token_bytes)?;

        let ln_vocab = (tok.vocab_size() as f64).ln();
        assert!(
            (m.val_loss - ln_vocab).abs() < 0.5,
            "val_loss {} not near ln(vocab) {ln_vocab}",
            m.val_loss
        );
        // Byte tokenizer ⇒ each scored token is one byte ⇒ bpb ≈ val_loss / ln2.
        assert!(
            (m.bpb - m.val_loss / LN_2).abs() < 0.3,
            "bpb {} inconsistent with val_loss {}",
            m.bpb,
            m.val_loss
        );
        Ok(())
    }
}
