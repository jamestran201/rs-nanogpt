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

    pub fn metrics(&self) -> EvalMetrics {
        let val_loss = if self.valid_tokens == 0 {
            f64::INFINITY
        } else {
            self.nats_loss / self.valid_tokens as f64
        };
        let bpb = if self.bytes == 0 {
            f64::INFINITY
        } else {
            self.nats_bytes / (LN_2 * self.bytes as f64)
        };
        EvalMetrics { val_loss, bpb }
    }
}

pub fn evaluate(model: &Gpt, batches: &[Batch], token_bytes: &[u32]) -> Result<EvalMetrics> {
    let mut acc = BpbAccumulator::default();
    for batch in batches {
        let logits = model.forward(&batch.inputs)?;
        let loss2d = cross_entropy(&logits, &batch.targets, -1, Reduction::None)?;
        acc.add(&loss2d, &batch.targets, token_bytes)?;
    }
    Ok(acc.metrics())
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
