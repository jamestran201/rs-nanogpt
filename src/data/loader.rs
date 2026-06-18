use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use crate::tokenizer::BpeTokenizer;

use super::parquet::{ShardTextIter, Split, list_shards, shards_for_split};

/// Greedy assembler over an injected, re-creatable document-token source.
///
/// `make_docs` produces a fresh iterator over the documents for one epoch;
/// `BatchAssembler` calls it again whenever the current epoch runs out, yielding
/// an effectively infinite token stream.
///
/// Caveat — cross-document contamination: documents are concatenated into one
/// flat stream and chopped on fixed `T+1` boundaries, with no separator token.
/// So at each document seam the model is trained on one spurious next-token pair
/// (predict the first token of doc N+1 from the last token of doc N)
/// — those two tokens are unrelated.
pub(crate) struct BatchAssembler<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = Vec<u32>>,
{
    make_docs: F,
    docs: I,
    buf: VecDeque<u32>,
    batch_rows: usize, // B
    row_len: usize,    // T + 1
    /// Tokens pulled since the current epoch began; reset on restart. Lets us
    /// detect a genuinely empty source (a full epoch yielding zero tokens)
    /// instead of looping forever.
    produced_this_epoch: usize,
}

impl<F, I> BatchAssembler<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = Vec<u32>>,
{
    pub(crate) fn new(make_docs: F, batch_rows: usize, tokens_dim: usize) -> Self {
        let docs = make_docs();
        Self {
            make_docs,
            docs,
            buf: VecDeque::new(),
            batch_rows,
            row_len: tokens_dim + 1,
            produced_this_epoch: 0,
        }
    }

    fn fill_to(&mut self, n: usize) {
        while self.buf.len() < n {
            match self.docs.next() {
                Some(doc) => {
                    self.produced_this_epoch += doc.len();
                    self.buf.extend(doc);
                }
                None => {
                    assert!(
                        self.produced_this_epoch > 0,
                        "data source produced no tokens in a full epoch — empty corpus"
                    );
                    self.docs = (self.make_docs)();
                    self.produced_this_epoch = 0;
                }
            }
        }
    }

    /// Drain exactly `B * (T+1)` tokens — one batch's worth of rows.
    /// Leftover tokens stay in `buf` for the next call.
    fn next_rows(&mut self) -> Vec<u32> {
        let n = self.batch_rows * self.row_len;
        self.fill_to(n);
        self.buf.drain(..n).collect()
    }

    /// One batch as flat `inputs (B * T, u32)` and `targets (B * T, i64)` vectors, already next-token shifted
    pub(crate) fn next_batch(&mut self) -> FlatBatch {
        let rows = self.next_rows();
        split_rows(&rows, self.batch_rows, self.row_len - 1)
    }
}

pub(crate) struct FlatBatch {
    pub(crate) inputs: Vec<u32>,
    pub(crate) targets: Vec<i64>,
}

/// Split `B` rows of `T+1` tokens into the model's `inputs`/`targets`
/// `inputs = row[..T]` and `targets = row[1..=T]`
/// Targets are widened to `i64` for `cross_entropy`
pub(crate) fn split_rows(rows: &[u32], batch_size: usize, tokens_dim: usize) -> FlatBatch {
    let row_len = tokens_dim + 1;
    debug_assert_eq!(rows.len(), batch_size * row_len, "rows must be exactly B * (T+1)");

    let mut inputs = Vec::with_capacity(batch_size * tokens_dim);
    let mut targets = Vec::with_capacity(batch_size * tokens_dim);
    for r in 0..batch_size {
        let row = &rows[r * row_len..(r + 1) * row_len];
        inputs.extend_from_slice(&row[..tokens_dim]);
        targets.extend(row[1..=tokens_dim].iter().map(|&x| x as i64));
    }
    FlatBatch { inputs, targets }
}

/// One epoch's worth of token documents: stream the `text` column of `shards`
/// and tokenize each non-empty document. This is the source the `BatchAssembler` consumes.
pub(crate) fn encode_docs<'a>(
    shards: Vec<PathBuf>,
    tokenizer: &'a BpeTokenizer,
) -> impl Iterator<Item = Vec<u32>> + 'a {
    ShardTextIter::new(shards).filter_map(move |res| {
        let text = res.unwrap_or_else(|e| panic!("failed reading shard text: {e}"));
        let tokens = tokenizer.encode(&text);
        (!tokens.is_empty()).then_some(tokens)
    })
}

/// One epoch's document-token iterator, type-erased so `DataLoader` can name it.
type DocIter<'a> = Box<dyn Iterator<Item = Vec<u32>> + 'a>;
/// Factory that produces a fresh `DocIter` per epoch — the cycling source the
/// `BatchAssembler` calls again whenever the current epoch is exhausted.
type DocFactory<'a> = Box<dyn Fn() -> DocIter<'a> + 'a>;

#[derive(Debug)]
pub struct Batch {
    pub inputs: Tensor,
    pub targets: Tensor,
}

pub struct DataLoader<'a> {
    assembler: BatchAssembler<DocFactory<'a>, DocIter<'a>>,
    batch_size: usize,
    tokens_dim: usize,
}

impl<'a> DataLoader<'a> {
    pub fn open(
        dir: &Path,
        split: Split,
        tokenizer: &'a BpeTokenizer,
        batch_size: usize,
        tokens_dim: usize,
    ) -> std::io::Result<Self> {
        let all = list_shards(dir)?;
        let shards = shards_for_split(&all, split)?;
        let factory: DocFactory<'a> =
            Box::new(move || Box::new(encode_docs(shards.clone(), tokenizer)) as DocIter<'a>);
        Ok(Self {
            assembler: BatchAssembler::new(factory, batch_size, tokens_dim),
            batch_size,
            tokens_dim,
        })
    }

    pub fn next_batch(&mut self, device: &Device) -> candle_core::Result<Batch> {
        let FlatBatch { inputs, targets } = self.assembler.next_batch();
        let inputs = Tensor::from_vec(inputs, (self.batch_size, self.tokens_dim), device)?;
        let targets = Tensor::from_vec(targets, (self.batch_size, self.tokens_dim), device)?;
        Ok(Batch { inputs, targets })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    /// A re-creatable source over a fixed document list: each call clones the docs into a fresh iterator
    fn fixed_docs(docs: Vec<Vec<u32>>) -> impl Fn() -> std::vec::IntoIter<Vec<u32>> {
        move || docs.clone().into_iter()
    }

    /// A byte-level tokenizer: a 256-entry vocab where each token is a single byte, so `encode` maps byte `b` to id `b`.
    fn byte_tokenizer(vocab_file: &mut tempfile::NamedTempFile) -> BpeTokenizer {
        use base64::Engine;
        use base64::engine::general_purpose::STANDARD;
        for b in 0u32..256 {
            writeln!(vocab_file, "{} {}", STANDARD.encode([b as u8]), b).unwrap();
        }
        vocab_file.flush().unwrap();
        BpeTokenizer::from_file(vocab_file.path()).unwrap()
    }

    fn write_shard(path: &Path, texts: Vec<Option<&str>>) {
        use std::sync::Arc;

        use arrow_array::{ArrayRef, RecordBatch, StringArray};
        use parquet::arrow::ArrowWriter;

        let arr: ArrayRef = Arc::new(StringArray::from(texts));
        let batch = RecordBatch::try_from_iter([("text", arr)]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn byte_id(b: u8) -> u32 {
        b as u32
    }

    #[test]
    fn single_long_doc_chunks_into_expected_rows() {
        // B=2, T=2 → row_len=3, batch = 6 tokens.
        // rows: [0,1,2] [3,4,5] → inputs [0,1 | 3,4], targets [1,2 | 4,5].
        let mut p = BatchAssembler::new(fixed_docs(vec![(0u32..12).collect()]), 2, 2);
        let FlatBatch { inputs, targets } = p.next_batch();
        assert_eq!(inputs, vec![0, 1, 3, 4]);
        assert_eq!(targets, vec![1, 2, 4, 5]);
    }

    #[test]
    fn short_docs_concatenate_across_row_boundary() {
        // Four 2-token docs form the stream 0..8; the doc boundaries (at 2,4,6)
        // fall inside/across the rows, proving documents are packed end-to-end.
        let docs = vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
        let mut p = BatchAssembler::new(fixed_docs(docs), 2, 2);
        let FlatBatch { inputs, targets } = p.next_batch();
        assert_eq!(inputs, vec![0, 1, 3, 4]);
        assert_eq!(targets, vec![1, 2, 4, 5]);
    }

    #[test]
    fn targets_are_inputs_shifted_by_one_within_each_row() {
        // B=2, T=3. Within a row, targets[i] must equal inputs[i+1].
        let (b, t) = (2usize, 3usize);
        let mut p = BatchAssembler::new(fixed_docs(vec![(0u32..100).collect()]), b, t);
        let FlatBatch { inputs, targets } = p.next_batch();
        for r in 0..b {
            for i in 0..t - 1 {
                assert_eq!(
                    targets[r * t + i],
                    inputs[r * t + i + 1] as i64,
                    "row {r} position {i}"
                );
            }
        }
    }

    #[test]
    fn pair_lengths_are_exactly_b_times_t() {
        let (b, t) = (3usize, 4usize);
        let mut p = BatchAssembler::new(fixed_docs(vec![(0u32..1000).collect()]), b, t);
        let FlatBatch { inputs, targets } = p.next_batch();
        assert_eq!(inputs.len(), b * t);
        assert_eq!(targets.len(), b * t);
    }

    #[test]
    fn leftover_tokens_carry_into_next_batch_without_loss() {
        // B=1, T=2 → 3 tokens per batch. Two batches over a 9-token stream must
        // reconstruct the stream prefix exactly — no token dropped or repeated.
        let mut p = BatchAssembler::new(fixed_docs(vec![(0u32..9).collect()]), 1, 2);
        let mut seen = p.next_rows();
        seen.extend(p.next_rows());
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn cycles_through_a_finite_source_for_multiple_epochs() {
        // Source = 5 tokens/epoch; B=1, T=1 → 2 tokens/batch. Six batches need
        // 12 tokens, forcing wraps. Expect the cyclic stream 0,1,2,3,4,0,1,...
        let mut p = BatchAssembler::new(fixed_docs(vec![(0u32..5).collect()]), 1, 1);
        let mut seen = Vec::new();
        for _ in 0..6 {
            seen.extend(p.next_rows());
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "empty corpus")]
    fn empty_source_panics_instead_of_hanging() {
        let mut p = BatchAssembler::new(fixed_docs(vec![]), 1, 1);
        let _ = p.next_batch();
    }

    #[test]
    fn split_rows_shifts_each_row_independently() {
        // Two rows of T+1=3: [10,11,12] and [20,21,22].
        let FlatBatch { inputs, targets } = split_rows(&[10, 11, 12, 20, 21, 22], 2, 2);
        assert_eq!(inputs, vec![10, 11, 20, 21]);
        assert_eq!(targets, vec![11, 12, 21, 22]);
    }

    #[test]
    fn encode_docs_tokenizes_each_nonnull_document() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("0.parquet");
        write_shard(&shard, vec![Some("ab"), None, Some("cd")]);

        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vocab_file);

        let docs: Vec<Vec<u32>> = encode_docs(vec![shard], &tok).collect();
        assert_eq!(
            docs,
            vec![
                vec![byte_id(b'a'), byte_id(b'b')],
                vec![byte_id(b'c'), byte_id(b'd')],
            ]
        );
    }

    #[test]
    fn encode_docs_factory_yields_identical_fresh_passes() {
        // The cycling primitive: calling the factory twice must replay the same
        // documents, so the BatchAssembler can restart for a new epoch.
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("0.parquet");
        write_shard(&shard, vec![Some("hello"), Some("world")]);

        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vocab_file);

        let factory = || encode_docs(vec![shard.clone()], &tok);
        let first: Vec<Vec<u32>> = factory().collect();
        let second: Vec<Vec<u32>> = factory().collect();
        assert!(!first.is_empty());
        assert_eq!(first, second);
    }

    /// Build a two-shard corpus (so a train/val split exists) where the chosen
    /// split's single shard holds `text`, then open a `DataLoader` over it.
    fn loader_over<'a>(
        dir: &Path,
        tok: &'a BpeTokenizer,
        split: Split,
        text: &str,
        b: usize,
        t: usize,
    ) -> DataLoader<'a> {
        // Sorted order: "0.parquet" = train, "1.parquet" = val.
        let (train_text, val_text) = match split {
            Split::Val => ("filler", text),
            Split::Train => (text, "filler"),
        };
        write_shard(&dir.join("0.parquet"), vec![Some(train_text)]);
        write_shard(&dir.join("1.parquet"), vec![Some(val_text)]);
        DataLoader::open(dir, split, tok, b, t).unwrap()
    }

    #[test]
    fn next_batch_has_expected_shapes_and_dtypes() {
        use candle_core::{DType, Device};

        let dir = tempfile::tempdir().unwrap();
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vocab_file);
        let mut loader = loader_over(dir.path(), &tok, Split::Val, "abcdefghij", 2, 3);

        let batch = loader.next_batch(&Device::Cpu).unwrap();
        assert_eq!(batch.inputs.dims(), &[2, 3]);
        assert_eq!(batch.targets.dims(), &[2, 3]);
        assert_eq!(batch.inputs.dtype(), DType::U32);
        assert_eq!(batch.targets.dtype(), DType::I64);
    }

    #[test]
    fn next_batch_packs_and_shifts_expected_token_ids() {
        use candle_core::Device;

        let dir = tempfile::tempdir().unwrap();
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vocab_file);
        // "abcde" → byte ids [a..e]; B=1, T=2 → first row = first 3 ids.
        let mut loader = loader_over(dir.path(), &tok, Split::Val, "abcde", 1, 2);

        let batch = loader.next_batch(&Device::Cpu).unwrap();
        let (a, b, c) = (byte_id(b'a'), byte_id(b'b'), byte_id(b'c'));
        assert_eq!(batch.inputs.to_vec2::<u32>().unwrap(), vec![vec![a, b]]);
        assert_eq!(
            batch.targets.to_vec2::<i64>().unwrap(),
            vec![vec![b as i64, c as i64]]
        );
    }

    #[test]
    fn next_batch_cycles_past_end_of_corpus() {
        use candle_core::Device;

        let dir = tempfile::tempdir().unwrap();
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vocab_file);
        // 4-token val shard, 2 tokens/batch → 5 batches need 10 tokens (>1 epoch).
        let mut loader = loader_over(dir.path(), &tok, Split::Val, "abcd", 1, 1);

        for _ in 0..5 {
            let batch = loader.next_batch(&Device::Cpu).unwrap();
            assert_eq!(batch.inputs.dims(), &[1, 1]);
        }
    }

    #[test]
    fn loader_feeds_model_and_loss_end_to_end() {
        // Capstone: loader → Gpt::forward → cross_entropy yields a finite loss,
        // closing the loop the training/eval step will run.
        use crate::model::{Gpt, GptConfig, Reduction, cross_entropy};
        use candle_core::{DType, Device};
        use candle_nn::{VarBuilder, VarMap};

        let dir = tempfile::tempdir().unwrap();
        let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
        let tok = byte_tokenizer(&mut vocab_file);
        let (b, t) = (2usize, 4usize);
        let mut loader =
            loader_over(dir.path(), &tok, Split::Val, "abcdefghijklmnop", b, t);

        let dev = Device::Cpu;
        let batch = loader.next_batch(&dev).unwrap();

        let cfg = GptConfig {
            vocab_size: 256, // byte ids are 0..=255
            sequence_len: t,
            n_layer: 1,
            n_head: 1,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        };
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
        let model = Gpt::new(cfg, vb).unwrap();

        let logits = model.forward(&batch.inputs).unwrap();
        assert_eq!(logits.dims(), &[b, t, 256]);
        let loss = cross_entropy(&logits, &batch.targets, -1, Reduction::Mean)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.is_finite(), "expected finite loss, got {loss}");
    }
}
