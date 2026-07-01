use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use crate::tokenizer::BpeTokenizer;

use super::parquet::{ShardTextIter, Split, list_shards, shards_for_split};

/// Size of the packing pool when constructed via [`DataLoader::open`]. A larger
/// pool gives the best-fit packer more candidates per gap, so less is cropped.
/// Matches nanochat's default.
const DEFAULT_BUFFER_SIZE: usize = 1000;

pub(crate) struct BatchAssembler<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = Vec<u32>>,
{
    make_docs: F,
    docs: I,
    pool: Vec<Vec<u32>>,
    buffer_size: usize,
    batch_rows: usize, // B
    row_len: usize,    // T + 1
    docs_produced_this_pass: usize,
}

impl<F, I> BatchAssembler<F, I>
where
    F: Fn() -> I,
    I: Iterator<Item = Vec<u32>>,
{
    pub(crate) fn new(
        make_docs: F,
        batch_rows: usize,
        tokens_dim: usize,
        buffer_size: usize,
    ) -> Self {
        assert!(buffer_size > 0, "buffer_size must be positive");
        let docs = make_docs();
        Self {
            make_docs,
            docs,
            pool: Vec::new(),
            buffer_size,
            batch_rows,
            row_len: tokens_dim + 1,
            docs_produced_this_pass: 0,
        }
    }

    /// Top the pool up to `buffer_size` whole docs, cycling to a fresh pass
    /// (via `make_docs`) whenever the current iterator is exhausted.
    fn refill_pool(&mut self) {
        while self.pool.len() < self.buffer_size {
            match self.docs.next() {
                Some(doc) => {
                    self.docs_produced_this_pass += 1;
                    self.pool.push(doc);
                }
                None => {
                    assert!(
                        self.docs_produced_this_pass > 0,
                        "data source produced no documents in a full pass over the source — empty corpus"
                    );
                    self.docs = (self.make_docs)();
                    self.docs_produced_this_pass = 0;
                }
            }
        }
    }

    /// Pack one row of exactly `row_len` (= T+1) tokens by best fit, appending it
    /// to `out`: place the largest pooled doc that fits the remaining gap; if none
    /// fits, crop the shortest doc to fill the gap exactly and drop its tail.
    fn pack_row(&mut self, out: &mut Vec<u32>) {
        let mut pos = 0;
        while pos < self.row_len {
            self.refill_pool();
            let remaining = self.row_len - pos;

            let best_fit_idx = self
                .pool
                .iter()
                .enumerate()
                .filter(|(_, doc)| doc.len() <= remaining)
                .max_by_key(|(_, doc)| doc.len())
                .map(|(idx, _)| idx);

            match best_fit_idx {
                Some(idx) => {
                    let doc = self.pool.swap_remove(idx);
                    out.extend_from_slice(&doc);
                    pos += doc.len();
                }
                None => {
                    let idx = self
                        .pool
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, doc)| doc.len())
                        .map(|(idx, _)| idx)
                        .expect("pool non-empty after refill_pool");
                    let doc = self.pool.swap_remove(idx);
                    out.extend_from_slice(&doc[..remaining]);
                    pos += remaining;
                }
            }
        }
    }

    /// Build one batch's worth of rows: `B` rows of `T+1` tokens
    fn next_rows(&mut self) -> Vec<u32> {
        let mut rows = Vec::with_capacity(self.batch_rows * self.row_len);
        for _ in 0..self.batch_rows {
            self.pack_row(&mut rows);
        }
        rows
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
    debug_assert_eq!(
        rows.len(),
        batch_size * row_len,
        "rows must be exactly B * (T+1)"
    );

    let mut inputs = Vec::with_capacity(batch_size * tokens_dim);
    let mut targets = Vec::with_capacity(batch_size * tokens_dim);
    for row in rows.chunks_exact(row_len) {
        inputs.extend_from_slice(&row[..tokens_dim]);
        targets.extend(row[1..].iter().map(|&x| x as i64));
    }
    FlatBatch { inputs, targets }
}

/// One pass's worth of token documents: stream the `text` column of `shards`
/// and tokenize each non-empty document. This is the source the `BatchAssembler` consumes.
pub(crate) fn encode_docs<'a>(
    shards: Vec<PathBuf>,
    tokenizer: &'a BpeTokenizer,
) -> impl Iterator<Item = Vec<u32>> + 'a {
    ShardTextIter::new(shards).filter_map(move |res| {
        let text = res.unwrap_or_else(|e| panic!("failed reading shard text: {e}"));
        let body = tokenizer.encode(&text);
        (!body.is_empty()).then(|| {
            let mut tokens = Vec::with_capacity(body.len() + 1);
            tokens.push(tokenizer.bos_id());
            tokens.extend(body);
            tokens
        })
    })
}

/// One pass's document-token iterator, type-erased so `DataLoader` can name it.
type DocIter<'a> = Box<dyn Iterator<Item = Vec<u32>> + 'a>;
/// Factory that produces a fresh `DocIter` per pass.
/// `BatchAssembler` calls this again when the current pass is exhausted.
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
        Self::open_with_buffer_size(
            dir,
            split,
            tokenizer,
            batch_size,
            tokens_dim,
            DEFAULT_BUFFER_SIZE,
        )
    }

    /// Like [`open`](Self::open) but with an explicit packing-pool size; tests
    /// pass a small pool to avoid recreating the shard source many times.
    pub fn open_with_buffer_size(
        dir: &Path,
        split: Split,
        tokenizer: &'a BpeTokenizer,
        batch_size: usize,
        tokens_dim: usize,
        buffer_size: usize,
    ) -> std::io::Result<Self> {
        let all = list_shards(dir)?;
        let shards = shards_for_split(&all, split)?;
        let factory: DocFactory<'a> =
            Box::new(move || Box::new(encode_docs(shards.clone(), tokenizer)) as DocIter<'a>);
        Ok(Self {
            assembler: BatchAssembler::new(factory, batch_size, tokens_dim, buffer_size),
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

    pub fn take_batches(&mut self, n: usize, device: &Device) -> candle_core::Result<Vec<Batch>> {
        (0..n).map(|_| self.next_batch(device)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::test_support::{byte_tokenizer, write_shard};

    /// A re-creatable source over a fixed document list: each call clones the docs into a fresh iterator
    fn fixed_docs(docs: Vec<Vec<u32>>) -> impl Fn() -> std::vec::IntoIter<Vec<u32>> {
        move || docs.clone().into_iter()
    }

    fn byte_id(b: u8) -> u32 {
        b as u32
    }

    #[test]
    fn over_long_doc_is_cropped_to_row_length() {
        // A doc longer than row_len (12 > 3) never fits whole, so each row is
        // its first row_len tokens; the tail is dropped and the doc recycles.
        // B=2, T=2 → row_len=3; buffer_size=1.
        let mut p = BatchAssembler::new(fixed_docs(vec![(0u32..12).collect()]), 2, 2, 1);
        let FlatBatch { inputs, targets } = p.next_batch();
        assert_eq!(inputs, vec![0, 1, 0, 1]);
        assert_eq!(targets, vec![1, 2, 1, 2]);
    }

    #[test]
    fn best_fit_places_largest_doc_that_fits_each_gap() {
        // B=1, T=4 → row_len=5. Docs of distinct lengths 3, 2, 1 all in the pool.
        // Gap 5 takes the len-3 doc; the remaining gap 2 takes the len-2 doc
        // (not the len-1 doc) → row = [20,21,22, 10,11], filling exactly.
        let docs = vec![vec![10, 11], vec![20, 21, 22], vec![30]];
        let mut p = BatchAssembler::new(fixed_docs(docs), 1, 4, 3);
        let FlatBatch { inputs, targets } = p.next_batch();
        assert_eq!(inputs, vec![20, 21, 22, 10]);
        assert_eq!(targets, vec![21, 22, 10, 11]);
    }

    #[test]
    fn every_packed_row_starts_on_a_doc_boundary() {
        // Each doc starts with a sentinel BOS (99). Best fit only ever begins a
        // row by placing (or cropping from the front of) a whole doc, so every
        // row's first token is a BOS — including the len-6 doc that only crops.
        const BOS: u32 = 99;
        let docs = vec![vec![BOS, 1, 2], vec![BOS, 3], vec![BOS, 4, 5, 6, 7, 8]];
        let (b, t) = (3usize, 3usize); // row_len=4
        let mut p = BatchAssembler::new(fixed_docs(docs), b, t, 3);
        let rows = p.next_rows();
        let row_len = t + 1;
        for r in 0..b {
            assert_eq!(rows[r * row_len], BOS, "row {r} must start on a BOS");
        }
    }

    #[test]
    fn packing_is_deterministic_for_a_fixed_source() {
        let docs: Vec<Vec<u32>> = (0..20u32)
            .map(|i| (0..(1 + i % 5)).map(|j| i * 10 + j).collect())
            .collect();
        let (b, t) = (2usize, 6usize);
        let mut p1 = BatchAssembler::new(fixed_docs(docs.clone()), b, t, 6);
        let mut p2 = BatchAssembler::new(fixed_docs(docs), b, t, 6);
        for _ in 0..3 {
            assert_eq!(p1.next_rows(), p2.next_rows());
        }
    }

    #[test]
    fn cycles_through_a_finite_source_across_many_batches() {
        // A small finite source must keep yielding full batches indefinitely by
        // recreating the pass; cropping/reordering makes the exact token stream
        // implementation-defined, so we only assert it never hangs or short-fills.
        let docs = vec![vec![0, 1], vec![2, 3, 4], vec![5]];
        let (b, t) = (2usize, 3usize); // row_len=4
        let mut p = BatchAssembler::new(fixed_docs(docs), b, t, 3);
        for _ in 0..10 {
            assert_eq!(p.next_rows().len(), b * (t + 1));
        }
    }

    #[test]
    #[should_panic(expected = "empty corpus")]
    fn empty_source_panics_instead_of_hanging() {
        let mut p = BatchAssembler::new(fixed_docs(vec![]), 1, 1, 1);
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

        let tok = byte_tokenizer();

        let docs: Vec<Vec<u32>> = encode_docs(vec![shard], &tok).collect();
        let bos = tok.bos_id();
        assert_eq!(
            docs,
            vec![
                vec![bos, byte_id(b'a'), byte_id(b'b')],
                vec![bos, byte_id(b'c'), byte_id(b'd')],
            ]
        );
    }

    #[test]
    fn encode_docs_factory_yields_identical_fresh_passes() {
        // The cycling primitive: calling the factory twice must replay the same
        // documents, so the BatchAssembler can restart for a new pass.
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("0.parquet");
        write_shard(&shard, vec![Some("hello"), Some("world")]);

        let tok = byte_tokenizer();

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
        // Small pool: the per-split shard holds a single doc, so a large pool
        // would recreate the source many times to fill.
        DataLoader::open_with_buffer_size(dir, split, tok, b, t, 4).unwrap()
    }

    #[test]
    fn next_batch_packs_and_shifts_expected_token_ids() {
        use candle_core::Device;

        let dir = tempfile::tempdir().unwrap();
        let tok = byte_tokenizer();
        // "abcde" → BOS-prefixed stream [bos,a,b,c,d,e]; B=1, T=2 → first row = first 3 ids.
        let mut loader = loader_over(dir.path(), &tok, Split::Val, "abcde", 1, 2);

        let batch = loader.next_batch(&Device::Cpu).unwrap();
        let bos = tok.bos_id();
        let (a, b) = (byte_id(b'a'), byte_id(b'b'));
        assert_eq!(batch.inputs.to_vec2::<u32>().unwrap(), vec![vec![bos, a]]);
        assert_eq!(
            batch.targets.to_vec2::<i64>().unwrap(),
            vec![vec![a as i64, b as i64]]
        );
    }

    #[test]
    fn take_batches_is_deterministic_across_fresh_loaders() {
        use candle_core::Device;

        // Two fresh loaders over the same corpus must snapshot an identical
        // prefix of batches — the property the val-set snapshot relies on.
        let dir = tempfile::tempdir().unwrap();
        let tok = byte_tokenizer();
        let text = "the quick brown fox jumps over the lazy dog";
        let (b, t) = (2usize, 4usize);

        let mut l1 = loader_over(dir.path(), &tok, Split::Val, text, b, t);
        let a = l1.take_batches(3, &Device::Cpu).unwrap();
        let mut l2 = loader_over(dir.path(), &tok, Split::Val, text, b, t);
        let c = l2.take_batches(3, &Device::Cpu).unwrap();

        assert_eq!(a.len(), 3);
        for (x, y) in a.iter().zip(&c) {
            assert_eq!(
                x.inputs.to_vec2::<u32>().unwrap(),
                y.inputs.to_vec2::<u32>().unwrap()
            );
            assert_eq!(
                x.targets.to_vec2::<i64>().unwrap(),
                y.targets.to_vec2::<i64>().unwrap()
            );
        }
    }

    #[test]
    fn loader_feeds_model_and_loss_end_to_end() {
        // Capstone: loader → Gpt::forward → cross_entropy yields a finite loss,
        // closing the loop the training/eval step will run.
        use crate::model::{Reduction, cross_entropy};
        use crate::test_support::tiny_gpt;
        use candle_core::Device;

        let dir = tempfile::tempdir().unwrap();
        let tok = byte_tokenizer();
        let (b, t) = (2usize, 4usize);
        let mut loader = loader_over(dir.path(), &tok, Split::Val, "abcdefghijklmnop", b, t);

        let dev = Device::Cpu;
        let batch = loader.next_batch(&dev).unwrap();

        let (_vm, model) = tiny_gpt(tok.vocab_size(), t);

        let logits = model.forward(&batch.inputs).unwrap();
        assert_eq!(logits.dims(), &[b, t, tok.vocab_size()]);
        let loss = cross_entropy(&logits, &batch.targets, -1, Reduction::Mean)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.is_finite(), "expected finite loss, got {loss}");
    }
}
