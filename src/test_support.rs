//! Shared fixtures for unit tests across modules (compiled only under `cfg(test)`).

use std::io::Write;
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use crate::model::{Gpt, GptConfig};
use crate::tokenizer::BpeTokenizer;

/// A byte-level tokenizer: one token per byte value, no merges.
/// `from_file` reads eagerly, so the backing tempfile can be dropped on return.
pub(crate) fn byte_tokenizer() -> BpeTokenizer {
    let mut vocab_file = tempfile::NamedTempFile::new().unwrap();
    for b in 0u32..256 {
        writeln!(vocab_file, "{} {}", STANDARD.encode([b as u8]), b).unwrap();
    }
    vocab_file.flush().unwrap();
    BpeTokenizer::from_file(vocab_file.path()).unwrap()
}

/// Write a single-column (`text`) parquet shard; `None` entries are null rows.
pub(crate) fn write_shard(path: &Path, texts: Vec<Option<&str>>) {
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

/// A two-shard corpus so a train/val split exists: shard 0 (sorted first) is
/// the Train split's filler, shard 1 the Val split's real text.
pub(crate) fn two_shard_corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        &dir.path().join("0.parquet"),
        vec![Some("filler text here")],
    );
    write_shard(
        &dir.path().join("1.parquet"),
        vec![Some("the quick brown fox jumps over the lazy dog")],
    );
    dir
}

/// A 1-layer, 1-head, 8-wide GPT on CPU for capstone tests.
pub(crate) fn tiny_gpt(vocab_size: usize, sequence_len: usize) -> (VarMap, Gpt) {
    let cfg = GptConfig {
        vocab_size,
        sequence_len,
        n_layer: 1,
        n_head: 1,
        n_embd: 8,
        rope_base: 100_000.0,
        norm_eps: 1e-6,
    };
    let vm = VarMap::new();
    let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
    (vm, model)
}
