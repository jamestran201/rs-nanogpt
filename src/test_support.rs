//! Shared fixtures for unit tests across modules (compiled only under `cfg(test)`).

use std::io::Write;
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use crate::tokenizer::BpeTokenizer;

/// A byte-level tokenizer: one token per byte value, no merges.
pub(crate) fn byte_tokenizer(vocab_file: &mut tempfile::NamedTempFile) -> BpeTokenizer {
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
