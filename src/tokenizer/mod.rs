use std::fs;
use std::io;
use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub type TokenId = u32;

pub struct BpeTokenizerTrainer {
    corpus_path: PathBuf,
    max_chars: usize,
}

impl BpeTokenizerTrainer {
    pub fn new(corpus_path: PathBuf, max_chars: usize) -> Self {
        Self {
            corpus_path,
            max_chars,
        }
    }

    pub fn read_corpus(&self) -> io::Result<CorpusIter> {
        if !self.corpus_path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("{} is not a directory", self.corpus_path.display()),
            ));
        }

        let mut parquet_files: Vec<PathBuf> = fs::read_dir(&self.corpus_path)?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                if path.extension().and_then(|e| e.to_str()) == Some("parquet") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        parquet_files.sort();

        Ok(CorpusIter {
            parquet_files,
            file_idx: 0,
            reader: None,
            batch: None,
            row_idx: 0,
            chars_read: 0,
            max_chars: self.max_chars,
            done: false,
        })
    }
}

use parquet::arrow::arrow_reader::ParquetRecordBatchReader;

pub struct CorpusIter {
    parquet_files: Vec<PathBuf>,
    file_idx: usize,
    reader: Option<ParquetRecordBatchReader>,
    batch: Option<arrow_array::RecordBatch>,
    row_idx: usize,
    chars_read: usize,
    max_chars: usize,
    done: bool,
}

impl Iterator for CorpusIter {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done {
                return None;
            }

            // Try to yield from the current batch
            if let Some(batch) = &self.batch {
                let text_col = match batch.column_by_name("text") {
                    Some(col) => col,
                    None => {
                        self.done = true;
                        return Some(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "missing 'text' column",
                        )));
                    }
                };
                let strings = text_col.as_string::<i32>();

                while self.row_idx < strings.len() {
                    let i = self.row_idx;
                    self.row_idx += 1;

                    if strings.is_null(i) {
                        continue;
                    }
                    let text = strings.value(i);
                    let remaining = self.max_chars - self.chars_read;
                    if text.len() > remaining {
                        let truncated = &text[..text.floor_char_boundary(remaining)];
                        self.done = true;
                        if !truncated.is_empty() {
                            return Some(Ok(truncated.to_string()));
                        }
                        return None;
                    }
                    self.chars_read += text.len();
                    return Some(Ok(text.to_string()));
                }

                // Batch exhausted, fall through to load next batch
                self.batch = None;
            }

            // Try to get the next batch from the current reader
            if let Some(reader) = &mut self.reader {
                match reader.next() {
                    Some(Ok(batch)) => {
                        self.batch = Some(batch);
                        self.row_idx = 0;
                        continue;
                    }
                    Some(Err(e)) => {
                        self.done = true;
                        return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e)));
                    }
                    None => {
                        // Reader exhausted, fall through to open next file
                        self.reader = None;
                    }
                }
            }

            // Try to open the next parquet file
            if self.file_idx >= self.parquet_files.len() {
                self.done = true;
                return None;
            }

            let path = &self.parquet_files[self.file_idx];
            self.file_idx += 1;

            let file = match fs::File::open(path) {
                Ok(f) => f,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };
            let reader =
                match ParquetRecordBatchReaderBuilder::try_new(file).and_then(|b| b.build()) {
                    Ok(r) => r,
                    Err(e) => {
                        self.done = true;
                        return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e)));
                    }
                };
            self.reader = Some(reader);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_corpus_not_a_directory() {
        let trainer = BpeTokenizerTrainer::new(PathBuf::from("Cargo.toml"), 1000);
        let err = trainer
            .read_corpus()
            .err()
            .expect("should fail for non-directory");
        assert_eq!(err.kind(), io::ErrorKind::NotADirectory);
    }

    #[test]
    fn read_corpus_respects_max_chars() {
        let trainer = BpeTokenizerTrainer::new(PathBuf::from("data"), 10000);
        let corpus: Vec<String> = trainer
            .read_corpus()
            .unwrap()
            .collect::<io::Result<_>>()
            .unwrap();
        let total_chars: usize = corpus.iter().map(|s| s.len()).sum();
        assert!(!corpus.is_empty());
        assert!(total_chars <= 10000);
    }
}
