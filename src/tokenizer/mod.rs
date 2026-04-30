use std::fs;
use std::io;
use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub type TokenId = u32;

const REGEX_PATTERNS: &[&str] = &[
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
];

/// Returns the prefix of `text` that fits within `remaining` bytes,
/// truncated at the nearest UTF-8 char boundary. Returns `None` if
/// the budget is too small to fit even one char.
fn fit_within_budget(text: &str, remaining: usize) -> Option<&str> {
    if text.len() <= remaining {
        return Some(text);
    }
    let cut = text.floor_char_boundary(remaining);
    if cut == 0 { None } else { Some(&text[..cut]) }
}

pub struct BpeTokenizerTrainer {
    corpus_path: PathBuf,
    max_chars: usize,
    pre_tokenize_pattern: Regex
}

impl BpeTokenizerTrainer {
    pub fn new(corpus_path: impl Into<PathBuf>, max_chars: usize) -> Self {
        let pattern =
            Regex::new(&REGEX_PATTERNS.join("|")).expect("Built-in regex pattern should be valid");

        Self {
            corpus_path: corpus_path.into(),
            max_chars,
            pre_tokenize_pattern: pattern,
        }
    }

    fn pre_tokenize<'a>(pattern: Regex, text: &'a str) -> Vec<&'a str> {
        let mut pieces = Vec::new();
        let mut start = 0;
        while let Some(m) = pattern
            .find_from_pos(text, start)
            .expect("Unexpected regex error in pre_tokenize")
        {
            pieces.push(&text[m.start()..m.end()]);
            start = m.end();
        }
        pieces
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
            files: parquet_files.into_iter(),
            state: State::NeedFile,
            chars_read: 0,
            max_chars: self.max_chars,
        })
    }
}

enum State {
    NeedFile,
    FetchingBatch(ParquetRecordBatchReader),
    InBatch {
        reader: ParquetRecordBatchReader,
        batch: arrow_array::RecordBatch,
        row_idx: usize,
    },
    Done,
}

struct BatchResult {
    /// The item to yield, if any.
    yielded: Option<io::Result<String>>,
    /// True if the byte budget is exhausted — iterator should terminate after this.
    budget_exhausted: bool,
    /// Row index to resume from; only meaningful when the caller decides to continue.
    next_row_idx: usize,
}

fn read_from_batch(
    batch: &arrow_array::RecordBatch,
    start_row: usize,
    remaining: usize,
) -> BatchResult {
    let Some(text_col) = batch.column_by_name("text") else {
        return BatchResult {
            yielded: Some(Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "missing 'text' column",
            ))),
            budget_exhausted: false,
            next_row_idx: start_row,
        };
    };
    let strings = text_col.as_string::<i32>();

    let mut row_idx = start_row;
    while row_idx < strings.len() {
        let i = row_idx;
        row_idx += 1;

        if strings.is_null(i) {
            continue;
        }

        let text = strings.value(i);
        let Some(prefix) = fit_within_budget(text, remaining) else {
            return BatchResult {
                yielded: None,
                budget_exhausted: true,
                next_row_idx: row_idx,
            };
        };

        let truncated = prefix.len() < text.len();
        return BatchResult {
            yielded: Some(Ok(prefix.to_string())),
            budget_exhausted: truncated,
            next_row_idx: row_idx,
        };
    }

    BatchResult {
        yielded: None,
        budget_exhausted: false,
        next_row_idx: row_idx,
    }
}

pub struct CorpusIter {
    files: std::vec::IntoIter<PathBuf>,
    state: State,
    chars_read: usize,
    max_chars: usize,
}

impl Iterator for CorpusIter {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match std::mem::replace(&mut self.state, State::Done) {
                State::Done => return None,

                State::NeedFile => {
                    let path = self.files.next()?;
                    let file = match fs::File::open(path) {
                        Ok(f) => f,
                        Err(e) => return Some(Err(e)),
                    };
                    let reader = match ParquetRecordBatchReaderBuilder::try_new(file)
                        .and_then(|b| b.build())
                    {
                        Ok(r) => r,
                        Err(e) => {
                            return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e)));
                        }
                    };
                    self.state = State::FetchingBatch(reader);
                }

                State::FetchingBatch(mut reader) => match reader.next() {
                    Some(Ok(batch)) => {
                        self.state = State::InBatch {
                            reader,
                            batch,
                            row_idx: 0,
                        };
                    }
                    Some(Err(e)) => {
                        return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e)));
                    }
                    None => {
                        self.state = State::NeedFile;
                    }
                },

                State::InBatch {
                    reader,
                    batch,
                    row_idx,
                } => {
                    let remaining = self.max_chars - self.chars_read;
                    let result = read_from_batch(&batch, row_idx, remaining);
                    match result.yielded {
                        Some(Ok(text)) => {
                            self.chars_read += text.len();
                            if !result.budget_exhausted {
                                self.state = State::InBatch {
                                    reader,
                                    batch,
                                    row_idx: result.next_row_idx,
                                };
                            }
                            return Some(Ok(text));
                        }
                        Some(Err(e)) => return Some(Err(e)),
                        None => {
                            if !result.budget_exhausted {
                                self.state = State::FetchingBatch(reader);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_within_budget_fits_when_under_budget() {
        assert_eq!(fit_within_budget("hello", 100), Some("hello"));
        assert_eq!(fit_within_budget("hello", 5), Some("hello"));
    }

    #[test]
    fn fit_within_budget_truncates_at_byte_boundary() {
        assert_eq!(fit_within_budget("hello world", 5), Some("hello"));
    }

    #[test]
    fn fit_within_budget_truncates_at_char_boundary() {
        // "café" is 5 bytes (é is 2 bytes); budget 4 falls inside é,
        // so the cut should snap back to byte 3.
        assert_eq!(fit_within_budget("café", 4), Some("caf"));
    }

    #[test]
    fn fit_within_budget_returns_none_when_no_char_fits() {
        assert_eq!(fit_within_budget("anything", 0), None);
        // 2-byte char with a 1-byte budget: nothing fits.
        assert_eq!(fit_within_budget("é", 1), None);
    }

    fn make_pattern() -> Regex {
        Regex::new(&REGEX_PATTERNS.join("|")).unwrap()
    }

    #[test]
    fn pre_tokenize_empty_string() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), "");
        assert!(pieces.is_empty());
    }

    #[test]
    fn pre_tokenize_splits_words_with_leading_space() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), "hello world");
        assert_eq!(pieces, vec!["hello", " world"]);
    }

    #[test]
    fn pre_tokenize_groups_digits_in_chunks_of_three() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), "12345");
        assert_eq!(pieces, vec!["123", "45"]);
    }

    #[test]
    fn pre_tokenize_keeps_contractions_attached() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), "don't");
        assert_eq!(pieces, vec!["don't"]);
    }

    #[test]
    fn pre_tokenize_pieces_concatenate_to_input() {
        let input = "Hello, World! It's 2026, isn't it?";
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), input);
        assert_eq!(pieces.concat(), input);
    }

    #[test]
    fn pre_tokenize_handles_newlines_without_dropping_bytes() {
        let input = "hello\n\nworld";
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), input);
        assert_eq!(pieces.concat(), input);
    }

    #[test]
    fn pre_tokenize_separates_punctuation_from_words() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(make_pattern(), "hi!");
        assert_eq!(pieces, vec!["hi", "!"]);
    }

    #[test]
    fn new_initializes_pre_tokenize_pattern() {
        let trainer = BpeTokenizerTrainer::new("data", 100);
        let pieces = BpeTokenizerTrainer::pre_tokenize(
            trainer.pre_tokenize_pattern.clone(),
            "hello world",
        );
        assert_eq!(pieces, vec!["hello", " world"]);
    }

    #[test]
    fn read_corpus_not_a_directory() {
        let trainer = BpeTokenizerTrainer::new("Cargo.toml", 1000);
        let err = trainer
            .read_corpus()
            .err()
            .expect("should fail for non-directory");
        assert_eq!(err.kind(), io::ErrorKind::NotADirectory);
    }

    #[test]
    fn read_corpus_respects_max_chars() {
        let trainer = BpeTokenizerTrainer::new("data", 10000);
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
