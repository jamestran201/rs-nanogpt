use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub type TokenId = u32;

type Pair = (Vec<u8>, Vec<u8>);

struct PreTokenState {
    bytes: Vec<Vec<u8>>,
    next: Vec<Option<usize>>,
    prev: Vec<Option<usize>>,
    count: u64,
}

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
    pre_tokenize_pattern: Regex,
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

    fn pre_tokenize<'a>(pattern: &Regex, text: &'a str) -> Vec<&'a str> {
        let mut pretokens = Vec::new();
        let mut start = 0;
        while let Some(m) = pattern
            .find_from_pos(text, start)
            .expect("Unexpected regex error in pre_tokenize")
        {
            pretokens.push(&text[m.start()..m.end()]);
            start = m.end();
        }
        pretokens
    }

    fn count_pretokens<I>(pattern: &Regex, corpus_iter: I) -> io::Result<HashMap<String, u64>>
    where
        I: IntoIterator<Item = io::Result<String>>,
    {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for doc in corpus_iter {
            let doc = doc?;
            for pretoken in Self::pre_tokenize(pattern, &doc) {
                if let Some(c) = counts.get_mut(pretoken) {
                    *c += 1;
                } else {
                    counts.insert(pretoken.to_string(), 1);
                }
            }
        }
        Ok(counts)
    }

    fn init_pretoken_states(counts: HashMap<String, u64>) -> Vec<PreTokenState> {
        counts
            .into_iter()
            .map(|(pretoken, count)| {
                let n = pretoken.len();
                let bytes: Vec<Vec<u8>> = pretoken.as_bytes().iter().map(|b| vec![*b]).collect();
                let next: Vec<Option<usize>> = (0..n)
                    .map(|i| if i + 1 < n { Some(i + 1) } else { None })
                    .collect();
                let prev: Vec<Option<usize>> = (0..n)
                    .map(|i| if i > 0 { Some(i - 1) } else { None })
                    .collect();
                PreTokenState {
                    bytes,
                    next,
                    prev,
                    count,
                }
            })
            .collect()
    }

    fn init_pair_tables(
        states: &[PreTokenState],
    ) -> (HashMap<Pair, u64>, HashMap<Pair, Vec<(usize, usize)>>) {
        let mut pair_counts: HashMap<Pair, u64> = HashMap::new();
        let mut pair_locations: HashMap<Pair, Vec<(usize, usize)>> = HashMap::new();

        for (state_idx, state) in states.iter().enumerate() {
            for left in 0..state.bytes.len() {
                let Some(right) = state.next[left] else { continue };
                let pair = (state.bytes[left].clone(), state.bytes[right].clone());
                *pair_counts.entry(pair.clone()).or_insert(0) += state.count;
                pair_locations.entry(pair).or_default().push((state_idx, left));
            }
        }

        (pair_counts, pair_locations)
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
            state: CorpusIterState::NeedFile,
            chars_read: 0,
            max_chars: self.max_chars,
        })
    }

    fn prepare_pretoken_states(&self) -> io::Result<Vec<PreTokenState>> {
        let corpus_iter = self.read_corpus()?;
        let counts = Self::count_pretokens(&self.pre_tokenize_pattern, corpus_iter)?;
        Ok(Self::init_pretoken_states(counts))
    }
}

enum CorpusIterState {
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
    state: CorpusIterState,
    chars_read: usize,
    max_chars: usize,
}

impl Iterator for CorpusIter {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match std::mem::replace(&mut self.state, CorpusIterState::Done) {
                CorpusIterState::Done => return None,

                CorpusIterState::NeedFile => {
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
                    self.state = CorpusIterState::FetchingBatch(reader);
                }

                CorpusIterState::FetchingBatch(mut reader) => match reader.next() {
                    Some(Ok(batch)) => {
                        self.state = CorpusIterState::InBatch {
                            reader,
                            batch,
                            row_idx: 0,
                        };
                    }
                    Some(Err(e)) => {
                        return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e)));
                    }
                    None => {
                        self.state = CorpusIterState::NeedFile;
                    }
                },

                CorpusIterState::InBatch {
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
                                self.state = CorpusIterState::InBatch {
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
                                self.state = CorpusIterState::FetchingBatch(reader);
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
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), "");
        assert!(pieces.is_empty());
    }

    #[test]
    fn pre_tokenize_splits_words_with_leading_space() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), "hello world");
        assert_eq!(pieces, vec!["hello", " world"]);
    }

    #[test]
    fn pre_tokenize_groups_digits_in_chunks_of_three() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), "12345");
        assert_eq!(pieces, vec!["123", "45"]);
    }

    #[test]
    fn pre_tokenize_keeps_contractions_attached() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), "don't");
        assert_eq!(pieces, vec!["don't"]);
    }

    #[test]
    fn pre_tokenize_pieces_concatenate_to_input() {
        let input = "Hello, World! It's 2026, isn't it?";
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), input);
        assert_eq!(pieces.concat(), input);
    }

    #[test]
    fn pre_tokenize_handles_newlines_without_dropping_bytes() {
        let input = "hello\n\nworld";
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), input);
        assert_eq!(pieces.concat(), input);
    }

    #[test]
    fn pre_tokenize_separates_punctuation_from_words() {
        let pieces = BpeTokenizerTrainer::pre_tokenize(&make_pattern(), "hi!");
        assert_eq!(pieces, vec!["hi", "!"]);
    }

    fn counts_from(entries: &[(&str, u64)]) -> HashMap<String, u64> {
        entries.iter().map(|(s, c)| (s.to_string(), *c)).collect()
    }

    #[test]
    fn init_pretoken_states_empty() {
        let words = BpeTokenizerTrainer::init_pretoken_states(HashMap::new());
        assert!(words.is_empty());
    }

    #[test]
    fn init_pretoken_states_single_piece_builds_correct_links() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("hi", 1)]));
        assert_eq!(words.len(), 1);
        let w = &words[0];
        assert_eq!(w.bytes, vec![vec![b'h'], vec![b'i']]);
        assert_eq!(w.next, vec![Some(1), None]);
        assert_eq!(w.prev, vec![None, Some(0)]);
        assert_eq!(w.count, 1);
    }

    #[test]
    fn init_pretoken_states_single_byte_piece() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("a", 1)]));
        assert_eq!(words.len(), 1);
        let w = &words[0];
        assert_eq!(w.bytes, vec![vec![b'a']]);
        assert_eq!(w.next, vec![None]);
        assert_eq!(w.prev, vec![None]);
        assert_eq!(w.count, 1);
    }

    #[test]
    fn init_pretoken_states_preserves_count() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("the", 3)]));
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].count, 3);
        assert_eq!(words[0].bytes, vec![vec![b't'], vec![b'h'], vec![b'e']]);
    }

    #[test]
    fn init_pretoken_states_distinct_pieces() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("a", 2), ("b", 1)]));
        assert_eq!(words.len(), 2);
        let counts: HashMap<Vec<u8>, u64> = words
            .iter()
            .map(|w| (w.bytes.iter().flatten().copied().collect(), w.count))
            .collect();
        assert_eq!(counts.get(&vec![b'a']), Some(&2));
        assert_eq!(counts.get(&vec![b'b']), Some(&1));
    }

    #[test]
    fn init_pretoken_states_splits_multibyte_chars_into_bytes() {
        // "é" is 0xC3 0xA9 in UTF-8 — BPE operates on bytes, so 2 nodes.
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("é", 1)]));
        assert_eq!(words.len(), 1);
        let w = &words[0];
        assert_eq!(w.bytes, vec![vec![0xC3], vec![0xA9]]);
        assert_eq!(w.next, vec![Some(1), None]);
        assert_eq!(w.prev, vec![None, Some(0)]);
    }

    #[test]
    fn init_pair_tables_empty_input() {
        let (counts, locs) = BpeTokenizerTrainer::init_pair_tables(&[]);
        assert!(counts.is_empty());
        assert!(locs.is_empty());
    }

    #[test]
    fn init_pair_tables_single_byte_state_has_no_pairs() {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("a", 4)]));
        let (counts, locs) = BpeTokenizerTrainer::init_pair_tables(&states);
        assert!(counts.is_empty());
        assert!(locs.is_empty());
    }

    #[test]
    fn init_pair_tables_two_byte_state_records_one_pair() {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("ab", 5)]));
        let (counts, locs) = BpeTokenizerTrainer::init_pair_tables(&states);
        let pair = (vec![b'a'], vec![b'b']);
        assert_eq!(counts.get(&pair), Some(&5));
        assert_eq!(locs.get(&pair), Some(&vec![(0, 0)]));
        assert_eq!(counts.len(), 1);
    }

    #[test]
    fn init_pair_tables_three_byte_state_records_adjacent_pairs() {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("abc", 2)]));
        let (counts, locs) = BpeTokenizerTrainer::init_pair_tables(&states);
        let ab = (vec![b'a'], vec![b'b']);
        let bc = (vec![b'b'], vec![b'c']);
        assert_eq!(counts.get(&ab), Some(&2));
        assert_eq!(counts.get(&bc), Some(&2));
        assert_eq!(locs.get(&ab), Some(&vec![(0, 0)]));
        assert_eq!(locs.get(&bc), Some(&vec![(0, 1)]));
    }

    #[test]
    fn init_pair_tables_aggregates_shared_pair_across_states() {
        // Two distinct states sharing the (a,b) pair, with counts 3 and 4.
        let states =
            BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("ab", 3), ("abx", 4)]));
        let (counts, locs) = BpeTokenizerTrainer::init_pair_tables(&states);
        let ab = (vec![b'a'], vec![b'b']);
        assert_eq!(counts.get(&ab), Some(&7));
        let ab_locs = locs.get(&ab).unwrap();
        assert_eq!(ab_locs.len(), 2);
        // HashMap iteration order over states is non-deterministic — assert as a set.
        let set: std::collections::HashSet<_> = ab_locs.iter().copied().collect();
        assert!(set.contains(&(0, 0)));
        assert!(set.contains(&(1, 0)));
    }

    #[test]
    fn count_pretokens_aggregates_across_documents() {
        let pattern = make_pattern();
        let docs = vec![Ok("hello world".to_string()), Ok("hello hello".to_string())];
        let counts = BpeTokenizerTrainer::count_pretokens(&pattern, docs).unwrap();
        // "hello" once at start of doc1, "hello" at start of doc2, " hello" once.
        // " world" once.
        assert_eq!(counts.get("hello"), Some(&2));
        assert_eq!(counts.get(" hello"), Some(&1));
        assert_eq!(counts.get(" world"), Some(&1));
    }

    #[test]
    fn count_pretokens_propagates_io_errors() {
        let pattern = make_pattern();
        let docs: Vec<io::Result<String>> = vec![
            Ok("ok".to_string()),
            Err(io::Error::new(io::ErrorKind::Other, "boom")),
        ];
        let result = BpeTokenizerTrainer::count_pretokens(&pattern, docs);
        assert!(result.is_err());
    }

    #[test]
    fn count_pretokens_empty_iterator() {
        let pattern = make_pattern();
        let docs: Vec<io::Result<String>> = vec![];
        let counts = BpeTokenizerTrainer::count_pretokens(&pattern, docs).unwrap();
        assert!(counts.is_empty());
    }

    #[test]
    fn new_initializes_pre_tokenize_pattern() {
        let trainer = BpeTokenizerTrainer::new("data", 100);
        let pieces =
            BpeTokenizerTrainer::pre_tokenize(&trainer.pre_tokenize_pattern, "hello world");
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
    fn prepare_pretoken_states_produces_nonempty_states() {
        let trainer = BpeTokenizerTrainer::new("data", 10000);
        let states = trainer.prepare_pretoken_states().unwrap();
        assert!(!states.is_empty());
        for s in &states {
            assert!(!s.bytes.is_empty());
            assert!(s.count >= 1);
        }
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
