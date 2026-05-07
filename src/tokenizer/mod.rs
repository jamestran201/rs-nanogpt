use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use fancy_regex::Regex;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub type TokenId = u32;

const TOMBSTONE: TokenId = TokenId::MAX;

const SINGLE_BYTE_TABLE: [[u8; 1]; 256] = {
    let mut table = [[0u8; 1]; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = [i as u8];
        i += 1;
    }
    table
};

struct Vocab {
    merged: Vec<Vec<u8>>,
}

impl Vocab {
    fn bytes_of(&self, id: TokenId) -> &[u8] {
        if (id as usize) < 256 {
            &SINGLE_BYTE_TABLE[id as usize]
        } else {
            &self.merged[(id as usize) - 256]
        }
    }

    fn push_merge(&mut self, bytes: Vec<u8>) -> TokenId {
        let id = 256 + self.merged.len() as TokenId;
        self.merged.push(bytes);
        id
    }
}

type Pair = (TokenId, TokenId);

struct PreTokenState {
    tokens: Vec<TokenId>,
    next: Vec<Option<usize>>,
    prev: Vec<Option<usize>>,
    count: u64,
}

#[derive(Default)]
struct PairInfo {
    count: u64,
    locations: Vec<(usize, usize)>,
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
                let tokens: Vec<TokenId> =
                    pretoken.as_bytes().iter().map(|b| *b as TokenId).collect();
                let next: Vec<Option<usize>> = (0..n)
                    .map(|i| if i + 1 < n { Some(i + 1) } else { None })
                    .collect();
                let prev: Vec<Option<usize>> = (0..n)
                    .map(|i| if i > 0 { Some(i - 1) } else { None })
                    .collect();
                PreTokenState {
                    tokens,
                    next,
                    prev,
                    count,
                }
            })
            .collect()
    }

    fn init_pair_tables(states: &[PreTokenState]) -> HashMap<Pair, PairInfo> {
        let mut pair_info: HashMap<Pair, PairInfo> = HashMap::new();

        for (state_idx, state) in states.iter().enumerate() {
            for left in 0..state.tokens.len() {
                let Some(right) = state.next[left] else {
                    continue;
                };
                let pair = (state.tokens[left], state.tokens[right]);
                let entry = pair_info.entry(pair).or_default();
                entry.count += state.count;
                entry.locations.push((state_idx, left));
            }
        }

        pair_info
    }

    fn find_best_pair(pair_info: &HashMap<Pair, PairInfo>, vocab: &Vocab) -> Option<Pair> {
        pair_info
            .iter()
            .filter(|(_, info)| info.count > 0)
            .max_by(|(p1, i1), (p2, i2)| {
                i1.count.cmp(&i2.count).then_with(|| {
                    let lhs = (vocab.bytes_of(p1.0), vocab.bytes_of(p1.1));
                    let rhs = (vocab.bytes_of(p2.0), vocab.bytes_of(p2.1));
                    lhs.cmp(&rhs)
                })
            })
            .map(|(pair, _)| *pair)
    }

    fn merge_pair(
        pair: Pair,
        states: &mut [PreTokenState],
        pair_info: &mut HashMap<Pair, PairInfo>,
        vocab: &mut Vocab,
    ) -> TokenId {
        // Allocate the merged byte string once per merge (not per location).
        // Note: this push happens before the live-locations check below; if the
        // assertion at the end fires, the entry is leaked in vocab. That's
        // acceptable because the panic indicates an invariant violation that
        // aborts the trainer.
        let p0_len = vocab.bytes_of(pair.0).len();
        let p1_len = vocab.bytes_of(pair.1).len();
        let mut merged_bytes = Vec::with_capacity(p0_len + p1_len);
        merged_bytes.extend_from_slice(vocab.bytes_of(pair.0));
        merged_bytes.extend_from_slice(vocab.bytes_of(pair.1));
        let merged_id = vocab.push_merge(merged_bytes);

        let locations = pair_info
            .remove(&pair)
            .map(|info| info.locations)
            .unwrap_or_default();

        let mut merged_any = false;

        for (state_idx, left) in locations {
            let state = &mut states[state_idx];

            let Some(right) = state.next[left] else {
                continue;
            };
            if state.tokens[left] != pair.0 || state.tokens[right] != pair.1 {
                continue;
            }

            let before = state.prev[left];
            let after = state.next[right];
            let count = state.count;

            if let Some(b_idx) = before {
                let key = (state.tokens[b_idx], state.tokens[left]);
                if let Some(info) = pair_info.get_mut(&key) {
                    assert!(
                        info.count >= count,
                        "pair count underflow: left-neighbor pair count {} < merge count {}",
                        info.count,
                        count,
                    );
                    info.count -= count;
                }
            }
            if let Some(a_idx) = after {
                let key = (state.tokens[right], state.tokens[a_idx]);
                if let Some(info) = pair_info.get_mut(&key) {
                    assert!(
                        info.count >= count,
                        "pair count underflow: right-neighbor pair count {} < merge count {}",
                        info.count,
                        count,
                    );
                    info.count -= count;
                }
            }

            state.tokens[left] = merged_id;
            state.tokens[right] = TOMBSTONE;
            state.next[left] = after;
            if let Some(a_idx) = after {
                state.prev[a_idx] = Some(left);
            }

            if let Some(b_idx) = before {
                let key = (state.tokens[b_idx], merged_id);
                let entry = pair_info.entry(key).or_default();
                entry.count += count;
                entry.locations.push((state_idx, b_idx));
            }
            if let Some(a_idx) = after {
                let key = (merged_id, state.tokens[a_idx]);
                let entry = pair_info.entry(key).or_default();
                entry.count += count;
                entry.locations.push((state_idx, left));
            }

            merged_any = true;
        }

        assert!(
            merged_any,
            "merge_pair called on pair with no live locations — count/location invariant violated",
        );

        merged_id
    }

    fn learn_merges(
        states: &mut [PreTokenState],
        pair_info: &mut HashMap<Pair, PairInfo>,
        num_merges: usize,
    ) -> Vec<Vec<u8>> {
        let mut vocab = Vocab {
            merged: Vec::with_capacity(num_merges),
        };
        for _ in 0..num_merges {
            let Some(pair) = Self::find_best_pair(pair_info, &vocab) else {
                break;
            };
            Self::merge_pair(pair, states, pair_info, &mut vocab);
            if vocab.merged.len() % 100 == 0 {
                eprintln!("trained {} / {} merges", vocab.merged.len(), num_merges);
            }
        }
        vocab.merged
    }

    fn write_vocab(output_path: &Path, merges: &[Vec<u8>]) -> io::Result<()> {
        let file = fs::File::create(output_path)?;
        let mut writer = io::BufWriter::new(file);

        let mut rank: usize = 0;
        for byte in 0u32..256 {
            rank += 1;
            let b64 = STANDARD.encode([byte as u8]);
            writeln!(writer, "{} {}", b64, rank)?;
        }

        for merge in merges {
            rank += 1;
            let b64 = STANDARD.encode(merge);
            writeln!(writer, "{} {}", b64, rank)?;
        }

        writer.flush()?;
        Ok(())
    }

    pub fn train(&self, output_path: impl AsRef<Path>, vocab_size: usize) -> io::Result<()> {
        if vocab_size < 256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("vocab_size must be at least 256, got {vocab_size}"),
            ));
        }
        let num_merges = vocab_size - 256;

        let mut states = self.prepare_pretoken_states()?;
        let mut pair_info = Self::init_pair_tables(&states);
        let merges = Self::learn_merges(&mut states, &mut pair_info, num_merges);

        eprintln!("trained {} merges (target: {})", merges.len(), num_merges);

        Self::write_vocab(output_path.as_ref(), &merges)?;
        Ok(())
    }

    fn read_corpus(&self) -> io::Result<CorpusIter> {
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
        assert_eq!(w.tokens, vec![b'h' as TokenId, b'i' as TokenId]);
        assert_eq!(w.next, vec![Some(1), None]);
        assert_eq!(w.prev, vec![None, Some(0)]);
        assert_eq!(w.count, 1);
    }

    #[test]
    fn init_pretoken_states_single_byte_piece() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("a", 1)]));
        assert_eq!(words.len(), 1);
        let w = &words[0];
        assert_eq!(w.tokens, vec![b'a' as TokenId]);
        assert_eq!(w.next, vec![None]);
        assert_eq!(w.prev, vec![None]);
        assert_eq!(w.count, 1);
    }

    #[test]
    fn init_pretoken_states_preserves_count() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("the", 3)]));
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].count, 3);
        assert_eq!(
            words[0].tokens,
            vec![b't' as TokenId, b'h' as TokenId, b'e' as TokenId]
        );
    }

    #[test]
    fn init_pretoken_states_distinct_pieces() {
        let words = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("a", 2), ("b", 1)]));
        assert_eq!(words.len(), 2);
        let counts: HashMap<Vec<u8>, u64> = words
            .iter()
            .map(|w| (w.tokens.iter().map(|&id| id as u8).collect(), w.count))
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
        assert_eq!(w.tokens, vec![0xC3 as TokenId, 0xA9 as TokenId]);
        assert_eq!(w.next, vec![Some(1), None]);
        assert_eq!(w.prev, vec![None, Some(0)]);
    }

    #[test]
    fn init_pair_tables_empty_input() {
        let info = BpeTokenizerTrainer::init_pair_tables(&[]);
        assert!(info.is_empty());
    }

    #[test]
    fn init_pair_tables_single_byte_state_has_no_pairs() {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("a", 4)]));
        let info = BpeTokenizerTrainer::init_pair_tables(&states);
        assert!(info.is_empty());
    }

    #[test]
    fn init_pair_tables_two_byte_state_records_one_pair() {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("ab", 5)]));
        let info = BpeTokenizerTrainer::init_pair_tables(&states);
        let pair = (b'a' as TokenId, b'b' as TokenId);
        let entry = info.get(&pair).expect("pair should be present");
        assert_eq!(entry.count, 5);
        assert_eq!(entry.locations, vec![(0, 0)]);
        assert_eq!(info.len(), 1);
    }

    #[test]
    fn init_pair_tables_three_byte_state_records_adjacent_pairs() {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("abc", 2)]));
        let info = BpeTokenizerTrainer::init_pair_tables(&states);
        let ab = (b'a' as TokenId, b'b' as TokenId);
        let bc = (b'b' as TokenId, b'c' as TokenId);
        let ab_entry = info.get(&ab).expect("ab pair should be present");
        let bc_entry = info.get(&bc).expect("bc pair should be present");
        assert_eq!(ab_entry.count, 2);
        assert_eq!(ab_entry.locations, vec![(0, 0)]);
        assert_eq!(bc_entry.count, 2);
        assert_eq!(bc_entry.locations, vec![(0, 1)]);
    }

    #[test]
    fn init_pair_tables_aggregates_shared_pair_across_states() {
        // Two distinct states sharing the (a,b) pair, with counts 3 and 4.
        let states =
            BpeTokenizerTrainer::init_pretoken_states(counts_from(&[("ab", 3), ("abx", 4)]));
        let info = BpeTokenizerTrainer::init_pair_tables(&states);
        let ab = (b'a' as TokenId, b'b' as TokenId);
        let entry = info.get(&ab).expect("ab pair should be present");
        assert_eq!(entry.count, 7);
        assert_eq!(entry.locations.len(), 2);
        let set: std::collections::HashSet<_> = entry.locations.iter().copied().collect();
        assert!(set.contains(&(0, 0)));
        assert!(set.contains(&(1, 0)));
    }

    fn pair(a: TokenId, b: TokenId) -> Pair {
        (a, b)
    }

    fn info(count: u64) -> PairInfo {
        PairInfo {
            count,
            locations: Vec::new(),
        }
    }

    fn empty_vocab() -> Vocab {
        Vocab { merged: Vec::new() }
    }

    #[test]
    fn find_best_pair_empty_map_returns_none() {
        let counts: HashMap<Pair, PairInfo> = HashMap::new();
        let vocab = empty_vocab();
        assert_eq!(BpeTokenizerTrainer::find_best_pair(&counts, &vocab), None);
    }

    #[test]
    fn find_best_pair_all_zero_counts_returns_none() {
        let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
        counts.insert(pair(b'a' as TokenId, b'b' as TokenId), info(0));
        counts.insert(pair(b'c' as TokenId, b'd' as TokenId), info(0));
        let vocab = empty_vocab();
        assert_eq!(BpeTokenizerTrainer::find_best_pair(&counts, &vocab), None);
    }

    #[test]
    fn find_best_pair_single_entry_returned() {
        let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
        counts.insert(pair(b'a' as TokenId, b'b' as TokenId), info(7));
        let vocab = empty_vocab();
        assert_eq!(
            BpeTokenizerTrainer::find_best_pair(&counts, &vocab),
            Some(pair(b'a' as TokenId, b'b' as TokenId))
        );
    }

    #[test]
    fn find_best_pair_skips_zero_count_entries() {
        let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
        counts.insert(pair(b'a' as TokenId, b'b' as TokenId), info(0));
        counts.insert(pair(b'c' as TokenId, b'd' as TokenId), info(3));
        let vocab = empty_vocab();
        assert_eq!(
            BpeTokenizerTrainer::find_best_pair(&counts, &vocab),
            Some(pair(b'c' as TokenId, b'd' as TokenId))
        );
    }

    #[test]
    fn find_best_pair_picks_highest_count() {
        let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
        counts.insert(pair(b'a' as TokenId, b'b' as TokenId), info(2));
        counts.insert(pair(b'c' as TokenId, b'd' as TokenId), info(9));
        counts.insert(pair(b'e' as TokenId, b'f' as TokenId), info(5));
        let vocab = empty_vocab();
        assert_eq!(
            BpeTokenizerTrainer::find_best_pair(&counts, &vocab),
            Some(pair(b'c' as TokenId, b'd' as TokenId))
        );
    }

    #[test]
    fn find_best_pair_tie_broken_lexicographically() {
        let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
        counts.insert(pair(b'a' as TokenId, b'b' as TokenId), info(5));
        counts.insert(pair(b'c' as TokenId, b'd' as TokenId), info(5));
        counts.insert(pair(b'a' as TokenId, b'a' as TokenId), info(5));
        // All tied at 5; lex-max pair wins.
        let vocab = empty_vocab();
        assert_eq!(
            BpeTokenizerTrainer::find_best_pair(&counts, &vocab),
            Some(pair(b'c' as TokenId, b'd' as TokenId))
        );
    }

    #[test]
    fn find_best_pair_tie_breaks_by_bytes_not_by_token_id() {
        // id 256 = b"a" — sorts before "b" lexically, but its numeric id (256)
        // is greater than b'b' (98). If the comparator used numeric id, it
        // would wrongly pick (256, 'b'). Lex-on-bytes picks ('b', 'b').
        let vocab = Vocab {
            merged: vec![b"a".to_vec()],
        };
        let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
        counts.insert((256, b'b' as TokenId), info(5));
        counts.insert((b'b' as TokenId, b'b' as TokenId), info(5));
        let best = BpeTokenizerTrainer::find_best_pair(&counts, &vocab);
        assert_eq!(best, Some((b'b' as TokenId, b'b' as TokenId)));
    }

    /// Walks `state`'s linked list from the head and collects the live byte
    /// sequences in order. Used to assert the post-merge shape of a state.
    fn live_token_bytes(state: &PreTokenState, vocab: &Vocab) -> Vec<Vec<u8>> {
        let head = (0..state.tokens.len())
            .find(|&i| state.prev[i].is_none())
            .expect("state must have a head");
        let mut result = Vec::new();
        let mut cur = Some(head);
        while let Some(i) = cur {
            result.push(vocab.bytes_of(state.tokens[i]).to_vec());
            cur = state.next[i];
        }
        result
    }

    fn setup(
        pretokens: &[(&str, u64)],
    ) -> (Vec<PreTokenState>, HashMap<Pair, PairInfo>, Vocab) {
        let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(pretokens));
        let pair_info = BpeTokenizerTrainer::init_pair_tables(&states);
        let vocab = Vocab { merged: Vec::new() };
        (states, pair_info, vocab)
    }

    #[test]
    fn merge_pair_two_byte_word_collapses_to_single_node() {
        let (mut states, mut info, mut vocab) = setup(&[("ab", 1)]);
        let merged_id = BpeTokenizerTrainer::merge_pair(
            pair(b'a' as TokenId, b'b' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
        assert_eq!(vocab.bytes_of(merged_id), b"ab");
        assert_eq!(live_token_bytes(&states[0], &vocab), vec![b"ab".to_vec()]);
        assert!(
            info.get(&pair(b'a' as TokenId, b'b' as TokenId))
                .is_none()
        );
    }

    #[test]
    fn merge_pair_three_byte_word_updates_right_neighbor_pair() {
        let (mut states, mut info, mut vocab) = setup(&[("abc", 1)]);
        let merged_id = BpeTokenizerTrainer::merge_pair(
            pair(b'a' as TokenId, b'b' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
        assert_eq!(vocab.bytes_of(merged_id), b"ab");
        assert_eq!(
            live_token_bytes(&states[0], &vocab),
            vec![b"ab".to_vec(), b"c".to_vec()]
        );
        assert_eq!(
            info.get(&pair(b'b' as TokenId, b'c' as TokenId))
                .map(|i| i.count),
            Some(0)
        );
        let abc_entry = info
            .get(&(merged_id, b'c' as TokenId))
            .expect("(ab,c) should exist");
        assert_eq!(abc_entry.count, 1);
        assert_eq!(abc_entry.locations, vec![(0, 0)]);
    }

    #[test]
    fn merge_pair_three_byte_word_updates_left_neighbor_pair() {
        let (mut states, mut info, mut vocab) = setup(&[("xab", 1)]);
        let merged_id = BpeTokenizerTrainer::merge_pair(
            pair(b'a' as TokenId, b'b' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
        assert_eq!(
            live_token_bytes(&states[0], &vocab),
            vec![b"x".to_vec(), b"ab".to_vec()]
        );
        assert_eq!(
            info.get(&pair(b'x' as TokenId, b'a' as TokenId))
                .map(|i| i.count),
            Some(0)
        );
        let entry = info
            .get(&(b'x' as TokenId, merged_id))
            .expect("(x,ab) should exist");
        assert_eq!(entry.count, 1);
        assert_eq!(entry.locations, vec![(0, 0)]);
    }

    #[test]
    fn merge_pair_handles_repeated_non_overlapping_pair() {
        let (mut states, mut info, mut vocab) = setup(&[("abab", 1)]);
        let merged_id = BpeTokenizerTrainer::merge_pair(
            pair(b'a' as TokenId, b'b' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
        assert_eq!(
            live_token_bytes(&states[0], &vocab),
            vec![b"ab".to_vec(), b"ab".to_vec()]
        );
        assert!(
            info.get(&pair(b'a' as TokenId, b'b' as TokenId))
                .is_none()
        );
        let entry = info
            .get(&(merged_id, merged_id))
            .expect("(ab,ab) should exist");
        assert_eq!(entry.count, 1);
    }

    #[test]
    fn merge_pair_skips_stale_overlapping_locations() {
        let (mut states, mut info, mut vocab) = setup(&[("aaa", 1)]);
        let merged_id = BpeTokenizerTrainer::merge_pair(
            pair(b'a' as TokenId, b'a' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
        assert_eq!(
            live_token_bytes(&states[0], &vocab),
            vec![b"aa".to_vec(), b"a".to_vec()]
        );
        assert!(
            info.get(&pair(b'a' as TokenId, b'a' as TokenId))
                .is_none()
        );
        let entry = info
            .get(&(merged_id, b'a' as TokenId))
            .expect("(aa,a) should exist");
        assert_eq!(entry.count, 1);
    }

    #[test]
    fn merge_pair_aggregates_across_multiple_states() {
        let (mut states, mut info, mut vocab) = setup(&[("ab", 3), ("abx", 4)]);
        let merged_id = BpeTokenizerTrainer::merge_pair(
            pair(b'a' as TokenId, b'b' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
        let (ab_state, abx_state): (&PreTokenState, &PreTokenState) = {
            let mut iter = states.iter();
            let s0 = iter.next().unwrap();
            let s1 = iter.next().unwrap();
            if s0.count == 3 { (s0, s1) } else { (s1, s0) }
        };
        assert_eq!(live_token_bytes(ab_state, &vocab), vec![b"ab".to_vec()]);
        assert_eq!(
            live_token_bytes(abx_state, &vocab),
            vec![b"ab".to_vec(), b"x".to_vec()]
        );
        assert!(
            info.get(&pair(b'a' as TokenId, b'b' as TokenId))
                .is_none()
        );
        let new_entry = info
            .get(&(merged_id, b'x' as TokenId))
            .expect("(ab,x) should exist");
        assert_eq!(new_entry.count, 4);
    }

    #[test]
    #[should_panic(expected = "merge_pair called on pair with no live locations")]
    fn merge_pair_panics_when_pair_has_no_live_locations() {
        let (mut states, mut info, mut vocab) = setup(&[("ab", 1)]);
        // (x, y) is not in pair_info at all → locations Vec is empty → no live merges.
        BpeTokenizerTrainer::merge_pair(
            pair(b'x' as TokenId, b'y' as TokenId),
            &mut states,
            &mut info,
            &mut vocab,
        );
    }

    #[test]
    fn learn_merges_zero_iterations_returns_empty() {
        let (mut states, mut info, _vocab) = setup(&[("ab", 1)]);
        let merges = BpeTokenizerTrainer::learn_merges(&mut states, &mut info, 0);
        assert!(merges.is_empty());
    }

    #[test]
    fn learn_merges_no_pairs_returns_empty() {
        let (mut states, mut info, _vocab) = setup(&[("a", 4)]);
        let merges = BpeTokenizerTrainer::learn_merges(&mut states, &mut info, 10);
        assert!(merges.is_empty());
    }

    #[test]
    fn learn_merges_repeated_pair_then_compounds() {
        let (mut states, mut info, _vocab) = setup(&[("abab", 1)]);
        let merges = BpeTokenizerTrainer::learn_merges(&mut states, &mut info, 2);
        assert_eq!(merges, vec![b"ab".to_vec(), b"abab".to_vec()]);
    }

    #[test]
    fn learn_merges_picks_higher_count_pair_first() {
        let (mut states, mut info, _vocab) = setup(&[("ab", 5), ("cd", 3)]);
        let merges = BpeTokenizerTrainer::learn_merges(&mut states, &mut info, 1);
        assert_eq!(merges, vec![b"ab".to_vec()]);
    }

    #[test]
    fn learn_merges_stops_early_when_corpus_exhausted() {
        let (mut states, mut info, _vocab) = setup(&[("ab", 1)]);
        let merges = BpeTokenizerTrainer::learn_merges(&mut states, &mut info, 5);
        assert_eq!(merges, vec![b"ab".to_vec()]);
    }

    #[test]
    fn learn_merges_deterministic_tie_breaking() {
        // (a,b) and (c,d) both have count 5; lex-max (c,d) wins first,
        // then (a,b) is the only remaining pair.
        let (mut states, mut info, _vocab) = setup(&[("ab", 5), ("cd", 5)]);
        let merges = BpeTokenizerTrainer::learn_merges(&mut states, &mut info, 2);
        assert_eq!(merges, vec![b"cd".to_vec(), b"ab".to_vec()]);
    }

    #[test]
    fn write_vocab_no_merges_writes_256_byte_lines() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        BpeTokenizerTrainer::write_vocab(temp.path(), &[]).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        assert_eq!(contents.lines().count(), 256);
    }

    #[test]
    fn write_vocab_byte_tokens_have_sequential_ranks_starting_at_one() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        BpeTokenizerTrainer::write_vocab(temp.path(), &[]).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        for (i, line) in contents.lines().enumerate() {
            let (_, rank_str) = line
                .split_once(' ')
                .expect("each line must have base64 and rank separated by space");
            let rank: usize = rank_str.parse().unwrap();
            assert_eq!(rank, i + 1);
        }
    }

    #[test]
    fn write_vocab_byte_tokens_decode_to_single_bytes() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        BpeTokenizerTrainer::write_vocab(temp.path(), &[]).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        for (i, line) in contents.lines().enumerate() {
            let (b64, _) = line.split_once(' ').unwrap();
            let bytes = STANDARD.decode(b64).unwrap();
            assert_eq!(bytes, vec![i as u8]);
        }
    }

    #[test]
    fn write_vocab_appends_merges_after_byte_tokens() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let merges = vec![b"ab".to_vec(), b"the".to_vec()];
        BpeTokenizerTrainer::write_vocab(temp.path(), &merges).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 258);

        let (b64_first_merge, rank_first_merge) = lines[256].split_once(' ').unwrap();
        assert_eq!(rank_first_merge, "257");
        assert_eq!(STANDARD.decode(b64_first_merge).unwrap(), b"ab".to_vec());

        let (b64_second_merge, rank_second_merge) = lines[257].split_once(' ').unwrap();
        assert_eq!(rank_second_merge, "258");
        assert_eq!(STANDARD.decode(b64_second_merge).unwrap(), b"the".to_vec());
    }

    #[test]
    fn write_vocab_overwrites_existing_file() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        BpeTokenizerTrainer::write_vocab(temp.path(), &[b"ab".to_vec()]).unwrap();
        BpeTokenizerTrainer::write_vocab(temp.path(), &[b"xy".to_vec(), b"zz".to_vec()]).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 258);
        let (b64_256, _) = lines[256].split_once(' ').unwrap();
        assert_eq!(STANDARD.decode(b64_256).unwrap(), b"xy".to_vec());
        let (b64_257, _) = lines[257].split_once(' ').unwrap();
        assert_eq!(STANDARD.decode(b64_257).unwrap(), b"zz".to_vec());
    }

    #[test]
    fn write_vocab_round_trips_via_reference_parser() {
        // Mirrors the rs-text-chunker tiktoken parser: split_once(' '),
        // parse rank as integer, base64-decode bytes.
        let temp = tempfile::NamedTempFile::new().unwrap();
        let merges = vec![b"hello".to_vec(), b" world".to_vec()];
        BpeTokenizerTrainer::write_vocab(temp.path(), &merges).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();

        let parsed: Vec<(u32, Vec<u8>)> = contents
            .lines()
            .map(|line| {
                let (b64, rank_str) = line.split_once(' ').unwrap();
                let rank: u32 = rank_str.parse().unwrap();
                let bytes = STANDARD.decode(b64).unwrap();
                (rank, bytes)
            })
            .collect();

        assert_eq!(parsed.len(), 258);
        for i in 0..256 {
            assert_eq!(parsed[i], ((i + 1) as u32, vec![i as u8]));
        }
        assert_eq!(parsed[256], (257, b"hello".to_vec()));
        assert_eq!(parsed[257], (258, b" world".to_vec()));
    }

    #[test]
    fn train_rejects_vocab_size_below_256() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let trainer = BpeTokenizerTrainer::new("data", 10000);
        let err = trainer
            .train(temp.path(), 100)
            .err()
            .expect("should error for vocab_size < 256");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn train_vocab_size_256_writes_only_byte_tokens() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let trainer = BpeTokenizerTrainer::new("data", 10000);
        trainer.train(temp.path(), 256).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        assert_eq!(contents.lines().count(), 256);
    }

    #[test]
    fn train_writes_target_vocab_size() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let trainer = BpeTokenizerTrainer::new("data", 10000);
        trainer.train(temp.path(), 300).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 300);
        for (i, line) in lines.iter().enumerate() {
            let (_, rank_str) = line.split_once(' ').unwrap();
            let rank: usize = rank_str.parse().unwrap();
            assert_eq!(rank, i + 1);
        }
    }

    #[test]
    fn train_output_decodes_via_reference_parser() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let trainer = BpeTokenizerTrainer::new("data", 10000);
        trainer.train(temp.path(), 300).unwrap();
        let contents = fs::read_to_string(temp.path()).unwrap();

        let parsed: Vec<(u32, Vec<u8>)> = contents
            .lines()
            .map(|line| {
                let (b64, rank_str) = line.split_once(' ').unwrap();
                let rank: u32 = rank_str.parse().unwrap();
                let bytes = STANDARD.decode(b64).unwrap();
                (rank, bytes)
            })
            .collect();

        assert_eq!(parsed.len(), 300);
        for i in 0..256 {
            assert_eq!(parsed[i], ((i + 1) as u32, vec![i as u8]));
        }
        for i in 256..300 {
            assert_eq!(parsed[i].0, (i + 1) as u32);
            assert!(!parsed[i].1.is_empty(), "merge bytes should not be empty");
        }
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
            assert!(!s.tokens.is_empty());
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
