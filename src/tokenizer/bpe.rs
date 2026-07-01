use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::io;
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use fancy_regex::Regex;
use rayon::prelude::*;

use super::shared::{self, NUM_SPECIAL_TOKENS, SPECIAL_TOKENS, TokenId, Vocab, build_pattern};

const PARALLEL_THRESHOLD: usize = 100;

/// Heap entry for the BPE merge priority queue.
/// (rank, left_idx, left_generation, right_idx, right_generation).
type HeapEntry = (u32, usize, u32, usize, u32);

pub struct BpeTokenizer {
    // Field order matters: `encoder` must drop before `arena` because its
    // keys are slices borrowed (via the unsafe transmute below) from the arena.
    encoder: HashMap<&'static [u8], TokenId>,
    vocab: Vocab,
    /// Id of the lowest special token (= `256 + merged.len()` = BOS). The 9
    /// specials occupy `first_special_id ..= first_special_id + 8` just above the learned merges.
    first_special_id: TokenId,
    pattern: Regex,
    byte_pair_ranks: Box<[[u32; 256]; 256]>,
    // Read indirectly: `encoder`'s keys are `&'static [u8]` slices into this
    // arena. Keeping it alive is what makes those slices valid.
    #[allow(dead_code)]
    arena: Vec<u8>,
}

impl BpeTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;

        // Pass 1: parse every line into (rank, bytes).
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (line_no, line) in contents.lines().enumerate() {
            let (b64, rank_str) = line.split_once(' ').ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("vocab line {} missing rank: {:?}", line_no + 1, line),
                )
            })?;
            let bytes = STANDARD.decode(b64).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("vocab line {} bad base64: {}", line_no + 1, e),
                )
            })?;
            let rank: u32 = rank_str.parse().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("vocab line {} bad rank: {}", line_no + 1, e),
                )
            })?;
            entries.push((rank, bytes));
        }

        entries.sort_by_key(|(rank, _)| *rank);

        // Ranks must be a contiguous 0..=N-1 sequence.
        for (i, (rank, _)) in entries.iter().enumerate() {
            let expected = i as u32;
            if *rank != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("expected rank {} at position {}, got {}", expected, i, rank),
                ));
            }
        }

        // The first 256 entries must be the single-byte tokens 0x00..=0xFF
        // in order, so id 0..=255 always decode to their corresponding byte.
        if entries.len() < 256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "vocab must contain at least 256 single-byte entries, got {}",
                    entries.len()
                ),
            ));
        }
        for (i, (_, bytes)) in entries.iter().take(256).enumerate() {
            if bytes.as_slice() != [i as u8] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "rank {} must encode single byte {:#x}, got {:?}",
                        i + 1,
                        i,
                        bytes
                    ),
                ));
            }
        }

        // Pass 2: build the encoder, vocab, and byte-pair lookup table.
        // The arena is sized exactly so it never reallocates — that invariant
        // is what keeps the unsafe transmute below sound.
        let total_bytes: usize = entries.iter().map(|(_, b)| b.len()).sum();
        let mut arena: Vec<u8> = Vec::with_capacity(total_bytes);
        let mut encoder: HashMap<&'static [u8], TokenId> = HashMap::with_capacity(entries.len());
        let mut merged: Vec<Vec<u8>> = Vec::with_capacity(entries.len().saturating_sub(256));
        let mut byte_pair_ranks: Box<[[u32; 256]; 256]> = Box::new([[u32::MAX; 256]; 256]);

        for (rank, bytes) in entries {
            let start = arena.len();
            arena.extend_from_slice(&bytes);
            // SAFETY: arena.capacity() == total_bytes exactly, and we never
            // push beyond that, so this slice address is stable for the life
            // of `arena`. The arena field is declared after `encoder` so it
            // outlives encoder at drop time.
            let slice: &'static [u8] = unsafe { std::mem::transmute(&arena[start..]) };
            encoder.insert(slice, rank as TokenId);

            if bytes.len() == 2 {
                byte_pair_ranks[bytes[0] as usize][bytes[1] as usize] = rank;
            }

            if rank >= 256 {
                merged.push(bytes);
            }
        }

        // Specials are appended in memory as a contiguous block above the
        // learned merges; they are not stored in the file.
        let first_special_id = 256 + merged.len() as TokenId;

        Ok(Self {
            encoder,
            vocab: Vocab { merged },
            first_special_id,
            pattern: build_pattern(),
            byte_pair_ranks,
            arena,
        })
    }

    pub fn bos_id(&self) -> TokenId {
        self.first_special_id
    }

    /// Id of a named special token (e.g. `"<|bos|>"`), or `None` if `name` is
    /// not one of the reserved specials.
    pub fn special_id(&self, name: &str) -> Option<TokenId> {
        SPECIAL_TOKENS
            .iter()
            .position(|s| *s == name)
            .map(|i| self.first_special_id + i as TokenId)
    }

    /// Total vocabulary size implied by this tokenizer: the 256 single-byte
    /// tokens + the learned merges + the special-token block (= highest token
    /// id + 1). Matches the `vocab_size` the model embedding must be sized to.
    pub fn vocab_size(&self) -> usize {
        256 + self.vocab.merged.len() + NUM_SPECIAL_TOKENS
    }

    pub fn token_byte_lengths(&self) -> Vec<u32> {
        (0..self.vocab_size() as TokenId)
            .map(|id| {
                if id < self.first_special_id {
                    self.bytes_of(id).len() as u32
                } else {
                    0
                }
            })
            .collect()
    }

    /// Bytes for a token id, dispatching across bytes/merges (delegated to
    /// `Vocab`) and the special-token block. A special id renders as its literal
    /// string (e.g. `<|bos|>`), matching tiktoken's decode behavior.
    fn bytes_of(&self, id: TokenId) -> &[u8] {
        if id < self.first_special_id {
            self.vocab.bytes_of(id)
        } else {
            let idx = (id - self.first_special_id) as usize;
            SPECIAL_TOKENS
                .get(idx)
                .unwrap_or_else(|| panic!("token id {id} is out of range (vocab_size {})", self.vocab_size()))
                .as_bytes()
        }
    }

    fn decode_bytes(&self, ids: &[TokenId]) -> Vec<u8> {
        let total: usize = ids.iter().map(|&id| self.bytes_of(id).len()).sum();
        let mut bytes = Vec::with_capacity(total);
        for &id in ids {
            bytes.extend_from_slice(self.bytes_of(id));
        }
        bytes
    }

    pub fn decode(&self, ids: &[TokenId]) -> String {
        String::from_utf8(self.decode_bytes(ids))
            .expect("decoded token bytes should be valid UTF-8")
    }

    /// Like [`decode`](Self::decode) but replaces invalid UTF-8 with U+FFFD instead of panicking.
    pub fn decode_lossy(&self, ids: &[TokenId]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    pub fn encode(&self, text: &str) -> Vec<TokenId> {
        let pieces = self.pre_tokenize(text);
        if pieces.len() >= PARALLEL_THRESHOLD {
            pieces
                .par_iter()
                .flat_map(|piece| self.bpe_encode_piece(piece.as_bytes()))
                .collect()
        } else {
            pieces
                .iter()
                .flat_map(|piece| self.bpe_encode_piece(piece.as_bytes()))
                .collect()
        }
    }

    fn pre_tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        shared::pre_tokenize(&self.pattern, text)
    }

    /// BPE-encode a single pre-token's bytes by repeatedly merging the
    /// adjacent pair with the lowest token id (= earliest learned merge).
    /// Uses a doubly-linked list with generation counters so stale heap
    /// entries can be detected and skipped without rebuilding the heap.
    fn bpe_encode_piece(&self, piece: &[u8]) -> Vec<TokenId> {
        let n = piece.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![piece[0] as TokenId];
        }

        // Linked-list state, one entry per starting byte.
        // bytes[i] grows as merges absorb neighbors into i.
        let mut bytes: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
        let mut prev: Vec<Option<usize>> = (0..n)
            .map(|i| if i > 0 { Some(i - 1) } else { None })
            .collect();
        let mut next: Vec<Option<usize>> = (0..n)
            .map(|i| if i + 1 < n { Some(i + 1) } else { None })
            .collect();
        let mut generation: Vec<u32> = vec![0; n];

        // Min-heap on HeapEntry. Lower rank = earlier-learned merge = higher priority.
        let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();

        // Seed with all initial adjacent byte pairs via the 2D table —
        // direct indexing is faster than HashMap lookup on the first pass.
        for i in 0..n - 1 {
            let rank = self.byte_pair_ranks[piece[i] as usize][piece[i + 1] as usize];
            if rank != u32::MAX {
                heap.push(Reverse((rank, i, 0, i + 1, 0)));
            }
        }

        let mut pair_buf: Vec<u8> = Vec::with_capacity(16);

        while let Some(Reverse((_, left, lg, right, rg))) = heap.pop() {
            // Skip stale entries: the linked-list shape changed since this
            // pair was queued, or one side has been merged away.
            if next[left] != Some(right) || generation[left] != lg || generation[right] != rg {
                continue;
            }

            // Merge "right" into "left".
            let right_bytes = std::mem::take(&mut bytes[right]);
            bytes[left].extend_from_slice(&right_bytes);
            generation[left] += 1;

            let after = next[right];
            next[left] = after;
            next[right] = None;
            if let Some(a) = after {
                prev[a] = Some(left);
            }

            // New pair: (node before left, left).
            if let Some(before) = prev[left] {
                pair_buf.clear();
                pair_buf.extend_from_slice(&bytes[before]);
                pair_buf.extend_from_slice(&bytes[left]);
                if let Some(&rank) = self.encoder.get(pair_buf.as_slice()) {
                    heap.push(Reverse((
                        rank,
                        before,
                        generation[before],
                        left,
                        generation[left],
                    )));
                }
            }

            // New pair: (left, node after right).
            if let Some(a) = after {
                pair_buf.clear();
                pair_buf.extend_from_slice(&bytes[left]);
                pair_buf.extend_from_slice(&bytes[a]);
                if let Some(&rank) = self.encoder.get(pair_buf.as_slice()) {
                    heap.push(Reverse((rank, left, generation[left], a, generation[a])));
                }
            }
        }

        // Walk the surviving linked list from the head and emit token ids.
        let mut result = Vec::new();
        let mut pos = Some(0usize);
        while let Some(i) = pos {
            let id = self
                .encoder
                .get(bytes[i].as_slice())
                .copied()
                .expect("every linked-list span must resolve to a known token");
            result.push(id);
            pos = next[i];
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::BpeTokenizerTrainer;

    fn write_vocab_file(lines: &[&str]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut temp = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(temp, "{}", line).unwrap();
        }
        temp.flush().unwrap();
        temp
    }

    fn train_tiny_vocab(vocab_size: usize) -> tempfile::NamedTempFile {
        let temp = tempfile::NamedTempFile::new().unwrap();
        let trainer = BpeTokenizerTrainer::new("data", 10_000, usize::MAX);
        trainer.train(temp.path(), vocab_size).unwrap();
        temp
    }

    #[test]
    fn from_file_loads_trainer_output() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        // 300 total vocab = 256 bytes + 35 merges + 9 reserved specials.
        assert_eq!(tok.encoder.len(), 291); // bytes + merges (file entries)
        assert_eq!(tok.vocab.merged.len(), 35);
        assert_eq!(tok.bos_id(), 291); // first special = 256 + 35 = V - 9
        assert_eq!(tok.vocab_size(), 300);
    }

    #[test]
    fn token_byte_lengths_zero_for_specials_and_byte_len_otherwise() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let lens = tok.token_byte_lengths();

        // One entry per token id.
        assert_eq!(lens.len(), tok.vocab_size());
        // The 256 single-byte tokens are exactly one byte each.
        for (id, len) in lens.iter().enumerate().take(256) {
            assert_eq!(*len, 1, "byte token {id} should be 1 byte");
        }
        // Each learned merge matches the byte length of its stored bytes.
        for (i, bytes) in tok.vocab.merged.iter().enumerate() {
            assert_eq!(lens[256 + i], bytes.len() as u32);
        }
        // The special-token block at the top of the vocab contributes 0.
        let first_special = tok.bos_id() as usize;
        for (offset, len) in lens[first_special..].iter().enumerate() {
            assert_eq!(*len, 0, "special token {} should be 0 bytes", first_special + offset);
        }
    }

    #[test]
    fn from_file_byte_only_vocab() {
        let vocab_file = train_tiny_vocab(265); // 256 bytes + 0 merges + 9 specials
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_eq!(tok.encoder.len(), 256);
        assert!(tok.vocab.merged.is_empty());
        assert_eq!(tok.bos_id(), 256); // first special sits right above the bytes
    }

    #[test]
    fn special_ids_form_contiguous_top_block() {
        let vocab_file = train_tiny_vocab(265); // byte-only: first special = 256
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_eq!(tok.bos_id(), 256);
        assert_eq!(tok.special_id("<|bos|>"), Some(256)); // first special
        assert_eq!(tok.special_id("<|assistant_start|>"), Some(259)); // index 3
        assert_eq!(tok.special_id("<|output_end|>"), Some(264)); // last special
        assert_eq!(tok.special_id("not_a_special"), None);
        assert_eq!(tok.vocab_size(), 265); // 256 + 9
    }

    #[test]
    fn decode_renders_special_tokens() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let bos = tok.bos_id();
        assert_eq!(tok.decode(&[bos]), "<|bos|>");
        let (a, b) = (b'a' as TokenId, b'b' as TokenId);
        assert_eq!(tok.decode(&[bos, a, b]), "<|bos|>ab");
    }

    #[test]
    fn encode_does_not_emit_special_ids() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let ids = tok.encode("<|bos|>");
        assert!(!ids.contains(&tok.bos_id()));
        assert_eq!(tok.decode(&ids), "<|bos|>");
    }

    #[test]
    fn from_file_byte_pair_ranks_populated_for_two_byte_merges() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let two_byte_merges = tok.vocab.merged.iter().filter(|m| m.len() == 2).count();
        let populated_cells = tok
            .byte_pair_ranks
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&r| r != u32::MAX)
            .count();
        assert_eq!(populated_cells, two_byte_merges);
    }

    #[test]
    fn from_file_rejects_missing_space() {
        let temp = write_vocab_file(&["abcd1"]);
        let err = BpeTokenizer::from_file(temp.path())
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn from_file_rejects_bad_base64() {
        let temp = write_vocab_file(&["!!!notb64 1"]);
        let err = BpeTokenizer::from_file(temp.path())
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn from_file_rejects_non_numeric_rank() {
        let temp = write_vocab_file(&["AA== abc"]);
        let err = BpeTokenizer::from_file(temp.path())
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn from_file_rejects_rank_gap() {
        // Two valid lines but ranks 0 and 2 — gap at 1.
        let temp = write_vocab_file(&["AA== 0", "AQ== 2"]);
        let err = BpeTokenizer::from_file(temp.path())
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn from_file_rejects_too_few_entries() {
        // Only one entry — not the required 256 single bytes.
        let temp = write_vocab_file(&["AA== 0"]);
        let err = BpeTokenizer::from_file(temp.path())
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn from_file_rejects_wrong_single_byte() {
        // 256 lines but the byte at rank 0 is "ab" instead of 0x00.
        let mut lines: Vec<String> = Vec::with_capacity(256);
        lines.push(format!("{} 0", STANDARD.encode(b"ab")));
        for i in 1..256u32 {
            lines.push(format!("{} {}", STANDARD.encode([i as u8]), i));
        }
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let temp = write_vocab_file(&line_refs);
        let err = BpeTokenizer::from_file(temp.path())
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn from_file_missing_file() {
        let err = BpeTokenizer::from_file("/nonexistent/path/vocab.txt")
            .err()
            .expect("expected an error");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn decode_empty() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn decode_single_byte_token() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        // id 0x61 = 97 → byte 'a'
        assert_eq!(tok.decode(&[b'a' as TokenId]), "a");
    }

    #[test]
    fn decode_byte_sequence_is_concatenation() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let ids: Vec<TokenId> = b"hello".iter().map(|&b| b as TokenId).collect();
        assert_eq!(tok.decode(&ids), "hello");
    }

    #[test]
    fn decode_lossy_replaces_invalid_utf8_without_panicking() {
        let vocab_file = train_tiny_vocab(265); // byte-only vocab
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        // A lone UTF-8 continuation byte (0x80) is invalid on its own; `decode`
        // would panic, but `decode_lossy` yields the replacement character.
        assert_eq!(tok.decode_lossy(&[0x80]), "\u{FFFD}");
    }

    #[test]
    fn decode_merge_token_returns_merged_bytes() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        // Merge id 256 must equal the bytes of vocab.merged[0].
        let expected = String::from_utf8(tok.vocab.merged[0].clone()).unwrap();
        assert_eq!(tok.decode(&[256]), expected);
    }

    #[test]
    fn encode_empty() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_eq!(tok.encode(""), Vec::<TokenId>::new());
    }

    #[test]
    fn encode_single_char_byte_only_vocab() {
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_eq!(tok.encode("a"), vec![b'a' as TokenId]);
    }

    #[test]
    fn encode_multi_char_byte_only_vocab_emits_bytes() {
        // With no merges, encode just maps each byte to its 1-based id.
        let vocab_file = train_tiny_vocab(265);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let expected: Vec<TokenId> = b"hi".iter().map(|&b| b as TokenId).collect();
        assert_eq!(tok.encode("hi"), expected);
    }

    #[test]
    fn encode_uses_two_byte_merge() {
        // Force a vocab where the only merge is the (b'a', b'b') pair, so
        // encoding "ab" must produce that merge's id rather than two bytes.
        let mut lines: Vec<String> = (0..256u32)
            .map(|i| format!("{} {}", STANDARD.encode([i as u8]), i))
            .collect();
        lines.push(format!("{} 256", STANDARD.encode(b"ab")));
        let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let temp = write_vocab_file(&line_refs);
        let tok = BpeTokenizer::from_file(temp.path()).unwrap();
        assert_eq!(tok.encode("ab"), vec![256]);
    }

    fn assert_round_trip(tok: &BpeTokenizer, text: &str) {
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text, "round-trip failed for {:?}", text);
    }

    #[test]
    fn round_trip_ascii() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_round_trip(&tok, "Hello, world!");
        assert_round_trip(&tok, "The quick brown fox jumps over the lazy dog.");
    }

    #[test]
    fn round_trip_contractions() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_round_trip(&tok, "I'm don't you're it's");
    }

    #[test]
    fn round_trip_whitespace_and_newlines() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_round_trip(&tok, "  \t\n\r\n  ");
    }

    #[test]
    fn round_trip_cjk() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_round_trip(&tok, "\u{4F60}\u{597D}\u{4E16}\u{754C}");
    }

    #[test]
    fn round_trip_emoji() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_round_trip(&tok, "\u{1F389}\u{1F38A}\u{1F388}");
    }

    #[test]
    fn round_trip_mixed() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        assert_round_trip(&tok, "Hello \u{4E16}\u{754C}! \u{1F389} 123");
    }

    #[test]
    fn round_trip_triggers_parallel_path() {
        let vocab_file = train_tiny_vocab(300);
        let tok = BpeTokenizer::from_file(vocab_file.path()).unwrap();
        let text = "word ".repeat(PARALLEL_THRESHOLD + 1);
        assert!(tok.pre_tokenize(&text).len() >= PARALLEL_THRESHOLD);
        assert_round_trip(&tok, &text);
    }
}
