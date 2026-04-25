# BPE Tokenizer Trainer Implementation Plan

## Context

We need a BPE tokenizer trainer in `src/tokenizer/mod.rs` that learns a vocabulary from a text corpus. The trainer reads raw text, pre-tokenizes it with GPT-4-style regex, then iteratively merges the most frequent byte-pair until the desired vocab size is reached. Output is written in tiktoken format for compatibility with existing tokenizer infrastructure.

We borrow two things from `/Users/teppi/rs-text-chunker/src/tokenizer.rs`:
- **`REGEX_PATTERNS`** (lines 14-22) — copied directly for pre-tokenization
- **Linked-list merge structure** from `bpe_merge_piece` (lines 156-242) — adapted from encoding (merge by rank) to training (merge by global frequency)

No new dependencies needed — `fancy-regex` and `base64` are already in `Cargo.toml`.

---

## Data Structures

```rust
type Pair = (Vec<u8>, Vec<u8>);

struct WordState {
    bytes: Vec<Vec<u8>>,       // token bytes at each node
    next: Vec<Option<usize>>,  // linked-list next pointers
    prev: Vec<Option<usize>>,  // linked-list prev pointers
    count: u64,                // corpus frequency of this word
}
```

Global state during training (local variables inside `train`):
- `pair_counts: HashMap<Pair, u64>` — total frequency of each adjacent pair
- `pair_locations: HashMap<Pair, Vec<(usize, usize)>>` — maps pair to `(word_index, left_node_index)` for efficient merging
- `vocab: Vec<Vec<u8>>` — merged tokens in order (ranks 256..)

Key design: **deduplicate words** — store each unique pre-tokenized word once with a `count` multiplier rather than duplicating linked lists.

---

## Implementation Steps

### Step 1: Constants and imports

Add to `mod.rs`:
- `use std::collections::HashMap`, `std::fs`, `std::io::Write`, `std::path::Path`, `base64::Engine`, `fancy_regex::Regex`
- Copy `REGEX_PATTERNS` from reference file

### Step 2: `WordState` struct and `Pair` type alias

Private to the module.

### Step 3: `BpeTokenizerTrainer` methods

**`pub fn train(corpus_path, output_path, vocab_size, max_chars)`** — the main entry point:
1. Read corpus, truncate to `max_chars` (respecting UTF-8 char boundaries)
2. Pre-tokenize with regex
3. Deduplicate words into `Vec<WordState>` with frequency counts
4. Build initial `pair_counts` and `pair_locations`
5. Loop `vocab_size - 256` times: find best pair, merge it, collect merged token
6. Write vocab to output file
7. Print progress to stderr every 100 merges

**`fn pre_tokenize(pattern, text) -> Vec<&str>`** — borrowed from reference lines 140-152. Walk text with `pattern.find_from_pos`, collect slices.

**`fn init_words(pieces) -> Vec<WordState>`** — count piece frequencies with a HashMap, then create one `WordState` per unique piece (each byte becomes a single-byte node in the linked list).

**`fn init_pair_tables(words) -> (pair_counts, pair_locations)`** — walk each word's linked list, count every adjacent pair weighted by `word.count`.

**`fn find_best_pair(pair_counts) -> Option<Pair>`** — linear scan for highest count, filter out zero-count entries.

**`fn merge_pair(pair, words, pair_counts, pair_locations) -> Vec<u8>`** — the critical method:
1. Take `pair_locations[pair]` (remove from map)
2. For each `(word_idx, left)` location:
   - Validate still live: check `next[left] == Some(right)` and `bytes[left]/bytes[right]` match pair (handles staleness without generation numbers)
   - Decrement old adjacent pairs: `(before_left, left)` and `(right, after_right)`
   - Merge: `bytes[left] = concat(pair.0, pair.1)`, unlink `right`
   - Increment new adjacent pairs: `(before_left, merged)` and `(merged, after_right)`, add to `pair_locations`
3. Return merged bytes

**`fn write_vocab(output_path, vocab)`** — write tiktoken format:
- 256 single-byte tokens at ranks 0..255
- Merged tokens at ranks 256..

### Step 4: Update `main.rs`

Add clap argument parsing to invoke the trainer:
```
--corpus <path>     Path to corpus file
--output <path>     Path to output vocabulary file
--vocab-size <n>    Target vocabulary size (default: 512)
--max-chars <n>     Max characters to inspect (optional)
```

---

## Key Design Decisions

1. **Staleness via content check** (not generation numbers) — simpler than the reference's approach and correct because merged content never reverts
2. **`Pair` as `(Vec<u8>, Vec<u8>)`** — allocates but is correct and clear; arena-based interning is a future optimization
3. **Linear scan for best pair** — O(pairs) per merge, adequate for vocab sizes up to ~50K
4. **`saturating_sub` for count decrements** — guards against underflow from stale double-decrements
5. **`max_chars` truncation** — use `floor_char_boundary` or `char_indices` to respect UTF-8 boundaries

---

## Files to Modify

- `/Users/teppi/rs-nanogpt/src/tokenizer/mod.rs` — all trainer code
- `/Users/teppi/rs-nanogpt/src/main.rs` — clap CLI to invoke trainer

## Verification

1. `cargo build` — compiles without errors
2. `cargo test` — unit tests for pre-tokenize, word init, small corpus training
3. Manual test: run trainer on a small text file, verify output is valid tiktoken format (base64 + rank per line, ranks sequential, base64 decodes correctly)
