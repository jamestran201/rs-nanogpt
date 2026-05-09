# Tokenizer evaluation command — plan

## Goal

Add an `eval-tokenizer` subcommand that loads a trained vocab from disk, runs encode/decode round-trip on a few text fixtures, and reports `bytes / tokens` compression ratios. Models nanochat's `scripts/tok_eval.py` but scoped down for our current state (small vocab, no baselines yet).

## What we have vs. what's missing

| Capability | Status |
|---|---|
| Train BPE → write tiktoken vocab | done (`BpeTokenizerTrainer`) |
| Load tiktoken vocab from disk | missing |
| Encode text → token ids | missing |
| Decode token ids → bytes | missing |
| GPT-2 / GPT-4 baseline comparison | missing (deferred) |

The biggest gap is **encoding**. Training finds best pairs *by frequency*; encoding applies merges *by rank order* — same data structure, different algorithm.

## Phases

### Phase 1 — `BpeTokenizer` (loadable, with encode/decode)

A new struct, separate from `BpeTokenizerTrainer`, that owns:
- the 256 single-byte tokens + learned merge bytes (a `Vocab`-equivalent),
- a `HashMap<(TokenId, TokenId), TokenId>` for merge ranks (rank = order learned).

API:
- `BpeTokenizer::from_file(path) -> io::Result<Self>` — parse tiktoken format (base64 + rank, 1-indexed; ranks 1–256 are single bytes, 257+ are merges in learned order).
- `encode(&self, text: &str) -> Vec<TokenId>` — same pre-tokenization regex as training, then for each pre-token: start with the byte sequence, repeatedly find the lowest-rank adjacent pair and merge, until no merges apply.
- `decode(&self, ids: &[TokenId]) -> Vec<u8>` — concatenate `vocab.bytes_of(id)` for each id.

Reuse from existing tokenizer module: pre-tokenization regex, `Vocab::bytes_of`, `TokenId` alias, `SINGLE_BYTE_TABLE`. Extract the regex to a shared `pub(crate) const` first so train and encode cannot diverge.

### Phase 2 — `eval-tokenizer` subcommand

```
cargo run --release -- eval-tokenizer --vocab vocab.txt
```

- Embedded fixtures via `include_str!` from `tests/fixtures/eval/{news,korean,code}.txt`, copied verbatim from nanochat's `tok_eval.py`.
- For each fixture: `encode`, `decode`, assert round-trip, record `(bytes, tokens, ratio)`.
- Print a table; non-zero exit on any round-trip mismatch.

### Phase 3 (deferred) — baseline comparison

- Add `--baseline <path>` flag pointing at a separately-downloaded GPT-2 / cl100k tiktoken file.
- Same encode/decode/ratio loop; print side-by-side with relative diff %.
- Out of scope for v1: shipping a downloader or mirroring nanochat's `from_pretrained`.

## Fixtures

v1 ships with three fixtures, copied verbatim from nanochat's `tok_eval.py`:

| Fixture | What it stresses |
|---|---|
| `news.txt` | English prose — common case |
| `korean.txt` | Multi-byte UTF-8, non-Latin script |
| `code.txt` | Punctuation-heavy, indentation, identifiers |

Verbatim copy (not handpicked) so numbers stay directly comparable to nanochat's published results if/when we add baselines.

**Deferred:**
- `math` / `science` — high overlap with prose and code; defer until they earn their place.
- `fwe-train` / `fwe-val` (samples from the corpus) — these measure in-distribution compression, which is the metric that matters most for pretraining throughput. At vocab=512 the number is dominated by single-byte fallbacks and isn't informative; add when vocab grows into the multi-thousand range.

## Decisions made

- **Three fixtures only in v1**, embedded via `include_str!`, copied verbatim from nanochat.
- **No baselines in v1.** Phase 3 lands as a follow-up.
- **No corpus samples in v1.** Defer until vocab is large enough for the number to be meaningful.

## Open questions

1. **Encode algorithm.** Textbook is a priority queue over adjacent pairs keyed by rank — same doubly-linked-list as training, but rank lookup instead of frequency count. Alternative: simple O(n × num_merges) per-pretoken loop. Suggest starting simple to match how training was built; optimize if it shows up in a profile.
2. **Decode return type.** `Vec<u8>` is honest (BPE is byte-level). Eval call site converts to `String` via `String::from_utf8` for round-trip assert. Confirms our fixtures round-trip cleanly without baking UTF-8 assumptions into the tokenizer.

## Risks

- **Pre-tokenization regex divergence** — encode and train MUST use the exact same regex. Mitigated by extracting to a single shared constant before Phase 1.
- **Round-trip on multi-byte UTF-8** — the regex operates on `&str` (char-boundary safe) and BPE operates on bytes. Should round-trip cleanly for valid UTF-8 input. The korean fixture is the test that proves this.
- **Tiktoken parse format** — 1-indexed ranks, base64 may or may not have padding depending on length. We wrote the writer, so we know the format, but loading is new code; needs its own tests.

## Commit order

1. Extract pre-tokenization regex to a shared `pub(crate) const` (small, safe refactor).
2. `BpeTokenizer::from_file` + `decode` + tests (decode is trivial; unblocks loading tests).
3. `BpeTokenizer::encode` + round-trip tests (including a multi-byte UTF-8 case).
4. `eval-tokenizer` subcommand wiring + fixtures + table output.
