# Plan: add `--doc-cap` to tokenizer training

## Goal

Match nanochat's per-document cap (`tok_train.py:18`, default 10,000) so a few unusually long documents can't dominate BPE pair statistics. Currently we have only a **global** byte budget (`max_chars`); a single 10MB document could swamp the next 1000 short ones.

## Current state

- `BpeTokenizerTrainer::new(corpus_path, max_chars)` — global byte budget only.
- `CorpusIter` yields `String` documents, tracks `chars_read`, terminates when `chars_read >= max_chars`.
- `read_from_batch` (`mod.rs:399`) calls `fit_within_budget(text, remaining)` to truncate **only the last** document when it would overflow the global budget.
- No per-document cap exists.

Note on naming: `max_chars` actually counts **bytes** (`text.len()` on `String` is byte length). Doc-cap follows the same convention. Renaming is out of scope.

## Design

### Where the cap applies

Inside `read_from_batch`, before the global-budget check. Cap each document to `min(doc_cap, text.len())` bytes at a char boundary, *then* apply the remaining global budget on top.

Logic sketch:
```rust
let doc_max = doc_cap.min(text.len());
let effective_max = doc_max.min(remaining);
let prefix = floor_to_char_boundary(text, effective_max);
// budget_exhausted fires ONLY when the global budget cut us short,
// not when doc_cap did.
let budget_exhausted = effective_max < doc_max;
```

The subtle bit: today `budget_exhausted = prefix.len() < text.len()` — *any* truncation terminates iteration. After the change, doc-cap truncation must NOT terminate; only global-budget truncation does.

### API shape

Add `doc_cap: usize` directly as a field on `BpeTokenizerTrainer`. Update the constructor signature:

```rust
pub fn new(
    corpus_path: impl Into<PathBuf>,
    max_chars: usize,
    doc_cap: usize,
) -> Self
```

Plumb `doc_cap` through `read_corpus → CorpusIter → read_from_batch`.

This is a breaking change to the constructor — every call site (tests, `main.rs`) updates to pass the new arg. Acceptable: small surface, single binary.

### CLI

Add to the `TrainTokenizer` subcommand:
```rust
/// Maximum bytes per document. Documents longer than this are truncated.
#[arg(long, default_value_t = 10_000)]
doc_cap: usize,
```

Default 10,000 matches nanochat. `main.rs` calls `BpeTokenizerTrainer::new(corpus, max_chars, doc_cap)`.

## Tests

Add to the existing `read_corpus_*` test cluster:

1. **`doc_cap_truncates_long_document`** — single document of 20,000 bytes, doc_cap=5,000 → yields exactly 5,000 bytes, iteration continues.
2. **`doc_cap_no_truncation_when_doc_shorter`** — document of 1,000 bytes, doc_cap=5,000 → yields full 1,000 bytes.
3. **`doc_cap_per_document`** — three 8,000-byte docs, doc_cap=3,000 → yields three 3,000-byte strings, total chars_read = 9,000.
4. **`doc_cap_with_multibyte_truncates_at_char_boundary`** — document containing UTF-8 multi-byte chars straddling the cap → prefix length ≤ doc_cap and `String::from_utf8` round-trips.
5. **`doc_cap_does_not_terminate_iteration`** — verify the iterator keeps producing items after a cap-truncated document (regression guard for the `budget_exhausted` logic).
6. **`global_budget_still_terminates`** — existing behavior preserved when the global budget hits first.

Existing tests update to pass a doc-cap large enough to be a no-op (e.g., `usize::MAX` or a value larger than any test fixture).

## Risks

- **`budget_exhausted` semantics regression** — the existing code conflates "truncated this doc" with "global budget hit." The change splits these. Test #5 pins the new behavior; without it a refactor could silently re-conflate them.
- **Breaking constructor change** — every call site updates. Mechanical; caught at compile time.
- **Default of 10,000 changes CLI default behavior** — anyone currently invoking `train-tokenizer` without `--doc-cap` will now see capping. Acceptable: README doesn't yet pin a doc-cap value; flag is documented in `--help`.
- **Char vs byte semantics** — nanochat counts characters; we count bytes. For ASCII they're identical; for the climbmix corpus (predominantly English) they're nearly identical. Documented in the flag help text; no functional impact for our typical inputs.

## Commit order

Single commit is fine — small, contained change with tests. If splitting:

1. Plumb `doc_cap` through `read_from_batch` / `CorpusIter` / `BpeTokenizerTrainer`; update existing tests; add new tests.
2. Wire CLI flag with default 10,000; update README example.
