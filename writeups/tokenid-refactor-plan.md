# BPE Tokenizer `Vec<u8>` → `TokenId` Refactor Plan

## Goal

Replace the inner data representation `Vec<u8>` (heap-allocated byte sequences) with `TokenId` (a `u32` index into a shared vocabulary table) in `src/tokenizer/mod.rs`. This eliminates the per-iteration heap allocations in the BPE hot loop (`merge_pair` and `init_pair_tables`).

The unused `pub type TokenId = u32;` alias at line 15 finally gets used.

## Why this speeds things up

The win isn't `u8 → u32`; it's `Vec<u8> → u32`. A `Vec<u8>` is a 24-byte stack header plus a separate heap allocation; cloning it allocates. A `u32` is `Copy` — assignment is a register move.

Concretely, every iteration of `merge_pair`'s inner loop currently does up to 8 heap allocations just to build hashmap keys and write merged bytes into slots. After the refactor:
- `Pair = (u32, u32)` — `Copy`, hashes 8 bytes inline.
- `state.tokens[left] = new_id` — register write, no allocation.
- Hashmap key compare is a register compare instead of a length check + memcmp through two pointer dereferences.
- A `Vec<u32>` of N tokens is contiguous; the prefetcher loves it. `Vec<Vec<u8>>` is N pointers to N scattered allocations.
- Allocator pressure drops by orders of magnitude.

Only one `Vec<u8>` allocation survives per merge (in `vocab.push_merge` for the merged bytes themselves), down from O(locations) per merge.

## 1. New / Changed Types

### 1.1 `Pair` (currently line 17)

```rust
type Pair = (TokenId, TokenId);
```

A 64-bit `Copy` type. No more `clone()` on hash lookups, no allocation on insert.

### 1.2 `PreTokenState` (currently lines 19–24)

```rust
struct PreTokenState {
    tokens: Vec<TokenId>,   // renamed from `bytes`
    next: Vec<Option<usize>>,
    prev: Vec<Option<usize>>,
    count: u64,
}
```

The dead "tombstone" written into `state.bytes[right]` after a merge (line 206 — `Vec::new()`) becomes a sentinel `TokenId`. Define `const TOMBSTONE: TokenId = TokenId::MAX;` and write that in.

### 1.3 `PairInfo` (currently lines 26–30)

Unchanged — `count: u64`, `locations: Vec<(usize, usize)>`.

### 1.4 New: vocabulary table

A new owned table that maps id → bytes, but only for ids ≥ 256:

```rust
struct Vocab {
    merged: Vec<Vec<u8>>,  // merged[i] holds bytes for TokenId 256 + i
}

impl Vocab {
    fn bytes_of(&self, id: TokenId) -> &[u8] {
        if id < 256 {
            &SINGLE_BYTE_TABLE[id as usize]
        } else {
            &self.merged[(id - 256) as usize]
        }
    }
    fn push_merge(&mut self, bytes: Vec<u8>) -> TokenId {
        let id = 256 + self.merged.len() as TokenId;
        self.merged.push(bytes);
        id
    }
}
```

`SINGLE_BYTE_TABLE` is a `&'static [[u8; 1]; 256]` produced via a `const fn` or `OnceLock`. It lets `bytes_of` return a `&[u8]` uniformly without per-call allocation.

The `vocab` lives as a local variable owned by `learn_merges`, threaded by `&mut` into `merge_pair` and by `&` into `find_best_pair`. It does NOT need to be a field on `BpeTokenizerTrainer` — it has no use outside training (encoding lives elsewhere; only `write_vocab` needs the merge byte list).

### 1.5 Return type of `learn_merges`

Stays `Vec<Vec<u8>>` so `write_vocab` is unchanged. Built by moving `vocab.merged` out at the end (no clone).

## 2. Function-by-Function Changes

### 2.1 `init_pretoken_states` — line 102

**New signature:** unchanged.

**Body change at line 107:** replace
```rust
let bytes: Vec<Vec<u8>> = pretoken.as_bytes().iter().map(|b| vec![*b]).collect();
```
with
```rust
let tokens: Vec<TokenId> = pretoken.as_bytes().iter().map(|b| *b as TokenId).collect();
```
and update the struct construction.

### 2.2 `init_pair_tables` — line 124

**New signature:** unchanged.

**Body change at line 132:** replace
```rust
let pair = (state.bytes[left].clone(), state.bytes[right].clone());
```
with
```rust
let pair = (state.tokens[left], state.tokens[right]);
```
This eliminates the two `.clone()` allocations that fired once per adjacent pair across the entire corpus.

### 2.3 `find_best_pair` — line 142

**New signature:**
```rust
fn find_best_pair(pair_info: &HashMap<Pair, PairInfo>, vocab: &Vocab) -> Option<Pair>
```

**Body change at line 146:** the comparator must compare by *bytes*, not by numeric id. Replace
```rust
.max_by(|(p1, i1), (p2, i2)| i1.count.cmp(&i2.count).then_with(|| p1.cmp(p2)))
```
with
```rust
.max_by(|(p1, i1), (p2, i2)| {
    i1.count.cmp(&i2.count).then_with(|| {
        let lhs = (vocab.bytes_of(p1.0), vocab.bytes_of(p1.1));
        let rhs = (vocab.bytes_of(p2.0), vocab.bytes_of(p2.1));
        lhs.cmp(&rhs)
    })
})
```

The return type is still `Option<Pair>` (now `(TokenId, TokenId)`).

### 2.4 `merge_pair` — line 150

**New signature:**
```rust
fn merge_pair(
    pair: Pair,
    states: &mut [PreTokenState],
    pair_info: &mut HashMap<Pair, PairInfo>,
    vocab: &mut Vocab,
) -> TokenId
```

**Body changes:**
- **Lines 155–157:** compute merged bytes by concatenating `vocab.bytes_of(pair.0)` and `vocab.bytes_of(pair.1)`. Push into vocab to get the new id:
  ```rust
  let mut merged_bytes = Vec::with_capacity(
      vocab.bytes_of(pair.0).len() + vocab.bytes_of(pair.1).len()
  );
  merged_bytes.extend_from_slice(vocab.bytes_of(pair.0));
  merged_bytes.extend_from_slice(vocab.bytes_of(pair.1));
  let merged_id = vocab.push_merge(merged_bytes);
  ```
  This is **the only `Vec<u8>` allocation per merge** — once, not once per location.
- **Line 172** (the safety guard): becomes `state.tokens[left] != pair.0 || state.tokens[right] != pair.1` — a `u32` compare.
- **Line 181** (left-neighbor pair lookup): `(state.tokens[b_idx], state.tokens[left])` — copy, no clone.
- **Line 193** (right-neighbor pair lookup): `(state.tokens[right], state.tokens[a_idx])`.
- **Lines 205–206:** replace
  ```rust
  state.bytes[left] = merged.clone();
  state.bytes[right] = Vec::new();
  ```
  with
  ```rust
  state.tokens[left] = merged_id;
  state.tokens[right] = TOMBSTONE;
  ```
- **Lines 213, 219:** key construction becomes `(state.tokens[b_idx], merged_id)` and `(merged_id, state.tokens[a_idx])`.
- **Line 233:** return `merged_id` instead of `merged`.

### 2.5 `learn_merges` — line 236

**New signature:** unchanged externally (still returns `Vec<Vec<u8>>`).

**Body change:** instantiate vocab locally, drop the `merges` collection, return `vocab.merged`:
```rust
let mut vocab = Vocab { merged: Vec::with_capacity(num_merges) };
for _ in 0..num_merges {
    let Some(pair) = Self::find_best_pair(pair_info, &vocab) else { break; };
    let _new_id = Self::merge_pair(pair, states, pair_info, &mut vocab);
    if vocab.merged.len() % 100 == 0 {
        eprintln!("trained {} / {} merges", vocab.merged.len(), num_merges);
    }
}
vocab.merged
```

### 2.6 `write_vocab` — line 255

**Unchanged.** Still takes `&[Vec<u8>]`. Wire format does not change. All `write_vocab_*` and reference-parser round-trip tests pass with no edits.

### 2.7 `train` — line 274

**Unchanged signature and body.** Edge case `vocab_size = 256` (zero merges) works because `learn_merges` returns the empty `vocab.merged` vec.

## 3. Tie-Breaking Strategy (Critical)

Three options:

| Option | Cost per `find_best_pair` call | Correctness |
|---|---|---|
| A. Lookup-on-compare (recommended) | 2 `vocab.bytes_of` indexings + slice cmp per compare | Identical to current |
| B. Store byte key alongside `Pair` in the hashmap value | Zero per-compare cost, but doubles memory and duplicates source of truth | Identical |
| C. Compare by numeric `TokenId` | Cheapest | **Wrong** — id 257 (e.g. `"ab"`) would order before id 300 (e.g. `"a"`), breaking `find_best_pair_tie_broken_lexicographically` and `learn_merges_deterministic_tie_breaking` |

**Choose Option A.** `find_best_pair` runs once per merge. The hot path is `merge_pair`. Two `vocab.bytes_of` calls in the comparator are negligible compared to the savings in `merge_pair`. Each `bytes_of` is a single bounds-checked array indexing + slice borrow — no allocation.

If profiling later shows `find_best_pair` dominates, switch to Option B by augmenting `PairInfo` with a precomputed `key: (Vec<u8>, Vec<u8>)`. Follow-up, not part of this refactor.

## 4. Test Migration

### 4.1 Test helper changes (lines 671–680)

```rust
fn pair(a: TokenId, b: TokenId) -> Pair { (a, b) }
fn info(count: u64) -> PairInfo { PairInfo { count, locations: Vec::new() } }
```

For tests using byte literals `b"a"`, `b"b"`: those are pre-merge single bytes, so `pair(b'a' as TokenId, b'b' as TokenId)` works. For multi-byte byte literals (`b"ab"`, `b"the"`) used post-merge, ids must be obtained from a vocab the test builds (capture from `merge_pair` return value).

Add a new helper:

```rust
fn live_token_bytes(state: &PreTokenState, vocab: &Vocab) -> Vec<Vec<u8>> {
    let head = (0..state.tokens.len()).find(|&i| state.prev[i].is_none())
        .expect("state must have a head");
    let mut out = Vec::new();
    let mut cur = Some(head);
    while let Some(i) = cur {
        out.push(vocab.bytes_of(state.tokens[i]).to_vec());
        cur = state.next[i];
    }
    out
}
```

Update `setup` to also return a `Vocab`:

```rust
fn setup(pretokens: &[(&str, u64)]) -> (Vec<PreTokenState>, HashMap<Pair, PairInfo>, Vocab) {
    let states = BpeTokenizerTrainer::init_pretoken_states(counts_from(pretokens));
    let pair_info = BpeTokenizerTrainer::init_pair_tables(&states);
    let vocab = Vocab { merged: Vec::new() };
    (states, pair_info, vocab)
}
```

### 4.2 Direct `state.bytes` comparison

| Line | Test | New assertion |
|---|---|---|
| 570 | `init_pretoken_states_single_piece_builds_correct_links` | `assert_eq!(w.tokens, vec![b'h' as TokenId, b'i' as TokenId]);` |
| 581 | `init_pretoken_states_single_byte_piece` | `assert_eq!(w.tokens, vec![b'a' as TokenId]);` |
| 592 | `init_pretoken_states_preserves_count` | `assert_eq!(words[0].tokens, vec![b't' as TokenId, b'h' as TokenId, b'e' as TokenId]);` |
| 599–604 | `init_pretoken_states_distinct_pieces` | Replace `w.bytes.iter().flatten().copied().collect()` with `w.tokens.iter().map(|&id| id as u8).collect()` (safe because all ids < 256). |
| 613 | `init_pretoken_states_splits_multibyte_chars_into_bytes` | `assert_eq!(w.tokens, vec![0xC3 as TokenId, 0xA9 as TokenId]);` |
| 1094 | `prepare_pretoken_states_produces_nonempty_states` | `assert!(!s.tokens.is_empty());` |

### 4.3 `init_pair_tables` tests using `Pair = (Vec<u8>, Vec<u8>)`

Lines 635, 646–647, 662 — replace `(vec![b'a'], vec![b'b'])` with `(b'a' as TokenId, b'b' as TokenId)`. All pre-merge bytes; no vocab needed.

### 4.4 `find_best_pair` tests (lines 682–740)

Update every `pair(b"a", b"b")` call site to `pair(b'a' as TokenId, b'b' as TokenId)`. Same for `b"c", b"d"`, `b"e", b"f"`, `b"a", b"a"`.

`find_best_pair_tie_broken_lexicographically` (line 729) still works because all ids < 256 (numeric and lex-byte ordering coincide). **But it does not exercise the case that motivates the lookup-on-compare design.** Add a new test:

```rust
#[test]
fn find_best_pair_tie_breaks_by_bytes_not_by_token_id() {
    // id 256 = b"a" (sorts before "b" lexically).
    let vocab = Vocab { merged: vec![b"a".to_vec()] };
    let mut counts: HashMap<Pair, PairInfo> = HashMap::new();
    counts.insert((256, b'b' as TokenId), info(5));            // ("a", "b")
    counts.insert((b'b' as TokenId, b'b' as TokenId), info(5));// ("b", "b")
    // By bytes: ("b","b") > ("a","b"), so ("b","b") wins.
    // By numeric id: (256, 98) > (98, 98), would wrongly pick ("a","b").
    let best = BpeTokenizerTrainer::find_best_pair(&counts, &vocab);
    assert_eq!(best, Some((b'b' as TokenId, b'b' as TokenId)));
}
```

This pins the lex-on-bytes contract that distinguishes Option A from Option C.

### 4.5 `live_bytes` helper (line 744)

Replace it with `live_token_bytes(&state, &vocab)` from §4.1. Affected tests: lines 768, 777, 788, 799, 809, 825, 826. Each `live_bytes(&states[0])` call becomes `live_token_bytes(&states[0], &vocab)`. Also update `setup` destructuring at lines 765, 774, 786, 797, 807, 817, 835, 842, 849, 856, 863, 870, 879.

### 4.6 `merge_pair` call sites

`merge_pair` now takes `&mut vocab` and returns `TokenId`. Call-site changes:

| Line | Change |
|---|---|
| 766 | `let merged_id = BpeTokenizerTrainer::merge_pair(pair(b'a' as TokenId, b'b' as TokenId), &mut states, &mut info, &mut vocab);` |
| 767 | `assert_eq!(vocab.bytes_of(merged_id), b"ab");` (replaces `assert_eq!(merged, b"ab".to_vec());`) |
| 769, 778, 789, 800, 810, 827 | All `info.get(&pair(b"a", b"b"))` → use `(b'a' as TokenId, b'b' as TokenId)` for pre-merge keys; resolve post-merge keys (`pair(b"ab", b"c")`) by capturing the id returned from `merge_pair`. E.g. `info.get(&(merged_id, b'c' as TokenId))`. |
| 779–781, 790–792, 801–802, 811–812, 828–829 | Same translation — post-merge keys must use the captured id. |
| 775, 787, 798, 808, 818, 837 | `merge_pair` arg list grows by `&mut vocab`. |

### 4.7 `learn_merges` tests (lines 840–882)

`learn_merges` still returns `Vec<Vec<u8>>`, so existing assertions like `assert_eq!(merges, vec![b"ab".to_vec(), b"abab".to_vec()]);` (line 858) **work unchanged**. Only edit needed is destructuring `setup`: `let (mut states, mut info, _vocab) = setup(...);`.

Affected lines: 842, 849, 856, 863, 870, 879.

### 4.8 `merge_pair` panic test (line 832)

The panic guard at line 228 still fires. The test calls `merge_pair(pair(b"x", b"y"), ...)` → `pair(b'x' as TokenId, b'y' as TokenId)`. Pair is absent from `pair_info` → empty locations → panic.

**Important:** the new `merge_pair` calls `vocab.push_merge(...)` *before* iterating locations. The panic test would leave a stale entry in vocab. Two fixes:

1. Defer `vocab.push_merge` until after `merged_any` is verified — but then we don't have an id to write into `state.tokens[left]` during the loop.
2. Push to vocab first; on the panic path the leak is harmless because the trainer aborts.

**Choose (2).** The panic is an invariant violation, not a recoverable error. Document in a comment.

### 4.9 `write_vocab` and `train` tests (lines 884–1037)

**No changes.** Wire format preserved. Reference-parser round-trip tests at lines 951–975 and 1013–1037 are the canary.

### 4.10 Untouched tests

Lines 482–553 (regex / fit-within-budget), 1040–1110 (count_pretokens, new, read_corpus). No edits.

## 5. Order of Operations

1. **Commit 1: Introduce `Vocab`, `SINGLE_BYTE_TABLE`, `TOMBSTONE`.** No call sites changed. Compiles clean, all tests pass. Clean checkpoint.

2. **Commit 2: Refactor in one shot — types, all six functions, and tests.** Atomic because field name `bytes` → `tokens` and `Pair` shape are entangled with every test that touches them. Adapter shims aren't worth the churn.

   Within this commit, work in this order (file won't compile mid-edit, but the commit will):
   - Update `Pair` and `PreTokenState`.
   - Update `init_pretoken_states`, `init_pair_tables`.
   - Update `find_best_pair` to take `&Vocab`.
   - Update `merge_pair` to take `&mut Vocab` and return `TokenId`.
   - Update `learn_merges` to own a `Vocab` and return `vocab.merged`.
   - Leave `write_vocab` and `train` unchanged.
   - Update test helpers (`pair`, `info`, `setup`, `live_bytes` → `live_token_bytes`).
   - Update each test in §4.2–§4.8.

3. **Commit 3 (optional): Add `find_best_pair_tie_breaks_by_bytes_not_by_token_id`.** Pins the new contract.

4. **Commit 4 (optional, perf validation): Benchmark `cargo build --release` end-to-end on the `data/` corpus** with `vocab_size = 1024` before and after. Expect ~2×–10× wallclock improvement.

## 6. Risks and Validations

### 6.1 Output bit-equivalence
After commit 2, `cargo test` should pass with no edits to `write_vocab_*` or `train_*`. The reference-parser round-trip tests (lines 951–975, 1013–1037) are the strongest signal that wire format is unchanged. Run `train_output_decodes_via_reference_parser` and `train_writes_target_vocab_size`.

### 6.2 Deterministic tie-breaking
`learn_merges_deterministic_tie_breaking` (line 875) is the existing canary; new `find_best_pair_tie_breaks_by_bytes_not_by_token_id` covers the multi-byte-id case.

### 6.3 End-to-end equivalence
Run `train` against `data/` at `vocab_size = 300` (matches `train_output_decodes_via_reference_parser`) before and after. Diff the output files byte-for-byte. They MUST be identical. If they differ, the most likely cause is tie-breaking divergence — verify Option A is wired correctly.

### 6.4 Vocab leak on panic
See §4.8. Mitigation: comment in code. No test impact.

### 6.5 Surviving allocation
`vocab.merged: Vec<Vec<u8>>` still has one heap allocation per merge (the merged byte string). Unavoidable and is the *single* allocation per merge that survives — down from O(locations) per merge in the old code. If profiling shows it matters, follow up with an arena (`bumpalo`) or a flat `bytes: Vec<u8>` + `offsets: Vec<u32>` representation.

### 6.6 Tombstone confusion
`TOMBSTONE = u32::MAX` written into unlinked slots. No code today reads `state.tokens[i]` outside the linked list. If a future change does, the assertion at line 172 would catch a `MAX` flowing into a pair lookup.

### 6.7 Integer width
`TokenId = u32` supports up to 4 billion ids. No overflow risk. `b'a' as TokenId` is infallible for `u8`.

## Files affected

- `src/tokenizer/mod.rs` (production code + inline test module)

No other files in the repo are affected.
