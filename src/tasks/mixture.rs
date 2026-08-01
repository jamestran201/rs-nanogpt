//! The SFT training mixture: several parsed datasets concatenated, rendered in
//! parallel, and shuffled once with a seeded PRNG. Port of nanochat's
//! `TaskMixture` (`tasks/common.py:54-86`), which shuffles a list of
//! `(task, index)` pairs and renders lazily; we render up front instead, so the
//! packer downstream knows the whole epoch and the training horizon is exact
//! before the loop starts.
//!
//! Oversampling a dataset is the same trick as nanochat's: pass it twice
//! (identity ×2, later MMLU ×3 / GSM8K ×4). Shuffling *rendered* conversations
//! is distribution-identical to shuffling first and rendering later, and keeps
//! rendering embarrassingly parallel.
//!
//! **Determinism is load-bearing.** Under multi-GPU DDP every rank builds the
//! identical mixture and pack, then consumes a disjoint stride of the packed
//! rows — no communication, no agreement protocol (see [`super::pack`]). That
//! only works because every step of the chain is order-deterministic: sorted
//! file order (`super::dataset_files`) → sequential parquet decode →
//! order-preserving indexed parallel render → seeded ChaCha shuffle →
//! sequential pack. Nothing here may depend on thread count or scheduling.
//!
//! (Rendering nests rayon: `render_conversation` calls `encode`, which spawns
//! its own `par_iter` for messages of ≥100 pre-tokens. That is safe and
//! order-preserving — just split/join overhead once the outer map has saturated
//! the pool.)

use std::io;

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::invalid;
use crate::tokenizer::{BpeTokenizer, Conversation, RenderedConversation};

/// One dataset's contribution to the mixture. `name` is error context only —
/// a failure reports `{name}[{index}]`, where `index` is the position within
/// this source — so it should be the dataset's name (`"smoltalk"`,
/// `"identity"`, …).
///
/// Deliberately not `Clone`: oversampling passes the same dataset twice, and
/// that second copy is a multi-hundred-megabyte deep clone. Making the caller
/// write `conversations: convs.clone()` keeps the cost where it can be seen.
pub struct Source {
    pub name: &'static str,
    pub conversations: Vec<Conversation>,
}

/// Render every conversation in `sources` (truncated to `row_capacity`) and
/// return them in one seeded-shuffle order.
///
/// `row_capacity` is the packer's row length, `seq_len + 1`: truncating here
/// means every rendered conversation fits an empty row by construction, so the
/// packer never has to crop. Truncation may cut mid-assistant-span, matching
/// nanochat (`tokenizer.py:347`).
///
/// Errors are the structure checks in
/// [`render_conversation`](BpeTokenizer::render_conversation) — the one place
/// conversation shape is validated — wrapped with source context. When several
/// conversations are broken, the first in stream order is reported.
pub fn render_mixture(
    tok: &BpeTokenizer,
    sources: Vec<Source>,
    row_capacity: usize,
    seed: u64,
) -> io::Result<Vec<RenderedConversation>> {
    let total: usize = sources.iter().map(|s| s.conversations.len()).sum();
    let mut out = Vec::with_capacity(total);
    // One buffer across sources: `collect_into_vec` clears its target, so it
    // is reused rather than reallocated. (Marginal either way — there are a
    // handful of sources — but it is the shape rayon documents.)
    let mut rendered = Vec::new();

    for source in sources {
        let name = source.name;
        // Two steps on purpose. `collect_into_vec` is rayon's *documented*
        // order-preserving form (plain `collect` only preserves order through
        // an implementation route), and collecting straight into
        // `io::Result<Vec<_>>` in parallel takes rayon's unindexed path, whose
        // docs say which error surfaces is unspecified when several fail —
        // two ranks could then report different messages for the same corpus.
        // Consuming by value frees each conversation's strings as it renders.
        source
            .conversations
            .into_par_iter()
            .enumerate()
            .map(|(index, conv)| {
                tok.render_conversation(&conv, row_capacity)
                    .map_err(|e| invalid(format!("{name}[{index}]: {e}")))
            })
            .collect_into_vec(&mut rendered);
        for conv in rendered.drain(..) {
            out.push(conv?);
        }
    }

    out.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_tokenizer;
    use crate::tokenizer::{Content, Message, Role};

    fn message(role: Role, text: &str) -> Message {
        Message {
            role,
            content: Content::Text(text.into()),
        }
    }

    /// A two-turn conversation whose user text is `text` — distinct texts make
    /// rendered conversations distinguishable by their ids.
    fn conv(text: &str) -> Conversation {
        Conversation {
            messages: vec![message(Role::User, text), message(Role::Assistant, "ok")],
        }
    }

    fn source(name: &'static str, texts: &[&str]) -> Source {
        Source {
            name,
            conversations: texts.iter().map(|t| conv(t)).collect(),
        }
    }

    /// Ten conversations across two sources: enough that two seeds coinciding
    /// is not a plausible flake.
    fn fixture() -> Vec<Source> {
        vec![
            source("alpha", &["a", "b", "c", "d", "e", "f"]),
            source("beta", &["g", "h", "i", "j"]),
        ]
    }

    /// The reference the parallel render must match, element for element.
    fn sequential(tok: &BpeTokenizer, row_capacity: usize) -> Vec<RenderedConversation> {
        fixture()
            .into_iter()
            .flat_map(|s| s.conversations)
            .map(|c| tok.render_conversation(&c, row_capacity).unwrap())
            .collect()
    }

    fn sorted_ids(rendered: &[RenderedConversation]) -> Vec<Vec<u32>> {
        let mut ids: Vec<Vec<u32>> = rendered.iter().map(|r| r.ids.clone()).collect();
        ids.sort();
        ids
    }

    #[test]
    fn render_error_carries_source_and_index() {
        let tok = byte_tokenizer();
        let mut conversations = vec![conv("ok"), conv("also ok")];
        // Assistant-first: broken alternation, caught by render_conversation.
        conversations.insert(
            1,
            Conversation {
                messages: vec![message(Role::Assistant, "no user turn")],
            },
        );

        let err = render_mixture(
            &tok,
            vec![Source {
                name: "smoltalk",
                conversations,
            }],
            64,
            42,
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        for needle in ["smoltalk[1]", "expected user, got assistant"] {
            assert!(
                msg.contains(needle),
                "error {msg:?} should contain {needle:?}"
            );
        }
    }

    /// Rendering is capped at the row capacity, so every conversation fits an
    /// empty packer row. The capacity is chosen so that only one of the two
    /// fixtures is truncated: `conv(text)` renders to `7 + text.len()` tokens
    /// with this tokenizer, so "short" lands at 12 and the long one at 207.
    #[test]
    fn truncation_caps_every_render_at_row_capacity() {
        let tok = byte_tokenizer();
        let long = "x".repeat(200);
        let sources = vec![source("s", &[&long, "short"])];

        let mut lengths: Vec<usize> = render_mixture(&tok, sources, 13, 42)
            .unwrap()
            .iter()
            .map(|r| r.ids.len())
            .collect();
        lengths.sort_unstable();
        assert_eq!(
            lengths,
            [12, 13],
            "one render untouched, one cut to the cap"
        );
    }

    /// The DDP invariant: one seed, one order — every rank builds the same
    /// mixture from the same inputs, with no communication.
    #[test]
    fn same_seed_is_a_deterministic_permutation() {
        let tok = byte_tokenizer();
        let first = render_mixture(&tok, fixture(), 64, 42).unwrap();
        let second = render_mixture(&tok, fixture(), 64, 42).unwrap();
        assert_eq!(first, second);
        assert_eq!(sorted_ids(&first), sorted_ids(&sequential(&tok, 64)));
    }

    #[test]
    fn different_seed_reorders_the_same_conversations() {
        let tok = byte_tokenizer();
        let a = render_mixture(&tok, fixture(), 64, 42).unwrap();
        let b = render_mixture(&tok, fixture(), 64, 43).unwrap();
        assert_ne!(a, b, "a different seed must give a different order");
        assert_eq!(
            sorted_ids(&a),
            sorted_ids(&b),
            "same conversations, reordered"
        );
    }

    /// Pins order-preservation of the parallel render: the parallel result must
    /// equal the sequential render shuffled with the same seed. This is what
    /// makes the pack thread-count-independent, and therefore rank-independent.
    #[test]
    fn parallel_render_matches_sequential_reference() {
        let tok = byte_tokenizer();
        let mut want = sequential(&tok, 64);
        want.shuffle(&mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(render_mixture(&tok, fixture(), 64, 42).unwrap(), want);
    }
}
