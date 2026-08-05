//! BOS-aligned best-fit-**pad** packing of a rendered mixture into training
//! rows. Port of nanochat's `sft_data_generator_bos_bestfit`
//! (`scripts/chat_sft.py:187-305`), packing the whole epoch up front instead of
//! streaming it: the row count is then known before training starts, which is
//! what lets the existing step-based loop and LR schedule run unchanged (no
//! `last_step` / `approx_progress` globals, no per-step all-reduce).
//!
//! Every row is exactly `row_capacity = seq_len + 1` tokens and starts with
//! BOS. The difference from the pretraining assembler
//! (`crate::data::loader::BatchAssembler`) is that **a conversation is atomic**:
//! when none of the buffered conversations fits the gap left in a row, the gap
//! is padded rather than a conversation cropped — training on a dialogue that
//! ends mid-sentence teaches the wrong thing.
//!
//! Two things worth knowing before reading the code:
//!
//! * Padding is `(bos, mask = false)`. The mask alone produces the `-1`
//!   (ignore_index) targets — nanochat additionally overwrites
//!   `targets[content_len-1:] = -1` (`chat_sft.py:299-303`), which is provably
//!   the same index set once targets are shifted, so it is not ported. But pad
//!   tokens are only ignored as *targets*: they still flow through the forward
//!   pass as inputs, so `bos` must really be the tokenizer's BOS id.
//! * Conversations sharing a row attend to each other — there is no
//!   per-conversation attention mask, exactly as in nanochat's SFT packing and
//!   in our own pretraining packing.

use std::vec::IntoIter;

use candle_core::{Device, Tensor};

use crate::data::{Batch, BatchSource};
use crate::tokenizer::{RenderedConversation, TokenId};

/// Best-fit lookahead window, in conversations (`chat_sft.py:187`). Smaller
/// than the pretraining assembler's 1000: SFT conversations are long relative
/// to a row, so a wider window buys little.
const BUFFER_SIZE: usize = 100;

/// What the pack cost — the observability the padding decision needs. Printed
/// at startup so a bad mixture (say, conversations just over half a row) is
/// visible as wasted compute rather than a mysteriously slow epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackStats {
    pub rows: usize,
    pub conversations: usize,
    pub pad_tokens: usize,
    /// `rows · row_capacity` — every token in the pack, padding included.
    pub total_tokens: usize,
    /// Tokens the loss actually scores — the SFT-specific number worth seeing,
    /// since it is the fraction of the compute that produces gradient
    /// (typically 30–45%), and the guard against a mixture that scores nothing.
    ///
    /// A raw count of `true` in `mask` is exactly the number of scored
    /// *targets*, even though [`PackedEpoch::batch`] scores `mask[1..]` per row:
    /// `mask[0]` of every row is always false — a rendered conversation opens
    /// with an unscored BOS (`crate::tokenizer::BpeTokenizer::render_conversation`)
    /// and a padding tail is BOS + false — so `count(mask) == count(mask[1..])`.
    pub scored_tokens: usize,
}

impl PackStats {
    /// Share of packed tokens that are padding (0.0 for an empty pack).
    pub fn pad_fraction(&self) -> f64 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.pad_tokens as f64 / self.total_tokens as f64
        }
    }
}

/// A packed epoch: `rows · row_capacity` token ids plus the parallel loss mask,
/// held on the host and tensorized one batch at a time.
pub struct PackedEpoch {
    rows: Vec<TokenId>,
    mask: Vec<bool>,
    row_capacity: usize,
    pub stats: PackStats,
}

/// Pack every rendered conversation into rows of `row_capacity`, padding with
/// `bos` wherever nothing fits. An empty mixture packs to zero rows — whether
/// that is fatal is the caller's call, and it has to make it anyway, since a
/// mixture too small to fill one batch is equally unusable.
///
/// Memory: the row buffers are reserved up front and the mixture is consumed
/// as it is packed, so the peak holds both sides — roughly 10 bytes per
/// content token (a u32 id and a bool mask, twice over). The conversations
/// freed along the way are hundreds of thousands of small allocations and
/// cannot back the single large row buffer, so that 2× is real rather than an
/// allocator artifact.
///
/// Panics if a conversation is empty, longer than `row_capacity`, or has
/// mismatched `ids`/`mask` lengths. Render with `max_tokens = row_capacity`
/// (see [`super::mixture::render_mixture`]) and none of those can happen; the
/// first two guards are also what make the packing loop terminate.
pub fn pack_epoch(
    rendered: Vec<RenderedConversation>,
    row_capacity: usize,
    bos: TokenId,
) -> PackedEpoch {
    pack_epoch_with_buffer(rendered, row_capacity, bos, BUFFER_SIZE)
}

/// Top `buf` back up to `buffer_size` conversations from `src`, validating each
/// one as it arrives.
fn refill(
    buf: &mut Vec<RenderedConversation>,
    src: &mut IntoIter<RenderedConversation>,
    buffer_size: usize,
    row_capacity: usize,
) {
    while buf.len() < buffer_size {
        let Some(conv) = src.next() else { break };
        // Not a `debug_assert`: `RenderedConversation`'s fields are public, so
        // a caller can hand over a desynchronized pair, and the box runs
        // release builds. A short mask would silently shift every later row's
        // scoring — a wrong loss curve with no crash to trace it back to.
        assert_eq!(
            conv.ids.len(),
            conv.mask.len(),
            "rendered ids and mask must have equal length"
        );
        assert!(
            !conv.ids.is_empty() && conv.ids.len() <= row_capacity,
            "conversation of {} tokens cannot be packed into rows of {row_capacity} \
             (render the mixture with max_tokens = row_capacity)",
            conv.ids.len()
        );
        buf.push(conv);
    }
}

/// [`pack_epoch`] with an explicit lookahead window; tests pass a tiny one so
/// the best-fit decisions are hand-checkable.
fn pack_epoch_with_buffer(
    rendered: Vec<RenderedConversation>,
    row_capacity: usize,
    bos: TokenId,
    buffer_size: usize,
) -> PackedEpoch {
    assert!(
        row_capacity >= 2,
        "row capacity must be at least 2 (one input/target pair), got {row_capacity}"
    );
    assert!(buffer_size > 0, "buffer_size must be positive");

    // The pack ends up `content + pad_tokens` long, so reserving `content`
    // alone would schedule a doubling at peak size. An eighth of headroom
    // covers any pad fraction below 1/9 ≈ 11% — `PackStats::pad_fraction` is
    // what checks that on real data. Reserving more than needed is nearly
    // free (untouched capacity is never faulted in); it is falling *short*
    // that costs the growth.
    let content: usize = rendered.iter().map(|c| c.ids.len()).sum();
    let reserve = content + content / 8;
    let mut rows: Vec<TokenId> = Vec::with_capacity(reserve);
    let mut mask: Vec<bool> = Vec::with_capacity(reserve);

    let mut src = rendered.into_iter();
    let mut buf: Vec<RenderedConversation> = Vec::with_capacity(buffer_size);
    let mut conversations = 0usize;
    let mut pad_tokens = 0usize;

    loop {
        // Entering a row: this refill both bounds the first best-fit scan and
        // decides whether there is anything left to pack at all.
        refill(&mut buf, &mut src, buffer_size, row_capacity);
        if buf.is_empty() {
            break;
        }

        let row_start = rows.len();
        while rows.len() - row_start < row_capacity {
            let remaining = row_capacity - (rows.len() - row_start);

            // The largest conversation that fits, keeping the *first* on ties:
            // nanochat's `conv_len > best_len` (`chat_sft.py:243`) keeps the
            // first maximum where `max_by_key` would keep the last. The buffer
            // is in arrival order, so "first" means "earliest in the stream".
            let mut best: Option<usize> = None;
            let mut best_len = 0usize;
            for (i, conv) in buf.iter().enumerate() {
                let len = conv.ids.len();
                if len <= remaining && len > best_len {
                    best = Some(i);
                    best_len = len;
                }
            }

            match best {
                Some(i) => {
                    // `remove`, not `swap_remove`: preserves arrival order,
                    // matching nanochat's `conv_buffer.pop(best_idx)`.
                    let conv = buf.remove(i);
                    rows.extend_from_slice(&conv.ids);
                    mask.extend_from_slice(&conv.mask);
                    conversations += 1;
                    // Top up before the next scan, as nanochat does inside its
                    // row loop (`chat_sft.py:233-234`): a window that shrank as
                    // the row filled would make different best-fit choices, and
                    // so a different number of rows.
                    refill(&mut buf, &mut src, buffer_size, row_capacity);
                }
                None => {
                    // Nothing fits: pad the gap instead of cropping. The row is
                    // now full.
                    rows.extend(std::iter::repeat_n(bos, remaining));
                    mask.extend(std::iter::repeat_n(false, remaining));
                    pad_tokens += remaining;
                    break;
                }
            }
        }
    }

    let stats = PackStats {
        rows: rows.len() / row_capacity,
        conversations,
        pad_tokens,
        total_tokens: rows.len(),
        scored_tokens: mask.iter().filter(|&&scored| scored).count(),
    };
    PackedEpoch {
        rows,
        mask,
        row_capacity,
        stats,
    }
}

impl PackedEpoch {
    pub fn num_rows(&self) -> usize {
        self.rows.len() / self.row_capacity
    }

    /// `seq_len + 1`; the model's sequence length is one less.
    pub fn row_capacity(&self) -> usize {
        self.row_capacity
    }

    /// `floor(rows / (world · device_batch))` — the shared training horizon in
    /// micro-batches (an optimizer step consumes `grad_accum` of them).
    ///
    /// Every rank computes this from its own identical pack, with no
    /// communication: rank `r` holds `ceil((rows − r) / world)` rows, whose
    /// minimum over ranks is `floor(rows / world)`, and
    /// `floor(floor(R/W)/B) = floor(R/(W·B))`. The remainder — at most
    /// `world · device_batch − 1` rows — is dropped for the epoch.
    pub fn num_batches(&self, world: usize, device_batch: usize) -> usize {
        assert!(
            world >= 1 && device_batch >= 1,
            "world {world} and device_batch {device_batch} must be positive"
        );
        self.num_rows() / (world * device_batch)
    }

    /// Micro-batch `i` for rank `rank`: rows `(i·B + j)·world + rank` for
    /// `j in 0..device_batch`, so the ranks' batches are disjoint by
    /// construction.
    ///
    /// `inputs` is `row[..T]` (u32) and `targets` is `row[1..]` (i64) with
    /// `-1` wherever the shifted mask is false — unscored context (BOS, user
    /// turns, tool output) and padding alike.
    pub fn batch(
        &self,
        i: usize,
        rank: usize,
        world: usize,
        device_batch: usize,
        device: &Device,
    ) -> candle_core::Result<Batch> {
        let num_batches = self.num_batches(world, device_batch);
        assert!(rank < world, "invalid rank {rank} / world {world}");
        assert!(
            i < num_batches,
            "batch {i} out of range: the epoch holds {num_batches} batches \
             at world {world} × device_batch {device_batch}"
        );

        let tokens_dim = self.row_capacity - 1;
        let mut inputs = Vec::with_capacity(device_batch * tokens_dim);
        let mut targets = Vec::with_capacity(device_batch * tokens_dim);
        for j in 0..device_batch {
            let start = ((i * device_batch + j) * world + rank) * self.row_capacity;
            let ids = &self.rows[start..start + self.row_capacity];
            let mask = &self.mask[start..start + self.row_capacity];
            inputs.extend_from_slice(&ids[..tokens_dim]);
            targets.extend(
                ids[1..]
                    .iter()
                    .zip(&mask[1..])
                    .map(|(&id, &scored)| if scored { id as i64 } else { -1 }),
            );
        }

        Ok(Batch {
            inputs: Tensor::from_vec(inputs, (device_batch, tokens_dim), device)?,
            targets: Tensor::from_vec(targets, (device_batch, tokens_dim), device)?,
        })
    }

    /// An order-sensitive 44-bit fingerprint of the pack's ids **and** mask,
    /// for the startup check that every rank built the identical epoch.
    ///
    /// DDP does zero communication about the data during training — each rank
    /// packs the same mixture independently and strides its own rows — so
    /// nothing else would ever notice a rank whose pack diverged. FNV-1a rather
    /// than a `Σ ids`: a sum is invariant under exactly the row reordering this
    /// has to rule out. Folding the mask in catches a desynchronized mask,
    /// which [`PackStats::scored_tokens`] would only catch if the *count*
    /// changed too.
    ///
    /// Width 44 is load-bearing, not cosmetic: the check reduces this value as
    /// f64 across ranks and compares against an exactly-computed
    /// `world × local`, so every partial sum the reduction forms must itself be
    /// an exactly-representable integer. `W · (2^b − 1) < 2^53` guarantees that
    /// for *any* reduction order or tree shape, and yields `b ≤ 44` at
    /// `W ≤ 512`.
    ///
    /// The **top** bits (`h >> 20`), not a low-bit mask: in a multiplicative
    /// hash chain the low bits barely avalanche (bit 0 is just the XOR-parity
    /// of the inputs' bit 0), so masking would keep the 20 worst-mixed bits and
    /// throw away the 20 best.
    ///
    /// A method rather than a [`PackStats`] field: `PackStats` is "what the
    /// pack cost", and a hash is not a cost — as a field it would put a magic
    /// constant into every test that asserts the stats as a struct literal.
    /// Computed on demand (one pass over ~300 M ids, ~0.4 s), so a single-GPU
    /// run — where the check is vacuous — never pays for it.
    pub fn digest(&self) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        let mut h = FNV_OFFSET;
        for (&id, &scored) in self.rows.iter().zip(&self.mask) {
            h ^= id as u64 | ((scored as u64) << 32);
            h = h.wrapping_mul(FNV_PRIME);
        }
        h >> 20
    }

    /// This rank's slice of the epoch as a sequential [`BatchSource`].
    ///
    /// Collapses [`batch`](Self::batch)'s five arguments to one call site each,
    /// and makes the val-snapshot rule — build it with `rank = 0, world = 1`,
    /// because `evaluate_shard_sums` shards the snapshot itself
    /// (`crate::eval::evaluate_shard_sums`) — a constructor argument rather
    /// than a comment to remember at two call sites.
    pub fn view(&self, rank: usize, world: usize, device_batch: usize) -> EpochView<'_> {
        assert!(rank < world, "invalid rank {rank} / world {world}");
        EpochView {
            epoch: self,
            rank,
            world,
            device_batch,
            next: 0,
        }
    }
}

/// A cursor over one rank's micro-batches of a [`PackedEpoch`], in order.
pub struct EpochView<'a> {
    epoch: &'a PackedEpoch,
    rank: usize,
    world: usize,
    device_batch: usize,
    next: usize,
}

impl EpochView<'_> {
    /// Micro-batches this view can serve in total — [`PackedEpoch::num_batches`],
    /// identical on every rank by construction.
    ///
    /// Deliberately **not** `len()`: the view holds a cursor, and a `len()`
    /// that never decreases as it is consumed is a trap (`while v.len() > 0`
    /// would spin forever).
    pub fn num_batches(&self) -> usize {
        self.epoch.num_batches(self.world, self.device_batch)
    }
}

impl BatchSource for EpochView<'_> {
    fn next_batch(&mut self, device: &Device) -> candle_core::Result<Batch> {
        // Error, not wrap-around: the training horizon is computed from
        // `num_batches()` up front, so running past the end means the caller's
        // arithmetic is wrong and silently replaying the epoch would hide it.
        let total = self.num_batches();
        if self.next >= total {
            candle_core::bail!(
                "epoch view exhausted: {total} micro-batches served at world {} × \
                 device_batch {}",
                self.world,
                self.device_batch
            );
        }
        let batch =
            self.epoch
                .batch(self.next, self.rank, self.world, self.device_batch, device)?;
        self.next += 1;
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_tokenizer;
    use crate::tokenizer::{Content, Conversation, Message, Role};

    const BOS: TokenId = 99;

    /// A conversation of `len` tokens: BOS (unscored, like a real render) then
    /// `len - 1` copies of the marker `k` (scored). Distinct markers make each
    /// conversation identifiable inside a packed row, and the mask
    /// distinguishes content from padding.
    fn conv(k: TokenId, len: usize) -> RenderedConversation {
        assert!(len >= 2 && k != BOS);
        RenderedConversation {
            ids: std::iter::once(BOS)
                .chain(std::iter::repeat_n(k, len - 1))
                .collect(),
            mask: std::iter::once(false)
                .chain(std::iter::repeat_n(true, len - 1))
                .collect(),
        }
    }

    /// Rows of `capacity` where row `r` holds `r*100, r*100+1, …` — the row
    /// index is readable back from any batch, which is what the striping tests
    /// need. Built directly so a row count can be chosen exactly.
    fn epoch_of(num_rows: usize, capacity: usize) -> PackedEpoch {
        assert!(capacity < 100);
        let rows: Vec<TokenId> = (0..num_rows)
            .flat_map(|r| (0..capacity).map(move |c| (r * 100 + c) as TokenId))
            .collect();
        PackedEpoch {
            mask: vec![true; rows.len()],
            stats: PackStats {
                rows: num_rows,
                conversations: num_rows,
                pad_tokens: 0,
                total_tokens: rows.len(),
                scored_tokens: rows.len(), // the fabricated mask is all-true
            },
            rows,
            row_capacity: capacity,
        }
    }

    /// Rows of 4, 5, 3 tokens into a capacity of 8.
    ///
    /// Row 0: gap 8 takes the largest that fits (5), leaving a gap of 3 that
    /// the len-3 conversation fills exactly. Row 1: the len-4 conversation,
    /// then nothing is left, so 4 tokens of padding.
    #[test]
    fn hand_computed_layout() {
        let epoch = pack_epoch_with_buffer(vec![conv(1, 4), conv(2, 5), conv(3, 3)], 8, BOS, 3);

        #[rustfmt::skip]
        assert_eq!(epoch.rows, vec![
            BOS, 2, 2, 2, 2,  BOS, 3, 3,
            BOS, 1, 1, 1,     BOS, BOS, BOS, BOS,
        ]);
        #[rustfmt::skip]
        assert_eq!(epoch.mask, vec![
            false, true, true, true, true,  false, true, true,
            false, true, true, true,        false, false, false, false,
        ]);
        assert_eq!(
            epoch.stats,
            PackStats {
                rows: 2,
                conversations: 3,
                pad_tokens: 4,
                total_tokens: 16,
                // Row 0 scores 4 + 2, row 1 scores 3 (the BOS of each
                // conversation and the whole pad tail are unscored).
                scored_tokens: 9,
            }
        );
        assert!((epoch.stats.pad_fraction() - 0.25).abs() < 1e-12);
    }

    /// Equal-length candidates: the earlier one in the stream wins (nanochat's
    /// strict `>` comparison), so packing order is stable.
    #[test]
    fn tie_keeps_the_first_candidate() {
        let epoch = pack_epoch_with_buffer(vec![conv(1, 4), conv(2, 4)], 5, BOS, 2);
        #[rustfmt::skip]
        assert_eq!(epoch.rows, vec![
            BOS, 1, 1, 1, BOS,
            BOS, 2, 2, 2, BOS,
        ]);
        assert_eq!(epoch.stats.pad_tokens, 2);
    }

    /// Best fit only sees the buffered window, and the window slides after
    /// every placement — so the decision that reveals the window size is the
    /// row's *first* scan. Lengths 3, 2, 6 into rows of 8: a window of 2 cannot
    /// see the len-6 conversation while the row is still empty, takes the len-3
    /// one, and the len-6 arrives too late to fit the gap it left; a window of
    /// 3 sees it immediately and fills the row exactly.
    #[test]
    fn buffer_bounds_the_lookahead() {
        let stream = || vec![conv(1, 3), conv(2, 2), conv(3, 6)];

        let narrow = pack_epoch_with_buffer(stream(), 8, BOS, 2);
        #[rustfmt::skip]
        assert_eq!(narrow.rows, vec![
            BOS, 1, 1,  BOS, 2,  BOS, BOS, BOS,   // 3 + 2, then 3 padded
            BOS, 3, 3, 3, 3, 3,  BOS, BOS,        // 6, then 2 padded
        ]);

        let wide = pack_epoch_with_buffer(stream(), 8, BOS, 3);
        #[rustfmt::skip]
        assert_eq!(wide.rows, vec![
            BOS, 3, 3, 3, 3, 3,  BOS, 2,          // 6 + 2 fills the row exactly
            BOS, 1, 1,  BOS, BOS, BOS, BOS, BOS,  // 3, then 5 padded
        ]);
        // Same two rows either way here, so the same total padding — the window
        // changed *which* conversation was picked, not how much fit.
        assert_eq!(narrow.stats.pad_tokens, wide.stats.pad_tokens);
    }

    /// Every conversation survives intact and exactly once (never cropped),
    /// every row is exactly `capacity` and BOS-aligned, and the token budget
    /// balances: content + padding = rows × capacity.
    #[test]
    fn packing_conserves_every_conversation() {
        let capacity = 12;
        let want: Vec<(TokenId, usize)> =
            (1..=30u32).map(|k| (k, 2 + (k as usize * 7) % 9)).collect();
        let epoch = pack_epoch_with_buffer(
            want.iter().map(|&(k, len)| conv(k, len)).collect(),
            capacity,
            BOS,
            5,
        );

        let content: usize = want.iter().map(|&(_, len)| len).sum();
        assert_eq!(epoch.stats.conversations, want.len());
        assert_eq!(epoch.stats.total_tokens, epoch.stats.rows * capacity);
        assert_eq!(content + epoch.stats.pad_tokens, epoch.stats.total_tokens);

        // Walk each row: a BOS whose successor is scored starts a conversation
        // (identified by its marker); a BOS whose successor is not is the start
        // of the row's padding tail.
        let mut got: Vec<(TokenId, usize)> = Vec::new();
        for row in 0..epoch.stats.rows {
            let start = row * capacity;
            let ids = &epoch.rows[start..start + capacity];
            let mask = &epoch.mask[start..start + capacity];
            assert_eq!(ids[0], BOS, "row {row} must start with BOS");
            let mut pos = 0;
            while pos < capacity {
                assert_eq!(ids[pos], BOS, "row {row} position {pos} must be a boundary");
                assert!(!mask[pos], "a BOS is never a target");
                // A conversation is at least 2 tokens, so a BOS at the last
                // position — or one whose successor is unscored — is padding.
                if pos + 1 == capacity || !mask[pos + 1] {
                    // Padding tail: BOS all the way to the end of the row.
                    assert!(ids[pos..].iter().all(|&id| id == BOS));
                    assert!(mask[pos..].iter().all(|&m| !m));
                    break;
                }
                let marker = ids[pos + 1];
                let len = 1 + ids[pos + 1..]
                    .iter()
                    .zip(&mask[pos + 1..])
                    .take_while(|&(&id, &scored)| id == marker && scored)
                    .count();
                got.push((marker, len));
                pos += len;
            }
        }
        got.sort_unstable();
        let mut want = want;
        want.sort_unstable();
        assert_eq!(got, want);
    }

    #[test]
    fn packing_is_deterministic() {
        let stream = || (1..=12u32).map(|k| conv(k, 2 + (k as usize) % 5)).collect();
        let a = pack_epoch_with_buffer(stream(), 9, BOS, 4);
        let b = pack_epoch_with_buffer(stream(), 9, BOS, 4);
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.mask, b.mask);
        assert_eq!(a.stats, b.stats);
        // The cross-rank check's premise: two processes packing the same
        // mixture agree on the fingerprint.
        assert_eq!(a.digest(), b.digest());
    }

    /// `scored_tokens` counts the `true` mask positions — and that count is
    /// exactly the number of targets `batch()` does not ignore.
    #[test]
    fn scored_tokens_counts_the_unignored_targets() -> candle_core::Result<()> {
        let epoch = pack_epoch_with_buffer(vec![conv(1, 4), conv(2, 5), conv(3, 3)], 8, BOS, 3);
        assert_eq!(
            epoch.stats.scored_tokens,
            epoch.mask.iter().filter(|&&m| m).count()
        );

        // Cross-check through the consumer. Pinned to `view(0, 1, 1)`: that
        // shape covers every row, whereas any other drops the tail remainder
        // and the identity would be off by it.
        let mut view = epoch.view(0, 1, 1);
        let mut unignored = 0usize;
        for _ in 0..view.num_batches() {
            let batch = view.next_batch(&Device::Cpu)?;
            unignored += batch
                .targets
                .flatten_all()?
                .to_vec1::<i64>()?
                .iter()
                .filter(|&&t| t != -1)
                .count();
        }
        assert_eq!(unignored, epoch.stats.scored_tokens);
        Ok(())
    }

    /// The digest must catch both divergences the startup check exists for:
    /// the same conversations packed in a different order (which a `Σ ids`
    /// would miss entirely), and an identical id stream with a different mask.
    /// And it must fit the 44 bits the f64 cross-rank sum needs.
    #[test]
    fn digest_is_order_and_mask_sensitive_and_fits_44_bits() {
        let a = pack_epoch_with_buffer(vec![conv(1, 4), conv(2, 4), conv(3, 4)], 4, BOS, 1);
        let b = pack_epoch_with_buffer(vec![conv(3, 4), conv(1, 4), conv(2, 4)], 4, BOS, 1);

        // Same rows, permuted — a sum over ids cannot tell these apart.
        let sum = |e: &PackedEpoch| e.rows.iter().map(|&id| id as u64).sum::<u64>();
        assert_eq!(sum(&a), sum(&b), "the fixture must defeat a Σ ids check");
        assert_ne!(a.digest(), b.digest(), "digest must be order-sensitive");

        // Identical ids, one flipped mask bit.
        let mut c = pack_epoch_with_buffer(vec![conv(1, 4), conv(2, 4), conv(3, 4)], 4, BOS, 1);
        assert_eq!(c.rows, a.rows);
        c.mask[1] = !c.mask[1];
        assert_ne!(a.digest(), c.digest(), "digest must cover the mask");

        for e in [&a, &b, &c] {
            assert!(e.digest() < 1u64 << 44, "digest must fit 44 bits");
        }
    }

    /// Through a real render: only the assistant's text and its
    /// `<|assistant_end|>` are targets. Everything else — BOS, the user turn,
    /// `<|assistant_start|>`, and the padding tail — is `-1`.
    #[test]
    fn targets_are_ignored_exactly_where_the_shifted_mask_is_false() {
        let tok = byte_tokenizer();
        let text = |role, s: &str| Message {
            role,
            content: Content::Text(s.into()),
        };
        let conversation = Conversation {
            messages: vec![text(Role::User, "Hi"), text(Role::Assistant, "Yo")],
        };
        let rendered = tok.render_conversation(&conversation, 12).unwrap();
        assert_eq!(rendered.ids.len(), 9);

        let epoch = pack_epoch(vec![rendered], 12, tok.bos_id());
        assert_eq!(epoch.num_rows(), 1);
        let batch = epoch.batch(0, 0, 1, 1, &Device::Cpu).unwrap();

        // bos us H i ue as Y o ae, then three BOS pad tokens.
        assert_eq!(
            batch.inputs.to_vec2::<u32>().unwrap(),
            vec![vec![256, 257, 72, 105, 258, 259, 89, 111, 260, 256, 256]]
        );
        assert_eq!(
            batch.targets.to_vec2::<i64>().unwrap(),
            vec![vec![-1, -1, -1, -1, -1, 89, 111, 260, -1, -1, -1]]
        );
    }

    /// The DDP contract: across ranks and batches, `batch()` reaches each of
    /// the leading `num_batches · world · device_batch` rows exactly once. It
    /// is a prefix, not the whole epoch — the tail is deliberately dropped, and
    /// is bounded by `world · device_batch − 1` rows.
    #[test]
    fn batches_partition_the_leading_rows() {
        for &(num_rows, world, device_batch) in &[
            (12usize, 1usize, 2usize),
            (12, 2, 2),
            (12, 3, 2),
            (13, 3, 2),
            (7, 2, 3),
            (10, 3, 1),
        ] {
            let epoch = epoch_of(num_rows, 4);
            let batches = epoch.num_batches(world, device_batch);

            let mut seen: Vec<u32> = Vec::new();
            for rank in 0..world {
                for i in 0..batches {
                    let batch = epoch
                        .batch(i, rank, world, device_batch, &Device::Cpu)
                        .unwrap();
                    for row in batch.inputs.to_vec2::<u32>().unwrap() {
                        seen.push(row[0] / 100); // the row index it was built from
                    }
                }
            }
            seen.sort_unstable();

            let covered = batches * world * device_batch;
            let want: Vec<u32> = (0..covered as u32).collect();
            assert_eq!(
                seen, want,
                "rows {num_rows}, world {world}, device_batch {device_batch}"
            );
            assert_eq!(
                num_rows - covered,
                num_rows % (world * device_batch),
                "the dropped tail is exactly the remainder"
            );
        }
    }

    /// The view is exactly `batch()` plus a cursor: same horizon, same batches
    /// in index order, and running past the end is an error rather than a
    /// silent replay of the epoch.
    #[test]
    fn view_serves_the_indexed_batches_in_order_then_errors() -> candle_core::Result<()> {
        for &(num_rows, rank, world, device_batch) in &[
            (12usize, 0usize, 1usize, 2usize),
            (12, 1, 2, 2),
            (13, 2, 3, 2),
        ] {
            let epoch = epoch_of(num_rows, 4);
            let mut view = epoch.view(rank, world, device_batch);
            assert_eq!(view.num_batches(), epoch.num_batches(world, device_batch));

            for i in 0..view.num_batches() {
                let want = epoch.batch(i, rank, world, device_batch, &Device::Cpu)?;
                let got = view.next_batch(&Device::Cpu)?;
                assert_eq!(
                    want.inputs.to_vec2::<u32>()?,
                    got.inputs.to_vec2::<u32>()?,
                    "rows {num_rows}, rank {rank}/{world}, B {device_batch}, batch {i}"
                );
                assert_eq!(
                    want.targets.to_vec2::<i64>()?,
                    got.targets.to_vec2::<i64>()?
                );
            }
            assert!(
                view.next_batch(&Device::Cpu).is_err(),
                "one call past the end must error"
            );
        }
        Ok(())
    }

    /// The horizon every rank agrees on without communicating equals what the
    /// slowest rank could actually serve from its own stride.
    #[test]
    fn num_batches_matches_a_brute_force_per_rank_count() {
        for num_rows in 0..14usize {
            for world in 1..=4usize {
                for device_batch in 1..=3usize {
                    let brute = (0..world)
                        .map(|rank| {
                            (0..num_rows).filter(|r| r % world == rank).count() / device_batch
                        })
                        .min()
                        .expect("world >= 1");
                    assert_eq!(
                        epoch_of(num_rows, 4).num_batches(world, device_batch),
                        brute,
                        "rows {num_rows}, world {world}, device_batch {device_batch}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "cannot be packed into rows")]
    fn conversation_longer_than_a_row_panics() {
        pack_epoch(vec![conv(1, 9)], 8, BOS);
    }

    /// The `<=` boundary of that guard, and the common case on real data: at
    /// seq len 2048 nearly half of smoltalk renders to exactly `row_capacity`.
    /// Such a conversation fills its row alone, with no padding.
    #[test]
    fn conversation_of_exactly_a_row_needs_no_padding() {
        let epoch = pack_epoch(vec![conv(1, 8)], 8, BOS);
        assert_eq!(epoch.rows, vec![BOS, 1, 1, 1, 1, 1, 1, 1]);
        assert_eq!(
            epoch.stats,
            PackStats {
                rows: 1,
                conversations: 1,
                pad_tokens: 0,
                total_tokens: 8,
                scored_tokens: 7, // all but the conversation's leading BOS
            }
        );
    }

    /// An empty mixture is not a panic: it packs to nothing, and the caller
    /// decides. It has to check anyway — a mixture too small to fill one batch
    /// is just as unusable, and that check covers both.
    #[test]
    fn empty_mixture_packs_to_zero_rows() {
        let epoch = pack_epoch(Vec::new(), 8, BOS);
        assert_eq!(epoch.num_rows(), 0);
        assert_eq!(epoch.num_batches(1, 1), 0);
        assert_eq!(epoch.stats.total_tokens, 0);
        assert_eq!(epoch.stats.pad_fraction(), 0.0);
    }

    /// The guardrail that replaces the old empty-mixture assert: an epoch with
    /// no batches cannot hand one out, so a caller that skips the horizon check
    /// fails loudly instead of training on nothing.
    #[test]
    #[should_panic(expected = "out of range")]
    fn zero_row_epoch_cannot_yield_a_batch() {
        pack_epoch(Vec::new(), 8, BOS)
            .batch(0, 0, 1, 1, &Device::Cpu)
            .unwrap();
    }
}
