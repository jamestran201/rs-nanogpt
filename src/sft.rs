//! Chat supervised finetuning: the SFT phase's *policy* — which datasets at
//! which multiplicities, how the val mixture is capped, how the training
//! horizon is derived — and the driver that runs it. Port of the driver half of
//! nanochat's `scripts/chat_sft.py` (everything but ChatCORE, the spelling
//! tasks, and optimizer warm-start). Spec: `writeups/sft-train-plan.md`.
//!
//! `crate::tasks` stays generic (parsers plus mixture/pack machinery); the
//! phase-specific decisions live here, next to the citations that justify them.
//!
//! The whole pipeline:
//!
//! ```text
//! checkpoint::load_meta  → seq_len, vocab_size          (cheap checks first)
//! checkpoint::load       → model + VarMap
//! val_batches            → the eval snapshot, capped and packed
//! train_sources          → Vec<Source>
//! render_mixture         → Vec<RenderedConversation>    (seeded shuffle)
//! pack_epoch             → PackedEpoch                  (+ PackStats printed)
//! epoch.view(rank, …)    → EpochView                    (a BatchSource)
//! train(…)               → <run>/best
//! ```
//!
//! **Every rank runs all of this independently and identically** — the mixture
//! and pack are deterministic given the same files and seed, so the ranks agree
//! on the row layout without communicating, and each then consumes a disjoint
//! stride of it (`crate::tasks::pack`). Step 7 of [`run`] is the one collective
//! that checks that premise actually held.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use candle_core::{Device, Result, bail};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::checkpoint;
use crate::data::{Batch, BatchSource};
use crate::metrics::{MetricsLogger, RunMeta, SftRunMeta, unique_run_dir, write_run_json};
use crate::tasks::mixture::{Source, render_mixture};
use crate::tasks::pack::{PackedEpoch, pack_epoch};
// `tasks::Split` names train/test *files*; `data::Split` is the pretraining
// shard-position split. Only the former is wanted here, so it is imported by
// name rather than reached through a glob.
use crate::tasks::{Split, custom_json, gsm8k, mmlu, smoltalk};
use crate::tokenizer::{BpeTokenizer, Conversation};
use crate::train::{
    DistCtx, EvalContext, GroupLrs, GroupWeightDecay, TrainConfig, train, validate_batch_geometry,
};

/// The persona file `download-sft-data` fetches (`tasks/download.rs`).
const IDENTITY_FILE: &str = "identity_conversations.jsonl";

/// Copies of the identity conversations in the train mixture. Hardcoded at 2,
/// as nanochat hardcodes it (`chat_sft.py:167-168`) — it is a fixed 1 K rows
/// against smoltalk's ~460 K, so there is nothing to tune.
const IDENTITY_COPIES: usize = 2;

/// nanochat's val slice sizes, in conversations: smoltalk, mmlu, gsm8k
/// (`chat_sft.py:176-180`). Used as *ratios* — see [`val_sources`].
const NANOCHAT_VAL_CAPS: (usize, usize, usize) = (24_000, 5_200, 420);

/// Conversations to render per val micro-batch row, so the capped mixture packs
/// to at least `eval_steps · device_batch` rows.
///
/// **Empirical**, and the one number here that no test can validate. One row
/// per 16 conversations puts the break-even at `row_capacity / 16` — 128 tokens
/// at seq 2048 — against an assumed quarter-row average, so ~4× headroom. The
/// formula never measures that average, and with a trained vocab and an
/// MMLU-heavy mixture it could approach the break-even; only a box run settles
/// it. [`val_batches`] rebuilds once at an uncapped mixture when it does.
const VAL_HEADROOM: usize = 16;

/// Resolved SFT inputs — the CLI's flags after mapping, which is the struct
/// [`validate`](Self::validate) checks. Validating the *mapped* values rather
/// than the clap struct is deliberate: several field pairs are same-typed and
/// adjacent in meaning (`eval_every`/`eval_steps`, `mmlu_epochs`/`gsm8k_epochs`,
/// `warmdown_ratio`/`final_lr_frac`), so a crossed assignment in the mapping is
/// invisible to a validator that runs on the source.
#[derive(Debug, Clone)]
pub struct SftConfig {
    /// The pretrained checkpoint to finetune, e.g. `out-d24/<run>/best`.
    pub checkpoint: PathBuf,
    /// Directory holding the downloaded SFT datasets.
    pub data: PathBuf,
    /// Tiktoken-format vocabulary — must be the one the checkpoint was
    /// pretrained with.
    pub vocab: PathBuf,
    /// Runs root; this run writes to `<out>/<unix-timestamp>/`.
    pub out: PathBuf,
    pub device_batch: usize,
    /// Global tokens per optimizer step, summed over all ranks.
    pub total_batch: usize,
    pub gpus: usize,
    /// `None` trains exactly one epoch over the packed mixture. `Some(n)` may
    /// only *shorten* that — the pack holds no more data, and nanochat's
    /// wrap-around is not ported.
    pub num_iters: Option<usize>,
    /// Base LRs, before `init_lr_frac`. Not inherited from the checkpoint:
    /// `meta.txt` stores geometry only.
    pub lrs: GroupLrs,
    /// nanochat's SFT LR scaling, applied to all three groups
    /// (`chat_sft.py:132`).
    pub init_lr_frac: f64,
    pub warmup_steps: usize,
    pub warmdown_ratio: f64,
    pub final_lr_frac: f64,
    pub log_every: usize,
    pub eval_every: usize,
    /// Val micro-batches in the eval snapshot. Also sizes the val mixture's cap
    /// (see [`VAL_HEADROOM`]).
    pub eval_steps: usize,
    pub mmlu_epochs: usize,
    pub gsm8k_epochs: usize,
    pub seed: u64,
    /// Print sample replies every N steps (0 disables). Unlike `eval_every`,
    /// turning this off costs nothing — samples are a diagnostic, not an
    /// output.
    pub sample_every: usize,
    pub sample_tokens: usize,
    /// Greedy by default. Greedy is the fastest read on whether the model has
    /// learned the chat *format* — turn structure, stopping — which is what an
    /// in-loop sample is for; it is a poor read on reply *quality*, since
    /// greedy decoding from a partly-finetuned model is the classic
    /// degenerate-repetition regime. Judge quality with `chat`.
    pub sample_temperature: f64,
}

impl SftConfig {
    /// Everything checkable without touching the checkpoint or the data.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.device_batch == 0 {
            return Err("--device-batch must be >= 1".into());
        }
        if self.gpus == 0 {
            return Err("--gpus must be >= 1".into());
        }
        if self.log_every == 0 {
            return Err("--log-every must be >= 1".into());
        }
        if !(0.0..=1.0).contains(&self.warmdown_ratio) {
            return Err(format!(
                "--warmdown-ratio must be in [0, 1], got {}",
                self.warmdown_ratio
            ));
        }
        if !(0.0..=1.0).contains(&self.final_lr_frac) {
            return Err(format!(
                "--final-lr-frac must be in [0, 1], got {}",
                self.final_lr_frac
            ));
        }
        // `is_finite` first so NaN and inf are rejected too — `x <= 0.0` alone
        // is false for NaN, which would let it through.
        if !self.init_lr_frac.is_finite() || self.init_lr_frac <= 0.0 {
            return Err(format!(
                "--init-lr-frac must be > 0, got {}",
                self.init_lr_frac
            ));
        }
        // The three LRs are *required* flags with no defaults, so a typo'd
        // `--matrix-lr 0` is a real possibility — and it would freeze every
        // block matrix for the whole run while the loss curve still moved
        // (the embeddings keep training), which is exactly the failure that
        // does not announce itself.
        for (name, lr) in [
            ("--embedding-lr", self.lrs.embedding),
            ("--unembedding-lr", self.lrs.unembedding),
            ("--matrix-lr", self.lrs.matrix),
        ] {
            if !lr.is_finite() || lr <= 0.0 {
                return Err(format!("{name} must be > 0, got {lr}"));
            }
        }
        if self.num_iters == Some(0) {
            return Err("--num-iters must be >= 1".into());
        }
        if self.eval_steps == 0 {
            return Err("--eval-steps must be >= 1".into());
        }
        // Unlike pretrain, where `--eval-every 0` documents "no eval, no
        // checkpointing". `<run>/best` is an SFT run's only output, so
        // disabling eval would train for an hour and save nothing.
        if self.eval_every == 0 {
            return Err("--eval-every must be >= 1 (it is what saves <run>/best)".into());
        }
        // `--sample-every 0` *is* legal, unlike `--eval-every 0`: a sample is a
        // diagnostic, so switching it off loses nothing.
        if self.sample_every > 0 && self.sample_tokens == 0 {
            return Err("--sample-tokens must be >= 1 when --sample-every > 0".into());
        }
        if self.sample_temperature < 0.0 {
            return Err(format!(
                "--sample-temperature must be >= 0, got {}",
                self.sample_temperature
            ));
        }
        Ok(())
    }
}

/// Push `copies` copies of `conversations` as sources named `name`.
///
/// The last copy moves rather than clones, so N copies cost N−1 deep clones.
/// `Source` is deliberately not `Clone` (`tasks/mixture.rs:44`) so that cost
/// stays visible; this is the one place that pays it.
fn push_copies(
    out: &mut Vec<Source>,
    name: &'static str,
    conversations: Vec<Conversation>,
    copies: usize,
) {
    for _ in 1..copies {
        out.push(Source {
            name,
            conversations: conversations.clone(),
        });
    }
    if copies > 0 {
        out.push(Source {
            name,
            conversations,
        });
    }
}

/// The train mixture, in nanochat's order (`chat_sft.py:165-173`): smoltalk
/// train once, the identity conversations twice, then MMLU and GSM8K
/// oversampled by their epoch counts. Zero epochs omits a dataset entirely,
/// which is the staging configuration.
///
/// Cost this accepts: at the defaults ~330 K of ~800 K renders are re-renders
/// of an identical conversation, on every process. If the render time printed
/// by [`run`] turns out to be minutes, the first mitigation is to render each
/// oversampled source once and clone the *rendered* result.
pub fn train_sources(
    dir: &Path,
    mmlu_epochs: usize,
    gsm8k_epochs: usize,
) -> io::Result<Vec<Source>> {
    let mut out = Vec::new();
    push_copies(&mut out, "smoltalk", smoltalk::load(dir, Split::Train)?, 1);
    push_copies(
        &mut out,
        "identity",
        custom_json::load(&dir.join(IDENTITY_FILE))?,
        IDENTITY_COPIES,
    );
    if mmlu_epochs > 0 {
        push_copies(
            &mut out,
            "mmlu",
            mmlu::load(dir, Split::Train)?,
            mmlu_epochs,
        );
    }
    if gsm8k_epochs > 0 {
        push_copies(
            &mut out,
            "gsm8k",
            gsm8k::load(dir, Split::Train)?,
            gsm8k_epochs,
        );
    }
    Ok(out)
}

/// Shuffle, then cap — never a prefix.
///
/// nanochat's tasks `.shuffle(seed=42)` at load, so its `stop=` slices are
/// random subsets. Ours return file order, and MMLU's test file is ordered by
/// subject, so truncating alone would keep only the alphabetically-first
/// subjects and score the model on a sliver of the domain.
fn shuffled_cap(mut convs: Vec<Conversation>, cap: usize, seed: u64) -> Vec<Conversation> {
    convs.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    convs.truncate(cap);
    convs
}

/// The nanochat caps of the sources this staging actually loads: the budget's
/// denominator, and equally the budget at or above which capping stops binding
/// at all (`f` clamps to 1.0). [`val_batches`] needs the second reading.
fn active_val_denominator(mmlu_epochs: usize, gsm8k_epochs: usize) -> usize {
    let (smoltalk_cap, mmlu_cap, gsm8k_cap) = NANOCHAT_VAL_CAPS;
    let mut denominator = smoltalk_cap;
    if mmlu_epochs > 0 {
        denominator += mmlu_cap;
    }
    if gsm8k_epochs > 0 {
        denominator += gsm8k_cap;
    }
    denominator
}

/// nanochat's cap scaled by `f`, floored at one conversation and clamped to
/// what the file actually holds.
fn scaled_cap(nanochat_cap: usize, f: f64, available: usize) -> usize {
    ((nanochat_cap as f64 * f).round() as usize)
        .max(1)
        .min(available)
}

/// The val mixture: the same datasets the train mixture uses, from their test
/// splits, each shuffled and capped to a share of `conversation_budget`.
///
/// nanochat evaluates a fixed `(24 000, 5 200, 420)` conversations. Rendering
/// and packing ~30 K conversations on every process to read a few hundred rows
/// of the result is a second full render+pack pass at ~1% utilization, so the
/// caps are instead scaled to what the snapshot will actually consume:
///
/// ```text
/// f     = min(1, budget / Σ nanochat caps of the *active* sources)
/// cap_i = max(1, round(nanochat_cap_i · f)).min(available_i)
/// ```
///
/// Ratios between active sources are exactly nanochat's. The denominator counts
/// only active sources, so a staging run with MMLU and GSM8K off spends the
/// whole budget on smoltalk rather than 81% of it.
///
/// A dataset is here only when its train epochs are `> 0` — a val bpb dominated
/// by formats the model never saw would not be a useful checkpoint selector.
/// (Divergence: nanochat's val mixture is fixed.)
pub fn val_sources(
    dir: &Path,
    mmlu_epochs: usize,
    gsm8k_epochs: usize,
    conversation_budget: usize,
    seed: u64,
) -> io::Result<Vec<Source>> {
    let (smoltalk_cap, mmlu_cap, gsm8k_cap) = NANOCHAT_VAL_CAPS;
    let f = (conversation_budget as f64 / active_val_denominator(mmlu_epochs, gsm8k_epochs) as f64)
        .min(1.0);

    let mut out = Vec::new();
    let add = |out: &mut Vec<Source>,
               name: &'static str,
               convs: Vec<Conversation>,
               nanochat_cap: usize| {
        let cap = scaled_cap(nanochat_cap, f, convs.len());
        out.push(Source {
            name,
            conversations: shuffled_cap(convs, cap, seed),
        });
    };

    add(
        &mut out,
        "smoltalk",
        smoltalk::load(dir, Split::Test)?,
        smoltalk_cap,
    );
    if mmlu_epochs > 0 {
        add(&mut out, "mmlu", mmlu::load(dir, Split::Test)?, mmlu_cap);
    }
    if gsm8k_epochs > 0 {
        add(&mut out, "gsm8k", gsm8k::load(dir, Split::Test)?, gsm8k_cap);
    }
    Ok(out)
}

/// The eval snapshot: `cfg.eval_steps` micro-batches off the packed val
/// mixture, plus the row count the pack produced.
///
/// The mixture is capped to the snapshot's budget, then rebuilt uncapped if
/// that cap made it too small to serve — see the retry below.
///
/// Built at **rank 0, world 1** on every rank: the snapshot is shared and
/// `evaluate_shard_sums` shards it itself (`crate::eval::evaluate_shard_sums`),
/// so striding it here would score each row `world` times over.
///
/// Shared with the capstone test, so the test cannot drift from what `run()`
/// actually evaluates on.
fn val_batches(
    cfg: &SftConfig,
    tok: &BpeTokenizer,
    seq_len: usize,
    device: &Device,
) -> Result<(Vec<Batch>, usize)> {
    // `saturating_mul`: two of the three factors are user-supplied, and a
    // wrapped budget in a release build would silently cap the mixture to
    // almost nothing instead of failing.
    let budget = VAL_HEADROOM
        .saturating_mul(cfg.eval_steps)
        .saturating_mul(cfg.device_batch);
    let mut epoch = pack_val(cfg, tok, seq_len, budget)?;

    // The capped budget is proportional to `eval_steps · device_batch`, so the
    // rows it produces are too, and the shortfall test below cancels both out:
    // `num_batches ≈ VAL_HEADROOM · eval_steps · rows_per_conversation`, so
    // `num_batches >= eval_steps` reduces to `rows_per_conversation >= 1 /
    // VAL_HEADROOM` — true or false regardless of either flag. Lowering
    // --eval-steps would shrink the mixture by exactly the same factor and fail
    // again, so telling the user to do that would send them in a circle.
    //
    // The retry is [`VAL_HEADROOM`]'s documented escape hatch: rebuild once at
    // nanochat's full caps (`f = 1.0`, the budget no longer binding). That
    // mixture is *fixed*, so the check finally depends on --eval-steps and the
    // error below can give advice that works. Only reached when the run would
    // otherwise abort, so the render+pack it costs is never on the happy path.
    //
    // Gated on the budget actually binding: past the denominator `f` is already
    // clamped to 1.0, so the "retry" would re-parse, re-render and re-pack the
    // identical mixture. That is the *likely* way this error fires (an
    // --eval-steps set far too high), which is exactly where the retry cannot
    // help.
    if budget < active_val_denominator(cfg.mmlu_epochs, cfg.gsm8k_epochs)
        && epoch.num_batches(1, cfg.device_batch) < cfg.eval_steps
    {
        epoch = pack_val(cfg, tok, seq_len, usize::MAX)?;
    }

    // Either branch above leaves `epoch` holding the uncapped mixture, so both
    // messages below describe a mixture no *snapshot-sizing* flag can widen
    // further. (`--mmlu-epochs`/`--gsm8k-epochs` still would, by adding sources
    // — but they are tied to the train mixture, so advising them here would
    // trade one wrong run for another.)
    let mut view = epoch.view(0, 1, cfg.device_batch);
    if view.num_batches() == 0 {
        // --eval-steps cannot fix this one: the message below would read "lower
        // --eval-steps to <= 0", which `SftConfig::validate` rejects.
        bail!(
            "the val mixture packs to {} rows, fewer than one micro-batch of --device-batch \
             {}; lower --device-batch, or check that --data holds the test split you expect",
            epoch.stats.rows,
            cfg.device_batch
        );
    }
    if view.num_batches() < cfg.eval_steps {
        bail!(
            "the val mixture packs to {} micro-batches at --device-batch {}, fewer than \
             --eval-steps {}; lower --eval-steps to <= {} (or lower --device-batch, which \
             splits the same rows into more micro-batches)",
            view.num_batches(),
            cfg.device_batch,
            cfg.eval_steps,
            view.num_batches()
        );
    }
    let batches = (0..cfg.eval_steps)
        .map(|_| view.next_batch(device))
        .collect::<Result<Vec<_>>>()?;
    Ok((batches, epoch.stats.rows))
}

/// Removes the run directory if [`run`] bails before writing anything into it.
///
/// The dir is claimed at step 4b so a bad `--out` fails in seconds, which puts
/// the val snapshot, both renders, the pack and the cross-rank check between
/// the claim and `run.json` — seven ways to exit leaving an empty directory in
/// `out-sft/`. `remove_dir` refuses a *non-empty* directory, so this can never
/// touch a run that reached `run.json`, `metrics.jsonl` or `best/`; on the
/// success path it is a no-op whose failure is the intended outcome.
struct RemoveIfEmpty<'a>(Option<&'a Path>);

impl Drop for RemoveIfEmpty<'_> {
    fn drop(&mut self) {
        if let Some(dir) = self.0 {
            let _ = fs::remove_dir(dir);
        }
    }
}

/// Build and pack the val mixture at a given conversation budget.
fn pack_val(
    cfg: &SftConfig,
    tok: &BpeTokenizer,
    seq_len: usize,
    budget: usize,
) -> Result<PackedEpoch> {
    let sources = val_sources(
        &cfg.data,
        cfg.mmlu_epochs,
        cfg.gsm8k_epochs,
        budget,
        cfg.seed,
    )?;
    let rendered = render_mixture(tok, sources, seq_len + 1, cfg.seed)?;
    Ok(pack_epoch(rendered, seq_len + 1, tok.bos_id()))
}

/// Every rank built the same pack, or the run is silently training on
/// overlapping data — and with zero communication about the data during
/// training, nothing later would ever notice.
///
/// The comparison is `sum == world × local` on values that are all exact
/// integers in f64 (`crate::tasks::pack::PackedEpoch::digest` is bounded at 44
/// bits precisely so its cross-rank sum stays exact for any reduction tree).
/// It must stay an **exact** comparison: relaxing it to an epsilon test would
/// disable the check.
fn check_packs_agree(dist: &DistCtx, local: [f64; 5]) -> Result<()> {
    const NAMES: [&str; 5] = ["rows", "pad_tokens", "scored_tokens", "digest", "val_rows"];
    let world = dist.world_size();
    let reduced = dist.all_reduce_sums(&local)?;
    for ((name, want), got) in NAMES.iter().zip(local).zip(reduced) {
        if got != want * world as f64 {
            bail!(
                "ranks disagree on the packed epoch: {name} is {want} on rank {} but the \
                 cross-rank sum is {got}, not {}. Every rank must build an identical pack \
                 from identical files and --seed",
                dist.rank(),
                want * world as f64
            );
        }
    }
    Ok(())
}

/// Finetune the checkpoint at `cfg.checkpoint` on the chat mixture for one
/// epoch, leaving the best-val-bpb model in `<run>/best`.
pub fn run(cfg: &SftConfig, device: &Device, dist: &DistCtx) -> Result<()> {
    // 0. Defend against our own callers: the capstone calls `run` directly, and
    // without this a malformed config would sail past every range check into
    // the geometry math below.
    cfg.validate().map_err(candle_core::Error::msg)?;

    // The world size is read from two places below — `cfg.gpus` drives the
    // batch geometry, `run.json` and the per-rank token share, while `dist`
    // drives the epoch stride, the gradient reduce and `ignore_rescale`'s world
    // factor — so they have to be the same number. Nothing else catches a
    // disagreement: at `gpus: 8` under a single-process `DistCtx` every check
    // below still passes (both sides of the shape assertion are divided by 8),
    // the view serves every row, and the run trains at an eighth of the
    // intended global batch while `run.json` claims eight ranks.
    if cfg.gpus != dist.world_size() {
        bail!(
            "--gpus {} disagrees with the distributed world size {}",
            cfg.gpus,
            dist.world_size()
        );
    }

    // 1-3. Everything cheap, before the safetensors read and long before the
    // ~300 M-token render.
    let meta = checkpoint::load_meta(&cfg.checkpoint)?;
    let seq_len = meta.config.sequence_len;

    let tok = BpeTokenizer::from_file(&cfg.vocab)?;
    if tok.vocab_size() != meta.config.vocab_size {
        bail!(
            "vocab {} has {} tokens but the checkpoint was trained with {}; a mismatched \
             vocabulary loads fine and scores garbage",
            cfg.vocab.display(),
            tok.vocab_size(),
            meta.config.vocab_size
        );
    }
    let token_bytes = tok.token_byte_lengths();

    let grad_accum = validate_batch_geometry(cfg.gpus, cfg.device_batch, seq_len, cfg.total_batch)
        .map_err(candle_core::Error::msg)?;

    if dist.is_master() {
        println!(
            "sft: finetuning {} | seq_len {seq_len} | vocab {} | gpus {} | device_batch {} | \
             grad_accum {grad_accum}",
            cfg.checkpoint.display(),
            meta.config.vocab_size,
            cfg.gpus,
            cfg.device_batch,
        );
    }

    // 4. Weights now, not later: `RunMeta` needs `n_params` from the VarMap
    // below, and a truncated model.safetensors should surface in seconds rather
    // than after the train render. The weights sit in device memory while the
    // host-side packs are built, which costs nothing.
    let (model, varmap, _) = checkpoint::load(&cfg.checkpoint, device)?;
    let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();

    // 4b. Claim the run dir before the renders, for the same reason step 5 runs
    // before step 6: an unwritable, misspelled or full `--out` is a
    // purely-CLI-driven mistake, and it should surface in seconds rather than
    // after ~800 K renders and a ~300 M-token pack on every rank. `run.json`
    // still waits for step 9 — it reports pack stats that do not exist yet, but
    // the *directory* does not depend on them.
    //
    // Master-only: no other rank ever writes here, so one dir per rank would
    // only race their timestamps into separate directories.
    let started_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let run_dir = if dist.is_master() {
        let dir = unique_run_dir(&cfg.out, &started_at_unix.to_string())?;
        println!("run dir: {}", dir.display());
        dir
    } else {
        cfg.out.clone()
    };
    // Everything from here to step 9 can still bail; none of it writes into the
    // run dir. Declared after `run_dir` so it drops first.
    let _cleanup = RemoveIfEmpty(dist.is_master().then_some(run_dir.as_path()));

    // 5. Val snapshot *first*, and in its own scope. First because a mis-sized
    // --eval-steps is the costliest thing a purely CLI-driven mistake can break,
    // and the user should hear about it in seconds. Scoped so the val pack and
    // its rendered input are freed before the train render allocates, instead
    // of the two host-memory peaks overlapping.
    let t_val = Instant::now();
    let (val_snapshot, val_rows) = val_batches(cfg, &tok, seq_len, device)?;
    if dist.is_master() {
        println!(
            "val mixture: {val_rows} rows -> {} micro-batches in {:.1}s",
            val_snapshot.len(),
            t_val.elapsed().as_secs_f64()
        );
    }

    // 6. The train epoch.
    let t_train_pack = Instant::now();
    let sources = train_sources(&cfg.data, cfg.mmlu_epochs, cfg.gsm8k_epochs)?;
    let rendered = render_mixture(&tok, sources, seq_len + 1, cfg.seed)?;
    let epoch = pack_epoch(rendered, seq_len + 1, tok.bos_id());
    let stats = epoch.stats;
    if stats.scored_tokens == 0 {
        bail!(
            "the train mixture packs {} rows but scores no tokens — every target is \
             ignore_index, so there is nothing to learn from",
            stats.rows
        );
    }
    let scored_fraction = stats.scored_tokens as f64 / stats.total_tokens as f64;
    if dist.is_master() {
        println!(
            "train mixture: {} conversations -> {} rows ({} tokens, {:.1}% padding, \
             {:.1}% scored) in {:.1}s",
            stats.conversations,
            stats.rows,
            stats.total_tokens,
            100.0 * stats.pad_fraction(),
            100.0 * scored_fraction,
            t_train_pack.elapsed().as_secs_f64(),
        );
    }

    // 7. The one collective that proves the no-communication premise. Skipped
    // at world 1, where it is vacuous — and skipping it there also saves the
    // digest's full pass over the pack on single-GPU startup.
    if dist.world_size() > 1 {
        check_packs_agree(
            dist,
            [
                stats.rows as f64,
                stats.pad_tokens as f64,
                stats.scored_tokens as f64,
                epoch.digest() as f64,
                val_rows as f64,
            ],
        )?;
    }

    // 8. The horizon, known exactly because the epoch is packed up front.
    let mut train_view = epoch.view(dist.rank(), dist.world_size(), cfg.device_batch);
    let epoch_iters = train_view.num_batches() / grad_accum;
    if epoch_iters == 0 {
        bail!(
            "the train mixture packs only {} rows, too few for one optimizer step: that \
             needs world({}) * device_batch({}) * grad_accum({grad_accum}) = {} rows",
            stats.rows,
            dist.world_size(),
            cfg.device_batch,
            dist.world_size() * cfg.device_batch * grad_accum
        );
    }
    let num_iters = match cfg.num_iters {
        None => epoch_iters,
        Some(n) if n <= epoch_iters => n,
        Some(n) => bail!(
            "--num-iters {n} exceeds the {epoch_iters} steps this epoch holds; the mixture \
             is packed once and not repeated, so lower --num-iters (or drop it to train \
             the whole epoch)"
        ),
    };

    // The LRs the optimizer is actually built with — what run.json records, and
    // the only place `init_lr_frac` becomes observable.
    let lrs = GroupLrs {
        embedding: cfg.lrs.embedding * cfg.init_lr_frac,
        unembedding: cfg.lrs.unembedding * cfg.init_lr_frac,
        matrix: cfg.lrs.matrix * cfg.init_lr_frac,
    };

    // 9. run.json and metrics.jsonl, master-only like the run dir claimed in
    // step 4b. Here rather than there because `SftRunMeta` reports the pack
    // stats, which only exist once step 6 has run.
    let metrics = if dist.is_master() {
        let run_meta = RunMeta {
            phase: "sft",
            device: format!("{device:?}"),
            dtype: "f32",
            started_at_unix,
            n_params,
            vocab_size: meta.config.vocab_size,
            sequence_len: seq_len,
            n_layer: meta.config.n_layer,
            n_head: meta.config.n_head,
            n_embd: meta.config.n_embd,
            rope_base: meta.config.rope_base,
            norm_eps: meta.config.norm_eps,
            num_iters,
            device_batch: cfg.device_batch,
            total_batch: cfg.total_batch,
            world_size: cfg.gpus,
            grad_accum,
            // Global, as pretraining records it, so runs stay comparable
            // across GPU counts; TrainConfig's field is the per-rank share.
            tokens_per_step: cfg.total_batch,
            embedding_lr: lrs.embedding,
            unembedding_lr: lrs.unembedding,
            matrix_lr: lrs.matrix,
            warmup_steps: cfg.warmup_steps,
            warmdown_ratio: cfg.warmdown_ratio,
            final_lr_frac: cfg.final_lr_frac,
            log_every: cfg.log_every,
            eval_every: cfg.eval_every,
            eval_steps: cfg.eval_steps,
            sample_every: cfg.sample_every,
            sft: Some(SftRunMeta {
                base_checkpoint: cfg.checkpoint.display().to_string(),
                seed: cfg.seed,
                mmlu_epochs: cfg.mmlu_epochs,
                gsm8k_epochs: cfg.gsm8k_epochs,
                conversations: stats.conversations,
                rows: stats.rows,
                pad_fraction: stats.pad_fraction(),
                scored_fraction,
                val_rows,
            }),
        };
        write_run_json(&run_dir.join("run.json"), &run_meta)?;
        MetricsLogger::create(&run_dir.join("metrics.jsonl"))?
    } else {
        MetricsLogger::null()
    };

    // 10-12.
    let tcfg = TrainConfig {
        num_iters,
        grad_accum,
        tokens_per_step: cfg.total_batch / cfg.gpus,
        // The whole point of SFT: only assistant spans are targets.
        targets_may_be_ignored: true,
        lrs,
        // nanochat SFT's `weight_decay=0.0` (chat_sft.py:134) reaches the
        // Muon/matrix groups only (gpt.py:407); pretraining anneals that decay
        // to zero by its last step (base_train.py:385), so SFT continues at
        // zero. The AdamW embedding/unembedding decays are fixed in both
        // phases, so they keep their defaults.
        weight_decays: GroupWeightDecay {
            matrix: 0.0,
            ..Default::default()
        },
        warmup_steps: cfg.warmup_steps,
        warmdown_ratio: cfg.warmdown_ratio,
        final_lr_frac: cfg.final_lr_frac,
        log_every: cfg.log_every,
    };
    let eval = EvalContext {
        val_batches: &val_snapshot,
        tokenizer: &tok,
        token_bytes: &token_bytes,
        ckpt_root: &run_dir,
        metrics: &metrics,
        eval_every: cfg.eval_every,
        sample_every: cfg.sample_every,
        sample_tokens: cfg.sample_tokens,
        sample_temperature: cfg.sample_temperature,
        // Chat-format prompts and the `<|assistant_end|>` stop set: the whole
        // point of an in-loop sample here is to show the turn format being
        // learned, which base-format completion stems would not.
        sample_chat: true,
    };
    train(&model, &varmap, &mut train_view, &tcfg, &eval, dist, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use crate::test_support::{write_gsm8k_shard, write_mmlu_shard, write_smoltalk_shard};

    /// A known-good config; tests mutate one field to probe a single rule.
    /// Paths are never touched by `validate`, so dummies are fine.
    fn valid_config() -> SftConfig {
        SftConfig {
            checkpoint: PathBuf::from("unused"),
            data: PathBuf::from("unused"),
            vocab: PathBuf::from("unused"),
            out: PathBuf::from("unused"),
            device_batch: 8,
            total_batch: 524_288,
            gpus: 1,
            num_iters: None,
            lrs: GroupLrs {
                embedding: 0.4243,
                unembedding: 0.01131,
                matrix: 0.003,
            },
            init_lr_frac: 0.8,
            warmup_steps: 0,
            warmdown_ratio: 0.5,
            final_lr_frac: 0.0,
            log_every: 10,
            eval_every: 200,
            eval_steps: 100,
            mmlu_epochs: 3,
            gsm8k_epochs: 4,
            seed: 42,
            // The CLI defaults, which this fixture mirrors — and the value
            // matters: the `sample_tokens >= 1` rule is *conditional* on
            // `sample_every > 0`, so at `sample_every: 0` the mutation below
            // would be legal and the test would fail on the wrong side.
            sample_every: 200,
            sample_tokens: 128,
            sample_temperature: 0.0,
        }
    }

    fn rejects(mutate: impl FnOnce(&mut SftConfig)) {
        let mut c = valid_config();
        mutate(&mut c);
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_accepts_a_good_config() {
        assert_eq!(valid_config().validate(), Ok(()));
        // Zero epochs on both oversampled datasets is the staging config, not
        // an error.
        let mut staging = valid_config();
        staging.mmlu_epochs = 0;
        staging.gsm8k_epochs = 0;
        assert_eq!(staging.validate(), Ok(()));
    }

    #[test]
    fn validate_rejects_one_rule_at_a_time() {
        rejects(|c| c.device_batch = 0);
        rejects(|c| c.gpus = 0);
        rejects(|c| c.log_every = 0);
        rejects(|c| c.warmdown_ratio = 1.5);
        rejects(|c| c.warmdown_ratio = -0.1);
        rejects(|c| c.final_lr_frac = 2.0);
        rejects(|c| c.init_lr_frac = 0.0);
        rejects(|c| c.num_iters = Some(0));
        rejects(|c| c.eval_steps = 0);
        // Unlike pretrain: <run>/best is the only output, so no-eval means the
        // run produces nothing.
        rejects(|c| c.eval_every = 0);
        rejects(|c| c.sample_tokens = 0);
        rejects(|c| c.sample_temperature = -0.1);
        // ...but sampling off entirely is fine, and then sample_tokens is moot.
        let mut off = valid_config();
        off.sample_every = 0;
        off.sample_tokens = 0;
        assert_eq!(off.validate(), Ok(()));
    }

    /// Each LR is checked separately — a typo'd zero on any one of the three
    /// required flags silently freezes that whole parameter group.
    #[test]
    fn validate_rejects_a_non_positive_lr_in_any_group() {
        rejects(|c| c.lrs.embedding = 0.0);
        rejects(|c| c.lrs.unembedding = 0.0);
        rejects(|c| c.lrs.matrix = 0.0);
        rejects(|c| c.lrs.matrix = -0.001);
    }

    /// One conversation per row, with `text` as the user turn.
    fn smoltalk_rows(texts: &[String]) -> Vec<Vec<(&str, &str)>> {
        texts
            .iter()
            .map(|t| vec![("user", t.as_str()), ("assistant", "ok")])
            .collect()
    }

    fn write_smoltalk(path: &Path, texts: &[String]) {
        let rows = smoltalk_rows(texts);
        let refs: Vec<&[(&str, &str)]> = rows.iter().map(|r| r.as_slice()).collect();
        write_smoltalk_shard(path, &refs);
    }

    fn write_identity(path: &Path, count: usize) {
        let mut file = std::fs::File::create(path).unwrap();
        for i in 0..count {
            writeln!(
                file,
                r#"[{{"role":"user","content":"Who are you? ({i})"}},{{"role":"assistant","content":"I am nanochat."}}]"#
            )
            .unwrap();
        }
    }

    /// A data dir with every dataset present in both splits. `marker` prefixes
    /// the smoltalk user turns so file order is recoverable from the parsed
    /// conversations.
    fn data_dir(smoltalk_rows: usize) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let texts: Vec<String> = (0..smoltalk_rows).map(|i| format!("row {i:04}")).collect();
        write_smoltalk(&dir.path().join("smoltalk-train-00000.parquet"), &texts);
        write_smoltalk(&dir.path().join("smoltalk-test-00000.parquet"), &texts);
        for split in ["train", "test"] {
            write_mmlu_shard(
                &dir.path().join(format!("mmlu-{split}-00000.parquet")),
                &[("What is 2+2?", &["3", "4", "5", "6"], 1)],
            );
            write_gsm8k_shard(
                &dir.path().join(format!("gsm8k-{split}-00000.parquet")),
                &[("How many?", "It is <<1+2=3>>3.\n#### 3")],
            );
        }
        write_identity(&dir.path().join(IDENTITY_FILE), 3);
        dir
    }

    fn names_and_counts(sources: &[Source]) -> Vec<(&'static str, usize)> {
        sources
            .iter()
            .map(|s| (s.name, s.conversations.len()))
            .collect()
    }

    /// nanochat's mixture shape: smoltalk once, identity twice, then each
    /// oversampled dataset repeated exactly `epochs` times, in that order.
    #[test]
    fn train_sources_repeats_each_dataset_by_its_epoch_count() {
        let dir = data_dir(5);
        assert_eq!(
            names_and_counts(&train_sources(dir.path(), 3, 4).unwrap()),
            vec![
                ("smoltalk", 5),
                ("identity", 3),
                ("identity", 3),
                ("mmlu", 1),
                ("mmlu", 1),
                ("mmlu", 1),
                ("gsm8k", 1),
                ("gsm8k", 1),
                ("gsm8k", 1),
                ("gsm8k", 1),
            ]
        );
    }

    /// Zero epochs omits a dataset entirely — the file is never even opened,
    /// which is what lets a staging run work without MMLU/GSM8K downloaded.
    #[test]
    fn train_sources_omits_a_dataset_at_zero_epochs() {
        let dir = data_dir(5);
        assert_eq!(
            names_and_counts(&train_sources(dir.path(), 0, 0).unwrap()),
            vec![("smoltalk", 5), ("identity", 3), ("identity", 3)]
        );

        // Nothing but smoltalk and identity is read: a dir missing the other
        // two datasets still works at zero epochs, and fails at one.
        let bare = tempfile::tempdir().unwrap();
        let texts: Vec<String> = (0..3).map(|i| format!("row {i}")).collect();
        write_smoltalk(&bare.path().join("smoltalk-train-00000.parquet"), &texts);
        write_identity(&bare.path().join(IDENTITY_FILE), 2);
        assert!(train_sources(bare.path(), 0, 0).is_ok());
        // `.err()` rather than `unwrap_err()`: `Source` is not `Debug`, and
        // deriving it would mean a formatter that can print the whole mixture.
        let err = train_sources(bare.path(), 1, 0)
            .err()
            .expect("mmlu is missing, so one epoch of it must fail");
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// The production caps, asserted exactly. At `eval_steps 100,
    /// device_batch 8` the budget is 12 800 conversations against an active
    /// denominator of 24 000 + 5 200 + 420 = 29 620, so f = 0.432140…:
    /// `(10 371.37, 2 247.13, 181.499)` — note the last one rounds *down*.
    /// A tolerance-based "ratios preserved" check would hide a wrong f.
    #[test]
    fn val_caps_scale_nanochat_ratios_to_the_snapshot_budget() {
        let budget = VAL_HEADROOM * 100 * 8;
        assert_eq!(budget, 12_800);

        let f = budget as f64 / 29_620.0;
        assert_eq!(
            [
                scaled_cap(24_000, f, usize::MAX),
                scaled_cap(5_200, f, usize::MAX),
                scaled_cap(420, f, usize::MAX),
            ],
            [10_371, 2_247, 181]
        );
    }

    /// A budget too small for the ratios still yields at least one conversation
    /// per source, and never more than the file holds.
    #[test]
    fn val_caps_floor_at_one_and_clamp_to_what_exists() {
        let f = 1.0 / 29_620.0; // budget of a single conversation
        assert_eq!(scaled_cap(420, f, usize::MAX), 1, "must floor at 1, not 0");
        // f = 1.0 (a budget at or above the full mixture) is capped by supply.
        assert_eq!(scaled_cap(24_000, 1.0, 40), 40);
        assert_eq!(scaled_cap(24_000, 1.0, usize::MAX), 24_000);
    }

    /// `val_batches`'s retry premise: `usize::MAX` really does mean "uncapped".
    ///
    /// The budget reaches `f` as `budget as f64 / denominator`, so a saturated
    /// budget has to land on `f = 1.0` (nanochat's own caps, clamped to supply)
    /// rather than overflowing into something small. Without this the retry
    /// would be a second identical attempt.
    #[test]
    fn the_retry_budget_uncaps_the_val_mixture() {
        let dir = data_dir(200);

        // The capstone's shape: eval_steps 1, device_batch 2 => budget 32, all
        // of it smoltalk's, so `round(24 000 · 32/24 000)` caps at 32 of 200.
        let tight = val_sources(dir.path(), 0, 0, VAL_HEADROOM * 2, 42).unwrap();
        let uncapped = val_sources(dir.path(), 0, 0, usize::MAX, 42).unwrap();

        assert_eq!(names_and_counts(&tight), [("smoltalk", 32)]);
        assert_eq!(
            names_and_counts(&uncapped),
            [("smoltalk", 200)],
            "an uncapped budget must take everything the file holds"
        );
    }

    /// Only active sources claim budget, so a staging run spends all of it on
    /// smoltalk instead of the 81% its ratio would give.
    #[test]
    fn val_sources_follow_the_train_mixtures_staging() {
        let dir = data_dir(200);
        let budget = 100;

        let all = val_sources(dir.path(), 3, 4, budget, 42).unwrap();
        assert_eq!(
            all.iter().map(|s| s.name).collect::<Vec<_>>(),
            vec!["smoltalk", "mmlu", "gsm8k"]
        );
        // 24 000/29 620 of 100 conversations.
        assert_eq!(all[0].conversations.len(), 81);

        let staged = val_sources(dir.path(), 0, 0, budget, 42).unwrap();
        assert_eq!(names_and_counts(&staged), vec![("smoltalk", 100)]);
    }

    /// Capping is shuffle-*then*-truncate. MMLU's real test file is ordered by
    /// subject, so a prefix cap would score the model on only the
    /// alphabetically-first subjects; the marker column makes that a hard
    /// assertion here.
    #[test]
    fn val_capping_shuffles_before_truncating() {
        let dir = data_dir(200);
        let budget = 24; // 24 000/29 620 · 24 ≈ 19 rows kept out of 200

        let marker = |s: &Source| -> Vec<String> {
            s.conversations
                .iter()
                .map(|c| match &c.messages[0].content {
                    crate::tokenizer::Content::Text(t) => t.clone(),
                    other => panic!("unexpected user content {other:?}"),
                })
                .collect()
        };

        let capped = val_sources(dir.path(), 0, 0, budget, 42).unwrap();
        let kept = marker(&capped[0]);
        assert_eq!(kept.len(), budget, "the whole budget goes to smoltalk");

        let full: Vec<String> = (0..200).map(|i| format!("row {i:04}")).collect();
        assert!(
            kept.iter().all(|k| full.contains(k)),
            "the capped set must be a subset of the file"
        );
        assert_ne!(
            kept,
            full[..budget].to_vec(),
            "a file-order prefix means the shuffle was skipped"
        );

        // Deterministic for a fixed seed (every rank must agree), and the seed
        // really drives it.
        assert_eq!(
            kept,
            marker(&val_sources(dir.path(), 0, 0, budget, 42).unwrap()[0])
        );
        assert_ne!(
            kept,
            marker(&val_sources(dir.path(), 0, 0, budget, 43).unwrap()[0])
        );
    }

    /// A tiny finetunable checkpoint plus the data and vocab `run()` needs.
    /// Returns `(tempdir, config)`; the tempdir must outlive the config.
    fn capstone_fixture() -> (tempfile::TempDir, SftConfig) {
        use crate::checkpoint::CheckpointMeta;
        use crate::test_support::{tiny_gpt, write_byte_vocab};

        let root = tempfile::tempdir().unwrap();
        let data = root.path().join("sft-data");
        std::fs::create_dir_all(&data).unwrap();

        // `val_sources` loads Split::Test for every *active* dataset, and
        // `dataset_files` errors NotFound on a missing prefix — hence both a
        // train and a test smoltalk shard, and zero MMLU/GSM8K epochs (which
        // also exercises the active-source denominator).
        let texts: Vec<String> = (0..12)
            .map(|i| format!("tell me about topic number {i:03}"))
            .collect();
        write_smoltalk(&data.join("smoltalk-train-00000.parquet"), &texts);
        write_smoltalk(&data.join("smoltalk-test-00000.parquet"), &texts);
        write_identity(&data.join(IDENTITY_FILE), 6);

        // A persistent vocab: `run()` opens the path itself, so
        // `byte_tokenizer`'s NamedTempFile (dropped on return) will not do.
        let vocab = root.path().join("vocab.txt");
        write_byte_vocab(&vocab);
        let tok = BpeTokenizer::from_file(&vocab).unwrap();

        let seq_len = 64;
        let checkpoint = root.path().join("base");
        let (vm, model) = tiny_gpt(tok.vocab_size(), seq_len);
        checkpoint::save(
            &checkpoint,
            &vm,
            &CheckpointMeta {
                config: *model.config(),
                step: 0,
                val_bpb: 9.0,
            },
        )
        .unwrap();

        let cfg = SftConfig {
            checkpoint,
            data,
            vocab,
            out: root.path().join("out-sft"),
            device_batch: 2,
            total_batch: 2 * 64, // grad_accum 1
            gpus: 1,
            num_iters: Some(2),
            lrs: GroupLrs {
                embedding: 0.02,
                unembedding: 0.004,
                matrix: 0.01,
            },
            init_lr_frac: 0.8,
            warmup_steps: 1,
            warmdown_ratio: 0.5,
            final_lr_frac: 0.0,
            log_every: 1_000_000, // silence the per-step log
            eval_every: 1,
            eval_steps: 1,
            mmlu_epochs: 0,
            gsm8k_epochs: 0,
            seed: 42,
            // The capstone deliberately samples: this is the only place
            // `sample_chat: true` + `push_user_turn` + `generate_ids` +
            // `assistant_stop_ids` + `decode_lossy` run together on a real
            // training loop. It costs 4 forwards — `cadence_fires` fires once
            // on a 2-step run (at `step == num_iters`), and `sample_tokens: 1`
            // is what keeps each one to a single token.
            sample_every: 5,
            sample_tokens: 1,
            sample_temperature: 0.0,
        };
        (root, cfg)
    }

    /// Capstone: fixtures → `run()` → a `<run>/best` that reloads and
    /// reproduces the bpb it was selected on, plus a `run.json` that describes
    /// the run it actually performed. The phase-4 analogue of
    /// `train_saves_loadable_best_checkpoint_matching_reported_bpb`
    /// (`train_loop.rs`), and the only cheap proof the whole wiring holds
    /// together.
    #[test]
    fn run_saves_a_loadable_best_checkpoint_and_records_its_provenance() -> Result<()> {
        use serde_json::Value;

        let dev = Device::Cpu;
        let (_root, cfg) = capstone_fixture();

        run(&cfg, &dev, &DistCtx::single())?;

        // Exactly one run dir, holding both artifacts and the checkpoint.
        let mut runs: Vec<PathBuf> = std::fs::read_dir(&cfg.out)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        runs.sort();
        assert_eq!(runs.len(), 1, "one run dir per run");
        let run_dir = &runs[0];

        // The saved model reproduces the bpb it was selected on, re-scored on
        // the *same* val batches — obtained from `val_batches` rather than
        // rebuilt by hand, so the test cannot drift from what `run()` used.
        let tok = BpeTokenizer::from_file(&cfg.vocab)?;
        let token_bytes = tok.token_byte_lengths();
        let (val_snapshot, val_rows) = val_batches(&cfg, &tok, 64, &dev)?;
        assert_eq!(val_snapshot.len(), cfg.eval_steps);

        let (loaded, _vm, meta) = checkpoint::load(&run_dir.join("best"), &dev)?;
        let rescored = crate::eval::evaluate(&loaded, &val_snapshot, &token_bytes)?;
        assert_eq!(
            rescored.bpb, meta.val_bpb,
            "the reloaded model must reproduce the saved bpb exactly"
        );
        assert!(
            meta.val_bpb.is_finite(),
            "bpb {} is not finite",
            meta.val_bpb
        );

        let json: Value =
            serde_json::from_str(&std::fs::read_to_string(run_dir.join("run.json")).unwrap())
                .unwrap();
        assert_eq!(json["phase"], "sft");
        assert_eq!(json["num_iters"], 2);
        assert_eq!(json["grad_accum"], 1);
        assert_eq!(json["tokens_per_step"], cfg.total_batch); // global, as pretrain records it
        // Non-zero in the fixture on purpose: pinned at 0 this would still pass
        // if `RunMeta.sample_every` went back to a hardcoded literal.
        assert_eq!(json["sample_every"], 5);

        // The one place `init_lr_frac` becomes observable: dropping the
        // multiplication is a silent 25% LR error at the default 0.8.
        for (key, base) in [
            ("embedding_lr", cfg.lrs.embedding),
            ("unembedding_lr", cfg.lrs.unembedding),
            ("matrix_lr", cfg.lrs.matrix),
        ] {
            assert_eq!(
                json[key].as_f64().unwrap(),
                base * cfg.init_lr_frac,
                "{key} must record the post-init_lr_frac value"
            );
        }

        let sft = &json["sft"];
        assert_eq!(sft["seed"], cfg.seed);
        assert_eq!(sft["mmlu_epochs"], 0);
        assert_eq!(sft["val_rows"], val_rows);
        assert_eq!(sft["base_checkpoint"], cfg.checkpoint.display().to_string());
        assert!(sft["conversations"].as_u64().unwrap() >= 12);
        let scored = sft["scored_fraction"].as_f64().unwrap();
        assert!(
            scored > 0.0 && scored < 1.0,
            "scored fraction {scored} should be a real share of the pack"
        );

        // Training telemetry landed.
        assert!(
            std::fs::metadata(run_dir.join("metrics.jsonl"))
                .unwrap()
                .len()
                > 0,
            "metrics.jsonl should be non-empty"
        );
        Ok(())
    }

    /// The val snapshot is built at rank 0 / world 1, so it covers **every**
    /// packed val row: asking for `val_rows / device_batch` batches must
    /// succeed exactly.
    ///
    /// This is the assertion that pins the `view(0, 1, …)` rule. The snapshot
    /// is shared by every rank and `evaluate_shard_sums` strides it itself, so
    /// striding it here as well would double-stride — each rank scoring a slice
    /// of its own slice, leaving most rows unscored. That failure only shows up
    /// at world > 1, which no local test can reach; what is reachable is the
    /// row-coverage consequence, and a wrong `world` here breaks it.
    #[test]
    fn the_val_snapshot_covers_every_packed_val_row() -> Result<()> {
        let dev = Device::Cpu;
        let (_root, mut cfg) = capstone_fixture();

        // The cap is already clamped by supply at this fixture's size, so
        // raising eval_steps widens the budget without changing val_rows.
        let (_, val_rows) = val_batches(&cfg, &BpeTokenizer::from_file(&cfg.vocab)?, 64, &dev)?;
        assert!(val_rows >= 2 * cfg.device_batch, "fixture too small");

        cfg.eval_steps = val_rows / cfg.device_batch;
        let tok = BpeTokenizer::from_file(&cfg.vocab)?;
        let (batches, rows) = val_batches(&cfg, &tok, 64, &dev)?;
        assert_eq!(rows, val_rows, "the cap must not move with eval_steps here");
        assert_eq!(
            batches.len(),
            val_rows / cfg.device_batch,
            "the snapshot must reach every val row"
        );
        Ok(())
    }

    /// The retry's *success* path: the cap makes the mixture too small to
    /// serve, the rebuild at `f = 1.0` rescues the run.
    ///
    /// The other two paths are already covered — no retry needed (the capstone,
    /// `the_val_snapshot_covers_every_packed_val_row`) and retry-then-still-fail
    /// (`too_few_val_batches_fails_before_the_train_render`) — but this is the
    /// only one where the retry changes an outcome. Without it, reverting to
    /// the pre-fix behaviour would leave every other test green.
    ///
    /// Trigger: conversations shorter than `row_capacity / VAL_HEADROOM`. The
    /// byte vocab renders `data_dir`'s rows to ~15 tokens, so at `seq_len 1024`
    /// the capped budget of 32 conversations packs to under one micro-batch
    /// while all 200 pack to more than one.
    #[test]
    fn the_retry_rescues_a_val_mixture_the_cap_made_too_small() -> Result<()> {
        let dev = Device::Cpu;
        let dir = data_dir(200);
        let vocab = dir.path().join("vocab.txt");
        crate::test_support::write_byte_vocab(&vocab);
        let tok = BpeTokenizer::from_file(&vocab)?;

        let mut cfg = valid_config();
        cfg.data = dir.path().to_path_buf();
        cfg.vocab = vocab;
        cfg.device_batch = 2;
        cfg.eval_steps = 1;
        cfg.mmlu_epochs = 0;
        cfg.gsm8k_epochs = 0;
        let seq_len = 1024;

        let budget = VAL_HEADROOM * cfg.eval_steps * cfg.device_batch;
        let capped_rows = pack_val(&cfg, &tok, seq_len, budget)?.stats.rows;
        assert!(
            capped_rows / cfg.device_batch < cfg.eval_steps,
            "fixture does not reach the retry: the capped mixture already packs {capped_rows} rows"
        );

        let (batches, val_rows) = val_batches(&cfg, &tok, seq_len, &dev)?;
        assert_eq!(batches.len(), cfg.eval_steps);
        assert!(
            val_rows > capped_rows,
            "the retry must have uncapped the mixture: got {val_rows} rows, capped is {capped_rows}"
        );
        Ok(())
    }

    /// The horizon is the packed epoch's, and `--num-iters` may only shorten
    /// it: the mixture is packed once and never repeated, so asking for more
    /// steps than it holds is a mistake, not a wrap-around.
    #[test]
    fn num_iters_past_the_epoch_is_rejected() {
        let dev = Device::Cpu;
        let (_root, mut cfg) = capstone_fixture();
        cfg.num_iters = Some(10_000);

        let err =
            run(&cfg, &dev, &DistCtx::single()).expect_err("a horizon past the epoch must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("--num-iters 10000") && msg.contains("steps this epoch holds"),
            "unhelpful message: {msg}"
        );

        // The run dir is claimed at step 4b, well before this bail at step 8.
        // `RemoveIfEmpty` takes it back, so a retried invocation does not leave
        // `out-sft/` filling with empty directories that no run.json describes.
        assert_eq!(
            std::fs::read_dir(&cfg.out).unwrap().count(),
            0,
            "a failed run must not orphan its run dir in {}",
            cfg.out.display()
        );
    }

    /// A val mixture too small for the requested snapshot fails in seconds,
    /// before the train render — and says what to do about it.
    #[test]
    fn too_few_val_batches_fails_before_the_train_render() {
        let dev = Device::Cpu;
        let (_root, mut cfg) = capstone_fixture();
        cfg.eval_steps = 10_000;

        let err =
            run(&cfg, &dev, &DistCtx::single()).expect_err("an oversized eval snapshot must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("--eval-steps") && msg.contains("lower"),
            "unhelpful message: {msg}"
        );
    }

    /// `run()` reads the world size from `cfg.gpus` *and* from `dist`; a
    /// disagreement between them is silent everywhere else.
    ///
    /// `gpus: 2` under a single-process `DistCtx` passes every other check —
    /// the geometry divides through, the view serves every row — and trains at
    /// half the intended global batch while `run.json` claims two ranks. The
    /// guard runs before the geometry check, so the message must be this one
    /// and not a divisibility complaint.
    #[test]
    fn gpus_disagreeing_with_the_world_size_is_rejected() {
        let dev = Device::Cpu;
        let (_root, mut cfg) = capstone_fixture();
        cfg.gpus = 2;

        // `total_batch` is left at the fixture's value, which is *not* a
        // multiple of `gpus · device_batch · seq_len` at gpus 2. That is what
        // gives the assertion teeth: a guard placed after the geometry check
        // would report a divisibility failure for what is really a world-size
        // bug, and the test would still see an error.
        let err = run(&cfg, &dev, &DistCtx::single()).expect_err("a world-size mismatch must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("world size") && !msg.contains("multiple of"),
            "the world-size guard must run before the geometry check: {msg}"
        );
    }

    /// A vocabulary that is not the checkpoint's is caught up front. It would
    /// otherwise load fine and score garbage — the footgun `checkpoint::load`
    /// documents but deliberately does not check itself.
    #[test]
    fn a_vocab_that_disagrees_with_the_checkpoint_is_rejected() {
        let dev = Device::Cpu;
        let (root, mut cfg) = capstone_fixture();

        // A *valid* vocab that is simply not this checkpoint's: the byte vocab
        // plus one learned merge, so it is one token wider. (Dropping an entry
        // instead would trip the tokenizer's own 256-single-byte check and
        // never reach the cross-check under test.)
        let wider = root.path().join("wider-vocab.txt");
        let mut text = std::fs::read_to_string(&cfg.vocab).unwrap();
        text.push_str("YWI= 256\n"); // base64("ab")
        std::fs::write(&wider, text).unwrap();
        assert_eq!(
            BpeTokenizer::from_file(&wider).unwrap().vocab_size(),
            BpeTokenizer::from_file(&cfg.vocab).unwrap().vocab_size() + 1
        );
        cfg.vocab = wider;

        let err = run(&cfg, &dev, &DistCtx::single()).expect_err("a mismatched vocab must fail");
        assert!(
            err.to_string().contains("scores garbage"),
            "unhelpful message: {err}"
        );
    }
}
