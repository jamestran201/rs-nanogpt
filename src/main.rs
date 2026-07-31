use std::io;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rs_nanogpt::data::{BASE_URL, DataLoader, MAX_SHARD, Split, download_shards};
use rs_nanogpt::eval::tokenizer as tokenizer_eval;
use rs_nanogpt::metrics::{MetricsLogger, RunMeta, write_run_json};
use rs_nanogpt::model::{
    DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_NORM_EPS, DEFAULT_ROPE_BASE,
    DEFAULT_SEQUENCE_LEN, Gpt, GptConfig, compute_dtype, default_device,
};
use rs_nanogpt::tasks::download_sft_data;
use rs_nanogpt::tokenizer::{BpeTokenizer, BpeTokenizerTrainer};
use rs_nanogpt::train::dist::{self, ChildEnv};
use rs_nanogpt::train::{
    DEFAULT_FINAL_LR_FRAC, DEFAULT_WARMDOWN_RATIO, DEFAULT_WARMUP_STEPS, DistCtx, EvalContext,
    GroupLrs, TrainConfig, train,
};

#[derive(Parser)]
#[command(name = "rs-nanogpt", version, about = "nanoGPT-style training tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a BPE tokenizer from a text corpus.
    TrainTokenizer {
        /// Directory containing the corpus parquet files.
        #[arg(long)]
        corpus: PathBuf,
        /// Path where the tiktoken-format vocabulary will be written.
        #[arg(long)]
        output: PathBuf,
        /// Target vocabulary size (must be at least 256).
        #[arg(long, default_value_t = 512)]
        vocab_size: usize,
        /// Maximum number of bytes to read from the corpus.
        #[arg(long)]
        max_chars: usize,
        /// Maximum bytes per document. Documents longer than this are
        /// truncated (at a UTF-8 char boundary) so a few unusually long
        /// documents can't dominate BPE pair statistics.
        #[arg(long, default_value_t = 10_000)]
        doc_cap: usize,
    },
    /// Evaluate a trained tokenizer on a small set of text fixtures.
    EvalTokenizer {
        /// Path to a tiktoken-format vocabulary file.
        #[arg(long)]
        vocab: PathBuf,
    },
    /// Pretrain a GPT model.
    Pretrain(PretrainArgs),
    /// Download pretraining dataset shards into a local directory.
    DownloadData(DownloadArgs),
    /// Download the SFT datasets (SmolTalk, MMLU, GSM8K, identity conversations).
    DownloadSftData(DownloadSftArgs),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::TrainTokenizer {
            corpus,
            output,
            vocab_size,
            max_chars,
            doc_cap,
        } => {
            let trainer = BpeTokenizerTrainer::new(corpus, max_chars, doc_cap);
            trainer.train(output, vocab_size)?;
        }
        Command::EvalTokenizer { vocab } => {
            let tok = BpeTokenizer::from_file(vocab)?;
            let results = tokenizer_eval::eval_fixtures(&tok);
            tokenizer_eval::print_table(&results);
            if !results.iter().all(|r| r.round_trip_ok) {
                process::exit(1);
            }
        }
        Command::Pretrain(args) => {
            // Three roles share this subcommand: the launcher parent (--gpus N
            // with a clean env, spawns one child per GPU and supervises), a
            // spawned rank (env set by the parent), and the plain
            // single-process run (everything else).
            let result: Result<(), Box<dyn std::error::Error>> = match dist::child_env() {
                Err(e) => Err(e.into()),
                Ok(None) if args.gpus > 1 => run_launcher(&args),
                Ok(child) => run_pretrain(args, child),
            };
            if let Err(err) = result {
                eprintln!("pretrain failed: {err}");
                process::exit(1);
            }
        }
        Command::DownloadData(args) => {
            if let Err(err) = run_download(args) {
                eprintln!("download failed: {err}");
                process::exit(1);
            }
        }
        Command::DownloadSftData(args) => {
            if let Err(err) = run_download_sft(args) {
                eprintln!("download failed: {err}");
                process::exit(1);
            }
        }
    }
    Ok(())
}

#[derive(clap::Args)]
struct PretrainArgs {
    /// Directory of parquet shards to train on (last shard = val split).
    #[arg(long)]
    data: PathBuf,
    /// Tiktoken-format vocabulary file. Loaded as the tokenizer and used to
    /// size the model's embedding/lm_head, so no separate --vocab-size flag.
    #[arg(long)]
    vocab: PathBuf,
    /// Maximum context length (tokens per training example).
    #[arg(long, default_value_t = DEFAULT_SEQUENCE_LEN)]
    sequence_len: usize,
    /// Number of transformer blocks.
    #[arg(long, default_value_t = DEFAULT_N_LAYER)]
    n_layer: usize,
    /// Number of attention heads.
    #[arg(long, default_value_t = DEFAULT_N_HEAD)]
    n_head: usize,
    /// Residual-stream width. Must be divisible by n_head.
    #[arg(long, default_value_t = DEFAULT_N_EMBD)]
    n_embd: usize,
    /// RoPE frequency base (larger base = longer effective context).
    #[arg(long, default_value_t = DEFAULT_ROPE_BASE)]
    rope_base: f32,
    /// RMSNorm epsilon.
    #[arg(long, default_value_t = DEFAULT_NORM_EPS)]
    norm_eps: f32,
    /// Number of optimizer steps (training horizon).
    #[arg(long, default_value_t = 5000)]
    num_iters: usize,
    /// Rows per forward pass (B). Memory-limited; reduce if you OOM.
    #[arg(long, default_value_t = 32)]
    device_batch: usize,
    /// Global tokens per optimizer step (summed over all GPUs). Must be a
    /// multiple of gpus*device_batch*seq_len; the quotient is the per-rank
    /// gradient-accumulation steps.
    #[arg(long, default_value_t = 16384)]
    total_batch: usize,
    /// Number of GPUs (data-parallel ranks). With N > 1 this process becomes
    /// a launcher: it re-executes itself once per GPU and supervises the
    /// ranks. Requires a build with --features nccl and one visible CUDA
    /// device per rank.
    #[arg(long, default_value_t = 1)]
    gpus: usize,
    /// AdamW LR for the token embedding (wte).
    #[arg(long, default_value_t = 0.2)]
    embedding_lr: f64,
    /// AdamW LR for the unembedding (lm_head).
    #[arg(long, default_value_t = 0.004)]
    unembedding_lr: f64,
    /// AdamW LR for the block matrices. Ballpark — sweep on the smoke run.
    #[arg(long, default_value_t = 0.003)]
    matrix_lr: f64,
    /// Linear LR warmup steps.
    #[arg(long, default_value_t = DEFAULT_WARMUP_STEPS)]
    warmup_steps: usize,
    /// Fraction of the run spent in linear LR warmdown.
    #[arg(long, default_value_t = DEFAULT_WARMDOWN_RATIO)]
    warmdown_ratio: f64,
    /// Final LR as a fraction of base LR (warmdown floor).
    #[arg(long, default_value_t = DEFAULT_FINAL_LR_FRAC)]
    final_lr_frac: f64,
    /// Log loss every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Compute val loss/bpb every N steps (0 disables eval and checkpointing).
    #[arg(long, default_value_t = 250)]
    eval_every: usize,
    /// Number of val batches to snapshot for eval.
    #[arg(long, default_value_t = 20)]
    eval_steps: usize,
    /// Sample from the model every N steps (0 disables).
    #[arg(long, default_value_t = 0)]
    sample_every: usize,
    /// Tokens to generate per sample.
    #[arg(long, default_value_t = 64)]
    sample_tokens: usize,
    /// Sampling temperature; 0 = greedy.
    #[arg(long, default_value_t = 0.0)]
    sample_temperature: f64,
    /// Runs root; each run writes to <out>/<unix-timestamp>/ (run.json,
    /// metrics.jsonl, best/), so runs never overwrite each other.
    #[arg(long, default_value = "out")]
    out: PathBuf,
}

fn validate_pretrain_args(args: &PretrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.device_batch == 0 {
        return Err("--device-batch must be >= 1".into());
    }
    // Also a GptConfig invariant; checked here so `micro` below is nonzero.
    if args.sequence_len == 0 {
        return Err("--sequence-len must be >= 1".into());
    }
    if args.num_iters == 0 {
        return Err("--num-iters must be >= 1".into());
    }
    if args.log_every == 0 {
        return Err("--log-every must be >= 1".into());
    }
    if !(0.0..=1.0).contains(&args.warmdown_ratio) {
        return Err(format!(
            "--warmdown-ratio must be in [0, 1], got {}",
            args.warmdown_ratio
        )
        .into());
    }
    if !(0.0..=1.0).contains(&args.final_lr_frac) {
        return Err(format!(
            "--final-lr-frac must be in [0, 1], got {}",
            args.final_lr_frac
        )
        .into());
    }

    if args.gpus == 0 {
        return Err("--gpus must be >= 1".into());
    }

    // Gradient accumulation: the global total_batch is reached by whole
    // micro-batches on every rank: grad_accum = total_batch / (gpus · micro).
    let micro = args.device_batch * args.sequence_len;
    let per_step = args.gpus * micro;
    if !args.total_batch.is_multiple_of(per_step) {
        return Err(format!(
            "--total-batch ({}) must be a multiple of gpus*device_batch*seq_len ({per_step})",
            args.total_batch
        )
        .into());
    }
    let grad_accum = args.total_batch / per_step;
    if grad_accum == 0 {
        return Err(format!(
            "--total-batch ({}) must be at least gpus*device_batch*seq_len ({per_step})",
            args.total_batch
        )
        .into());
    }

    if args.eval_every > 0 && args.eval_steps == 0 {
        return Err("--eval-steps must be >= 1 when --eval-every > 0".into());
    }
    if args.sample_every > 0 && args.sample_tokens == 0 {
        return Err("--sample-tokens must be >= 1 when --sample-every > 0".into());
    }
    if args.sample_temperature < 0.0 {
        return Err(format!(
            "--sample-temperature must be >= 0, got {}",
            args.sample_temperature
        )
        .into());
    }
    Ok(())
}

fn unique_run_dir(root: &Path, id: &str) -> io::Result<PathBuf> {
    std::fs::create_dir_all(root)?;
    let names = std::iter::once(String::new()).chain((2u32..).map(|n| format!("-{n}")));
    for suffix in names {
        let candidate = root.join(format!("{id}{suffix}"));
        match std::fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    unreachable!("u32 run-dir suffixes exhausted")
}

/// The `--gpus N` parent: generate the NCCL rendezvous id, re-execute this
/// binary once per GPU with the rank identity in env vars, and supervise.
fn run_launcher(args: &PretrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Fail fast on bad args before N processes get spawned to hit the same error.
    validate_pretrain_args(args)?;
    #[cfg(not(feature = "nccl"))]
    {
        Err(format!(
            "--gpus {} needs multi-GPU support; rebuild with --features nccl",
            args.gpus
        )
        .into())
    }
    #[cfg(feature = "nccl")]
    {
        let id_hex = dist::generate_nccl_id_hex()?;
        let exe = std::env::current_exe()?;
        let argv: Vec<String> = std::env::args().skip(1).collect();
        println!("spawning {} ranks (one process per GPU)", args.gpus);
        let mut children = Vec::with_capacity(args.gpus);
        for rank in 0..args.gpus {
            let child = process::Command::new(&exe)
                .args(&argv)
                .env(dist::ENV_RANK, rank.to_string())
                .env(dist::ENV_WORLD_SIZE, args.gpus.to_string())
                .env(dist::ENV_NCCL_ID, &id_hex)
                .spawn()
                .map_err(|e| format!("failed to spawn rank {rank}: {e}"))?;
            children.push(child);
        }
        supervise_ranks(children)
    }
}

/// Watchdog over the spawned ranks. Polls with `try_wait` so the *first*
/// abnormal exit is noticed while the other ranks are still running — a dead
/// NCCL peer leaves the survivors hung in a collective forever, so they are
/// killed rather than awaited. (A sequential `wait()` loop cannot do this:
/// blocked on rank 0, it would never notice rank 5 dying.)
#[cfg(feature = "nccl")]
fn supervise_ranks(mut children: Vec<process::Child>) -> Result<(), Box<dyn std::error::Error>> {
    let n = children.len();
    let mut running = n;
    let mut exited = vec![false; n];
    loop {
        for i in 0..n {
            if exited[i] {
                continue;
            }
            let Some(status) = children[i].try_wait()? else {
                continue;
            };
            exited[i] = true;
            running -= 1;
            if !status.success() {
                for (j, child) in children.iter_mut().enumerate() {
                    if !exited[j] {
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
                return Err(format!(
                    "rank {i} exited abnormally ({status}); killed the other ranks"
                )
                .into());
            }
        }
        if running == 0 {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

fn run_pretrain(
    args: PretrainArgs,
    child: Option<ChildEnv>,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_pretrain_args(&args)?;

    // Single process: today's device selection, no communicator. Spawned
    // rank: bind to this rank's GPU and join the NCCL world (nccl builds
    // only — the launcher refuses to spawn without the feature, so reaching
    // here without it means hand-set env vars).
    let (device, dist) = match child {
        None => (default_device()?, DistCtx::single()),
        #[cfg(feature = "nccl")]
        Some(env) => {
            if env.world_size != args.gpus {
                return Err(format!(
                    "{} ({}) disagrees with --gpus ({}); the env vars are launcher-internal",
                    dist::ENV_WORLD_SIZE,
                    env.world_size,
                    args.gpus
                )
                .into());
            }
            let device = candle_core::Device::new_cuda(env.rank)?;
            let dist = DistCtx::new_nccl(&device, &env)?;
            (device, dist)
        }
        #[cfg(not(feature = "nccl"))]
        Some(env) => {
            return Err(format!(
                "{} is set (rank {}) but this binary was built without the nccl feature",
                dist::ENV_RANK,
                env.rank
            )
            .into());
        }
    };

    let micro = args.device_batch * args.sequence_len;
    let grad_accum = args.total_batch / (args.gpus * micro);
    let train_cfg = TrainConfig {
        num_iters: args.num_iters,
        grad_accum,
        // Per-rank tokens: total_batch / gpus = grad_accum · device_batch ·
        // sequence_len (divisibility validated above). The cross-rank Avg of
        // the gradient reduce supplies the remaining 1/gpus factor.
        tokens_per_step: args.total_batch / args.gpus,
        // The pretraining packer scores every token — no ignore_index targets.
        targets_may_be_ignored: false,
        lrs: GroupLrs {
            embedding: args.embedding_lr,
            unembedding: args.unembedding_lr,
            matrix: args.matrix_lr,
        },
        warmup_steps: args.warmup_steps,
        warmdown_ratio: args.warmdown_ratio,
        final_lr_frac: args.final_lr_frac,
        log_every: args.log_every,
    };

    let tokenizer = BpeTokenizer::from_file(&args.vocab)?;
    let config = GptConfig {
        vocab_size: tokenizer.vocab_size(),
        sequence_len: args.sequence_len,
        n_layer: args.n_layer,
        n_head: args.n_head,
        n_embd: args.n_embd,
        rope_base: args.rope_base,
        norm_eps: args.norm_eps,
    };
    config.validate()?;
    if dist.is_master() {
        print_config_summary(&config);
    }

    // Train docs stride by rank (disjoint per-GPU streams); the val split is
    // NOT strided — every rank snapshots the same batches and eval shards them.
    let mut loader = DataLoader::open_strided(
        &args.data,
        Split::Train,
        &tokenizer,
        args.device_batch,
        config.sequence_len,
        dist.rank(),
        dist.world_size(),
    )?;

    let mut val_loader = DataLoader::open(
        &args.data,
        Split::Val,
        &tokenizer,
        args.device_batch,
        config.sequence_len,
    )?;
    let val_batches = val_loader.take_batches(args.eval_steps, &device)?;
    let token_bytes = tokenizer.token_byte_lengths();

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Gpt::new(config, vb)?;
    let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    if dist.is_master() {
        println!(
            "\nmodel built on {device:?} | compute dtype {:?} (fp32 masters) | {n_params} parameters",
            compute_dtype(&device)
        );
        println!(
            "\ntraining: {} iters, total_batch {} tokens, gpus {}, device_batch {}, grad_accum {}",
            args.num_iters, args.total_batch, args.gpus, args.device_batch, grad_accum
        );
    }
    let started_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // Run dir, run.json, and metrics.jsonl are master-only side effects; the
    // other ranks get a discard sink and a path that train() never writes to
    // (checkpointing is master-gated). This also avoids per-rank timestamp
    // races into different run dirs.
    let (run_dir, metrics) = if dist.is_master() {
        let run_dir = unique_run_dir(&args.out, &started_at_unix.to_string())?;
        println!("run dir: {}", run_dir.display());

        let run_meta = RunMeta {
            device: format!("{device:?}"),
            dtype: "f32",
            started_at_unix,
            n_params,
            vocab_size: config.vocab_size,
            sequence_len: config.sequence_len,
            n_layer: config.n_layer,
            n_head: config.n_head,
            n_embd: config.n_embd,
            rope_base: config.rope_base,
            norm_eps: config.norm_eps,
            num_iters: args.num_iters,
            device_batch: args.device_batch,
            total_batch: args.total_batch,
            world_size: args.gpus,
            grad_accum,
            // Global tokens per optimizer step (== total_batch ==
            // world_size * grad_accum * device_batch * seq_len), so run.json
            // stays comparable across GPU counts; TrainConfig's field is the
            // per-rank share.
            tokens_per_step: args.total_batch,
            embedding_lr: args.embedding_lr,
            unembedding_lr: args.unembedding_lr,
            matrix_lr: args.matrix_lr,
            warmup_steps: args.warmup_steps,
            warmdown_ratio: args.warmdown_ratio,
            final_lr_frac: args.final_lr_frac,
            log_every: args.log_every,
            eval_every: args.eval_every,
            eval_steps: args.eval_steps,
            sample_every: args.sample_every,
        };
        write_run_json(&run_dir.join("run.json"), &run_meta)?;
        let metrics = MetricsLogger::create(&run_dir.join("metrics.jsonl"))?;
        (run_dir, metrics)
    } else {
        (args.out.clone(), MetricsLogger::null())
    };

    let eval = EvalContext {
        val_batches: &val_batches,
        tokenizer: &tokenizer,
        token_bytes: &token_bytes,
        ckpt_root: &run_dir,
        metrics: &metrics,
        eval_every: args.eval_every,
        sample_every: args.sample_every,
        sample_tokens: args.sample_tokens,
        sample_temperature: args.sample_temperature,
    };
    train(
        &model,
        &varmap,
        &mut loader,
        &train_cfg,
        &eval,
        &dist,
        &device,
    )?;
    Ok(())
}

fn print_config_summary(config: &GptConfig) {
    println!("GPT config:");
    println!("  vocab_size   {}", config.vocab_size);
    println!("  sequence_len {}", config.sequence_len);
    println!("  n_layer      {}", config.n_layer);
    println!("  n_head       {}", config.n_head);
    println!("  n_embd       {}", config.n_embd);
    println!("  head_dim     {}  (n_embd / n_head)", config.head_dim());
    println!("  rope_base    {}", config.rope_base);
    println!("  norm_eps     {}", config.norm_eps);
}

#[derive(clap::Args)]
struct DownloadArgs {
    /// Directory shards are written to (created if missing).
    #[arg(long, default_value = "data")]
    out: PathBuf,
    /// First shard index to download.
    #[arg(long, default_value_t = 0)]
    start: usize,
    /// Number of shards to download in total, starting at --start.
    #[arg(long)]
    num: usize,
    /// Number of parallel downloads.
    #[arg(long, default_value_t = 4)]
    workers: usize,
    /// Validation shard to also fetch. It pins the val split (sorts last, so the
    /// loader holds it out and keeps the whole --start/--num range as train).
    /// Pass "none" to download only the range.
    #[arg(long, default_value = "6542")]
    val_shard: String,
    /// Base URL the shards are fetched from.
    #[arg(long, default_value = BASE_URL)]
    base_url: String,
}

fn parse_val_shard(spec: &str) -> Result<Option<usize>, String> {
    let s = spec.trim();
    if s.eq_ignore_ascii_case("none") || s.eq_ignore_ascii_case("skip") {
        return Ok(None);
    }
    let v: usize = s
        .parse()
        .map_err(|_| format!("--val-shard must be a shard index or \"none\", got {spec:?}"))?;
    if v > MAX_SHARD {
        return Err(format!(
            "--val-shard {v} exceeds the last shard index ({MAX_SHARD})"
        ));
    }
    Ok(Some(v))
}

fn validate_download_args(
    args: &DownloadArgs,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    if args.num == 0 {
        return Err("--num must be >= 1".into());
    }
    if args.workers == 0 {
        return Err("--workers must be >= 1".into());
    }
    // Last index in the contiguous range; saturating guards against usize
    // overflow on an absurd --num before the range check can reject it.
    let last = args.start.saturating_add(args.num - 1);
    if last > MAX_SHARD {
        return Err(format!(
            "shard range ends at {last}, past the last shard index ({MAX_SHARD}); reduce --start/--num"
        )
        .into());
    }
    let val_shard = parse_val_shard(&args.val_shard)?;
    Ok(val_shard)
}

fn run_download(args: DownloadArgs) -> Result<(), Box<dyn std::error::Error>> {
    let val_shard = validate_download_args(&args)?;
    let summary = download_shards(
        &args.out,
        args.start,
        args.num,
        val_shard,
        args.workers,
        &args.base_url,
    )?;
    if summary.failed > 0 {
        return Err(format!(
            "{} of {} shard(s) failed to download",
            summary.failed, summary.requested
        )
        .into());
    }
    Ok(())
}

#[derive(clap::Args)]
struct DownloadSftArgs {
    /// Directory the SFT data files are written to (created if missing).
    #[arg(long, default_value = "sft-data")]
    out: PathBuf,
    /// Number of parallel downloads.
    #[arg(long, default_value_t = 4)]
    workers: usize,
}

/// The file set is a fixed manifest (`src/tasks/download.rs`), so unlike
/// `download-data` there are no range flags.
fn run_download_sft(args: DownloadSftArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.workers == 0 {
        return Err("--workers must be >= 1".into());
    }
    let summary = download_sft_data(&args.out, args.workers)?;
    if summary.failed > 0 {
        return Err(format!(
            "{} of {} file(s) failed to download",
            summary.failed, summary.requested
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First call claims `root/<id>`; each subsequent call with the same id walks
    /// to the next `-N` suffix, and every returned path is a real, distinct dir.
    #[test]
    fn unique_run_dir_walks_suffixes_on_collision() {
        let root = tempfile::tempdir().unwrap();
        let a = unique_run_dir(root.path(), "run").unwrap();
        let b = unique_run_dir(root.path(), "run").unwrap();
        let c = unique_run_dir(root.path(), "run").unwrap();
        assert_eq!(a, root.path().join("run"));
        assert_eq!(b, root.path().join("run-2"));
        assert_eq!(c, root.path().join("run-3"));
        assert!(a.is_dir() && b.is_dir() && c.is_dir());
    }

    /// The root is created on demand, so a not-yet-existing `--out` just works.
    #[test]
    fn unique_run_dir_creates_missing_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("does/not/exist/yet");
        let dir = unique_run_dir(&root, "1752192000").unwrap();
        assert_eq!(dir, root.join("1752192000"));
        assert!(dir.is_dir());
    }

    /// A known-good set of args; tests mutate one field to probe a single rule.
    /// `data`/`vocab` are never touched by `validate_pretrain_args`, so dummy paths are fine.
    fn valid_args() -> PretrainArgs {
        PretrainArgs {
            data: PathBuf::from("unused"),
            vocab: PathBuf::from("unused"),
            sequence_len: 512,
            n_layer: 6,
            n_head: 6,
            n_embd: 384,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
            num_iters: 5000,
            device_batch: 32,
            total_batch: 16384, // == device_batch * sequence_len => grad_accum 1
            gpus: 1,
            embedding_lr: 0.2,
            unembedding_lr: 0.004,
            matrix_lr: 0.003,
            warmup_steps: 40,
            warmdown_ratio: 0.65,
            final_lr_frac: 0.05,
            log_every: 10,
            eval_every: 250,
            eval_steps: 20,
            sample_every: 2000,
            sample_tokens: 64,
            sample_temperature: 0.0,
            out: PathBuf::from("unused"),
        }
    }

    /// Mutating one field of `valid_args` must make validation fail.
    fn rejects(mutate: impl FnOnce(&mut PretrainArgs)) {
        let mut a = valid_args();
        mutate(&mut a);
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn accepts_valid_args() {
        assert!(validate_pretrain_args(&valid_args()).is_ok());
    }

    #[test]
    fn accepts_total_batch_multiple_of_micro() {
        let mut a = valid_args();
        a.total_batch = 2 * 16384; // 2 micro-batches per step
        assert!(validate_pretrain_args(&a).is_ok());
    }

    #[test]
    fn rejects_zero_device_batch() {
        rejects(|a| a.device_batch = 0);
    }

    #[test]
    fn rejects_zero_sequence_len() {
        rejects(|a| a.sequence_len = 0);
    }

    #[test]
    fn rejects_zero_num_iters() {
        rejects(|a| a.num_iters = 0);
    }

    #[test]
    fn rejects_zero_log_every() {
        rejects(|a| a.log_every = 0);
    }

    #[test]
    fn rejects_out_of_range_warmdown_ratio() {
        rejects(|a| a.warmdown_ratio = 1.5);
        rejects(|a| a.warmdown_ratio = -0.1);
    }

    #[test]
    fn rejects_out_of_range_final_lr_frac() {
        rejects(|a| a.final_lr_frac = 2.0);
    }

    #[test]
    fn rejects_total_batch_not_multiple_of_micro() {
        rejects(|a| a.total_batch = 16384 + 1);
    }

    #[test]
    fn rejects_zero_gpus() {
        rejects(|a| a.gpus = 0);
    }

    #[test]
    fn total_batch_divisibility_includes_gpus() {
        // One micro-batch per rank per step is fine...
        let mut a = valid_args();
        a.gpus = 2;
        a.total_batch = 2 * 16384;
        assert!(validate_pretrain_args(&a).is_ok());
        // ...but a total_batch only one rank's micro-batches can fill is not.
        rejects(|a| a.gpus = 2); // total_batch stays 16384 = 1 * micro
    }

    #[test]
    fn rejects_zero_total_batch() {
        // Zero is a multiple of micro, so it reaches the grad_accum == 0 check.
        rejects(|a| a.total_batch = 0);
    }

    #[test]
    fn rejects_zero_eval_steps_when_eval_enabled() {
        rejects(|a| a.eval_steps = 0);
        // ...but zero eval_steps is fine when eval is disabled entirely.
        let mut a = valid_args();
        a.eval_steps = 0;
        a.eval_every = 0;
        assert!(validate_pretrain_args(&a).is_ok());
    }

    #[test]
    fn rejects_zero_sample_tokens_when_sampling_enabled() {
        rejects(|a| a.sample_tokens = 0);
        // ...but zero sample_tokens is fine when sampling is disabled.
        let mut a = valid_args();
        a.sample_tokens = 0;
        a.sample_every = 0;
        assert!(validate_pretrain_args(&a).is_ok());
    }

    #[test]
    fn rejects_negative_sample_temperature() {
        rejects(|a| a.sample_temperature = -0.1);
    }

    /// A known-good set of download args; tests mutate one field to probe a rule.
    fn valid_download_args() -> DownloadArgs {
        DownloadArgs {
            out: PathBuf::from("data"),
            start: 0,
            num: 170,
            workers: 4,
            val_shard: "6542".to_string(),
            base_url: BASE_URL.to_string(),
        }
    }

    #[test]
    fn accepts_valid_download_args() {
        assert_eq!(
            validate_download_args(&valid_download_args()).unwrap(),
            Some(6542)
        );
    }

    #[test]
    fn download_val_shard_none_parses_to_none() {
        let mut a = valid_download_args();
        a.val_shard = "none".to_string();
        assert_eq!(validate_download_args(&a).unwrap(), None);
    }

    #[test]
    fn rejects_zero_num() {
        let mut a = valid_download_args();
        a.num = 0;
        assert!(validate_download_args(&a).is_err());
    }

    #[test]
    fn rejects_zero_download_workers() {
        let mut a = valid_download_args();
        a.workers = 0;
        assert!(validate_download_args(&a).is_err());
    }

    #[test]
    fn rejects_range_past_last_shard() {
        // start at the last shard with num 2 => range ends one past the dataset.
        let mut a = valid_download_args();
        a.start = MAX_SHARD;
        a.num = 2;
        assert!(validate_download_args(&a).is_err());
        // ...but exactly the last shard alone is fine.
        let mut b = valid_download_args();
        b.start = MAX_SHARD;
        b.num = 1;
        assert!(validate_download_args(&b).is_ok());
    }

    #[test]
    fn rejects_val_shard_past_last() {
        let mut a = valid_download_args();
        a.val_shard = (MAX_SHARD + 1).to_string();
        assert!(validate_download_args(&a).is_err());
    }

    #[test]
    fn rejects_unparseable_val_shard() {
        let mut a = valid_download_args();
        a.val_shard = "abc".to_string();
        assert!(validate_download_args(&a).is_err());
    }
}
