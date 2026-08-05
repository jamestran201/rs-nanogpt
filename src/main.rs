use std::path::PathBuf;
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rs_nanogpt::checkpoint;
use rs_nanogpt::data::{BASE_URL, DataLoader, MAX_SHARD, Split, download_shards};
use rs_nanogpt::eval::tokenizer as tokenizer_eval;
use rs_nanogpt::metrics::{MetricsLogger, RunMeta, unique_run_dir, write_run_json};
use rs_nanogpt::model::{
    DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_NORM_EPS, DEFAULT_ROPE_BASE,
    DEFAULT_SEQUENCE_LEN, Gpt, GptConfig, compute_dtype, default_device,
};
use rs_nanogpt::sft::{self, SftConfig};
use rs_nanogpt::tasks::download_sft_data;
use rs_nanogpt::tokenizer::{BpeTokenizer, BpeTokenizerTrainer};
use rs_nanogpt::train::dist::{self, ChildEnv};
use rs_nanogpt::train::{
    DEFAULT_FINAL_LR_FRAC, DEFAULT_WARMDOWN_RATIO, DEFAULT_WARMUP_STEPS, DistCtx, EvalContext,
    GroupLrs, GroupWeightDecay, TrainConfig, train, validate_batch_geometry,
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
    /// Chat-finetune a pretrained checkpoint on the SFT mixture.
    Sft(SftArgs),
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
                Ok(None) if args.gpus > 1 => {
                    // Validate before spawning, so N processes do not start
                    // only to hit the same bad flag.
                    validate_pretrain_args(&args).and_then(|()| run_launcher(args.gpus))
                }
                Ok(child) => run_pretrain(args, child),
            };
            if let Err(err) = result {
                eprintln!("pretrain failed: {err}");
                process::exit(1);
            }
        }
        Command::Sft(args) => {
            // The same three roles as Pretrain: launcher parent, spawned rank,
            // plain single-process run.
            let result: Result<(), Box<dyn std::error::Error>> = match dist::child_env() {
                Err(e) => Err(e.into()),
                Ok(None) if args.gpus > 1 => launch_sft(args),
                Ok(child) => run_sft(args, child),
            };
            if let Err(err) = result {
                eprintln!("sft failed: {err}");
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

    // The divisibility math, shared with `sft` so the two cannot drift. The
    // zero checks above are now redundant with its own, but they carry
    // flag-specific messages, so they stay.
    validate_batch_geometry(
        args.gpus,
        args.device_batch,
        args.sequence_len,
        args.total_batch,
    )?;

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

/// The `--gpus N` parent: generate the NCCL rendezvous id, re-execute this
/// binary once per GPU with the rank identity in env vars, and supervise.
///
/// The caller validates first — spawning N processes only for all of them to
/// hit the same bad flag is the failure this ordering avoids — and the two
/// subcommands validate differently (SFT has to read the checkpoint meta for
/// its sequence length), which is why this takes only `gpus`.
fn run_launcher(gpus: usize) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "nccl"))]
    {
        Err(format!("--gpus {gpus} needs multi-GPU support; rebuild with --features nccl").into())
    }
    #[cfg(feature = "nccl")]
    {
        let id_hex = dist::generate_nccl_id_hex()?;
        let exe = std::env::current_exe()?;
        let argv: Vec<String> = std::env::args().skip(1).collect();
        println!("spawning {gpus} ranks (one process per GPU)");
        let mut children = Vec::with_capacity(gpus);
        for rank in 0..gpus {
            let child = process::Command::new(&exe)
                .args(&argv)
                .env(dist::ENV_RANK, rank.to_string())
                .env(dist::ENV_WORLD_SIZE, gpus.to_string())
                .env(dist::ENV_NCCL_ID, &id_hex)
                .spawn()
                .map_err(|e| format!("failed to spawn rank {rank}: {e}"))?;
            children.push(child);
        }
        supervise_ranks(children)
    }
}

/// The compute device and distributed context for one process.
///
/// Single process: today's device selection, no communicator. Spawned rank:
/// bind to this rank's GPU and join the NCCL world (nccl builds only — the
/// launcher refuses to spawn without the feature, so reaching the `Some` arm
/// without it means hand-set env vars).
fn init_device(
    child: Option<ChildEnv>,
    gpus: usize,
) -> Result<(candle_core::Device, DistCtx), Box<dyn std::error::Error>> {
    match child {
        None => Ok((default_device()?, DistCtx::single())),
        #[cfg(feature = "nccl")]
        Some(env) => {
            if env.world_size != gpus {
                return Err(format!(
                    "{} ({}) disagrees with --gpus ({gpus}); the env vars are launcher-internal",
                    dist::ENV_WORLD_SIZE,
                    env.world_size,
                )
                .into());
            }
            let device = candle_core::Device::new_cuda(env.rank)?;
            let dist = DistCtx::new_nccl(&device, &env)?;
            Ok((device, dist))
        }
        #[cfg(not(feature = "nccl"))]
        Some(env) => {
            let _ = gpus;
            Err(format!(
                "{} is set (rank {}) but this binary was built without the nccl feature",
                dist::ENV_RANK,
                env.rank
            )
            .into())
        }
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
    let (device, dist) = init_device(child, args.gpus)?;

    let grad_accum = validate_batch_geometry(
        args.gpus,
        args.device_batch,
        args.sequence_len,
        args.total_batch,
    )?;
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
        // Pretraining keeps the decays that were hardcoded before the knob
        // existed; only SFT deviates.
        weight_decays: GroupWeightDecay::default(),
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
            phase: "pretrain",
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
            sft: None,
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

#[derive(clap::Args)]
struct SftArgs {
    /// Pretrained checkpoint to finetune, e.g. out-d24/<run>/best.
    #[arg(long)]
    checkpoint: PathBuf,
    /// Directory holding the downloaded SFT datasets (run download-sft-data).
    #[arg(long)]
    data: PathBuf,
    /// Tiktoken-format vocabulary. Must be the one the checkpoint was
    /// pretrained with — a mismatch loads fine and scores garbage.
    #[arg(long)]
    vocab: PathBuf,
    /// AdamW LR for the token embedding (wte), before --init-lr-frac. Required:
    /// pass the value your pretraining run used.
    #[arg(long)]
    embedding_lr: f64,
    /// AdamW LR for the unembedding (lm_head), before --init-lr-frac.
    #[arg(long)]
    unembedding_lr: f64,
    /// AdamW LR for the block matrices, before --init-lr-frac.
    #[arg(long)]
    matrix_lr: f64,
    /// Runs root; each run writes to <out>/<unix-timestamp>/ (run.json,
    /// metrics.jsonl, best/).
    #[arg(long, default_value = "out-sft")]
    out: PathBuf,
    /// Number of GPUs (data-parallel ranks). With N > 1 this process becomes a
    /// launcher, as `pretrain --gpus N` does.
    #[arg(long, default_value_t = 1)]
    gpus: usize,
    /// Rows per forward pass (B). Memory-limited; reduce if you OOM.
    #[arg(long, default_value_t = 8)]
    device_batch: usize,
    /// Global tokens per optimizer step (summed over all GPUs). Must be a
    /// multiple of gpus*device_batch*seq_len, where seq_len comes from the
    /// checkpoint.
    #[arg(long, default_value_t = 524_288)]
    total_batch: usize,
    /// Optimizer steps. Omit to train exactly one epoch over the packed
    /// mixture; a value may only shorten that, never extend it.
    #[arg(long)]
    num_iters: Option<usize>,
    /// Scales all three LRs at the start of the finetune (nanochat's SFT
    /// scaling). run.json records the post-scaling values.
    #[arg(long, default_value_t = 0.8)]
    init_lr_frac: f64,
    /// Linear LR warmup steps.
    #[arg(long, default_value_t = 0)]
    warmup_steps: usize,
    /// Fraction of the run spent in linear LR warmdown.
    #[arg(long, default_value_t = 0.5)]
    warmdown_ratio: f64,
    /// Final LR as a fraction of base LR (warmdown floor).
    #[arg(long, default_value_t = 0.0)]
    final_lr_frac: f64,
    /// Log loss every N steps.
    #[arg(long, default_value_t = 10)]
    log_every: usize,
    /// Compute val loss/bpb every N steps. Must be >= 1: unlike pretrain, the
    /// best-bpb checkpoint is this run's only output.
    #[arg(long, default_value_t = 200)]
    eval_every: usize,
    /// Val micro-batches to snapshot. Also sizes the val mixture's cap.
    #[arg(long, default_value_t = 100)]
    eval_steps: usize,
    /// Copies of MMLU auxiliary_train in the mixture (0 omits it entirely).
    #[arg(long, default_value_t = 3)]
    mmlu_epochs: usize,
    /// Copies of GSM8K train in the mixture (0 omits it entirely).
    #[arg(long, default_value_t = 4)]
    gsm8k_epochs: usize,
    /// Seed for the mixture shuffle and the val caps. Every rank must use the
    /// same one — they build identical packs without communicating.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Map the parsed flags onto the library's config. Consumes `args` so the
/// paths move rather than clone.
///
/// Mechanical, but not free of risk: several destination fields are same-typed
/// and adjacent in meaning, so a crossed line here would type-check and pass
/// `SftConfig::validate`. `sft_flags_map_to_the_right_config_fields` is what
/// pins it.
fn sft_config(args: SftArgs) -> SftConfig {
    SftConfig {
        checkpoint: args.checkpoint,
        data: args.data,
        vocab: args.vocab,
        out: args.out,
        device_batch: args.device_batch,
        total_batch: args.total_batch,
        gpus: args.gpus,
        num_iters: args.num_iters,
        lrs: GroupLrs {
            embedding: args.embedding_lr,
            unembedding: args.unembedding_lr,
            matrix: args.matrix_lr,
        },
        init_lr_frac: args.init_lr_frac,
        warmup_steps: args.warmup_steps,
        warmdown_ratio: args.warmdown_ratio,
        final_lr_frac: args.final_lr_frac,
        log_every: args.log_every,
        eval_every: args.eval_every,
        eval_steps: args.eval_steps,
        mmlu_epochs: args.mmlu_epochs,
        gsm8k_epochs: args.gsm8k_epochs,
        seed: args.seed,
    }
}

/// The `sft --gpus N` parent. Unlike pretrain's, it has to read the checkpoint
/// meta first: `sequence_len` is not a flag here, and the batch geometry cannot
/// be checked without it. `load_meta`, not `load` — the parent never
/// materializes weights.
fn launch_sft(args: SftArgs) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = sft_config(args);
    cfg.validate()?;
    let meta = checkpoint::load_meta(&cfg.checkpoint)?;
    validate_batch_geometry(
        cfg.gpus,
        cfg.device_batch,
        meta.config.sequence_len,
        cfg.total_batch,
    )?;
    run_launcher(cfg.gpus)
}

fn run_sft(args: SftArgs, child: Option<ChildEnv>) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = sft_config(args);
    // Before `init_device`, so a typo'd flag does not first initialize CUDA and
    // join an NCCL world. `sft::run` re-checks — it is a library entry point
    // with its own callers — but this is where the *CLI* error surfaces.
    cfg.validate()?;
    let (device, dist) = init_device(child, cfg.gpus)?;
    sft::run(&cfg, &device, &dist)?;
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

    /// clap's own consistency check over the whole derive: duplicate long
    /// names, conflicting flags, defaults that do not parse as their type. It
    /// covers every subcommand at once, and catches these at test time rather
    /// than on first invocation on the box.
    #[test]
    fn cli_is_well_formed() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    /// Parse an `sft` argv and return the mapped config, panicking on a parse
    /// error. Goes through `try_parse_from` rather than a struct literal so the
    /// real flag names and `required`/default attributes are exercised.
    fn parse_sft(argv: &[&str]) -> SftConfig {
        match Cli::try_parse_from(argv)
            .expect("argv should parse")
            .command
        {
            Command::Sft(args) => sft_config(args),
            _ => panic!("expected the sft subcommand"),
        }
    }

    /// The minimum viable invocation: the four paths plus the three required
    /// LRs.
    fn minimal_sft_argv() -> Vec<&'static str> {
        vec![
            "rs-nanogpt",
            "sft",
            "--checkpoint",
            "ckpt",
            "--data",
            "sft-data",
            "--vocab",
            "vocab.txt",
            "--embedding-lr",
            "0.4243",
            "--unembedding-lr",
            "0.01131",
            "--matrix-lr",
            "0.003",
        ]
    }

    /// Every flag lands in its own config field.
    ///
    /// Each same-typed group gets a *distinct* value — `eval_every`/`eval_steps`,
    /// `mmlu_epochs`/`gsm8k_epochs`, `warmdown_ratio`/`final_lr_frac`/`init_lr_frac`,
    /// `warmup_steps`/`log_every`, and the three LRs — because a crossed line
    /// in ~20 lines of hand mapping type-checks and passes `validate`. Distinct
    /// values are what make the crossing observable.
    #[test]
    fn sft_flags_map_to_the_right_config_fields() {
        let cfg = parse_sft(&[
            "rs-nanogpt",
            "sft",
            "--checkpoint",
            "out-d24/run/best",
            "--data",
            "my-sft-data",
            "--vocab",
            "my-vocab.txt",
            "--out",
            "my-runs",
            "--embedding-lr",
            "0.11",
            "--unembedding-lr",
            "0.22",
            "--matrix-lr",
            "0.33",
            "--gpus",
            "8",
            "--device-batch",
            "4",
            "--total-batch",
            "262144",
            "--num-iters",
            "77",
            "--init-lr-frac",
            "0.6",
            "--warmup-steps",
            "5",
            "--warmdown-ratio",
            "0.4",
            "--final-lr-frac",
            "0.2",
            "--log-every",
            "9",
            "--eval-every",
            "300",
            "--eval-steps",
            "50",
            "--mmlu-epochs",
            "2",
            "--gsm8k-epochs",
            "6",
            "--seed",
            "1234",
        ]);

        assert_eq!(cfg.checkpoint, PathBuf::from("out-d24/run/best"));
        assert_eq!(cfg.data, PathBuf::from("my-sft-data"));
        assert_eq!(cfg.vocab, PathBuf::from("my-vocab.txt"));
        assert_eq!(cfg.out, PathBuf::from("my-runs"));
        assert_eq!(cfg.lrs.embedding, 0.11);
        assert_eq!(cfg.lrs.unembedding, 0.22);
        assert_eq!(cfg.lrs.matrix, 0.33);
        assert_eq!(cfg.gpus, 8);
        assert_eq!(cfg.device_batch, 4);
        assert_eq!(cfg.total_batch, 262_144);
        assert_eq!(cfg.num_iters, Some(77));
        assert_eq!(cfg.init_lr_frac, 0.6);
        assert_eq!(cfg.warmup_steps, 5);
        assert_eq!(cfg.warmdown_ratio, 0.4);
        assert_eq!(cfg.final_lr_frac, 0.2);
        assert_eq!(cfg.log_every, 9);
        assert_eq!(cfg.eval_every, 300);
        assert_eq!(cfg.eval_steps, 50);
        assert_eq!(cfg.mmlu_epochs, 2);
        assert_eq!(cfg.gsm8k_epochs, 6);
        assert_eq!(cfg.seed, 1234);
        // The mapped config is one `sft::run` would accept.
        assert_eq!(cfg.validate(), Ok(()));
    }

    /// The three LRs are `required`, not defaulted. Defaulting them to
    /// pretrain's CLI values would let a forgotten flag finetune a d24
    /// checkpoint at roughly half the embedding LR it was pretrained with, and
    /// nothing in the output would say so.
    #[test]
    fn sft_requires_each_learning_rate() {
        assert!(Cli::try_parse_from(minimal_sft_argv()).is_ok());
        for flag in ["--embedding-lr", "--unembedding-lr", "--matrix-lr"] {
            let argv = minimal_sft_argv();
            let at = argv.iter().position(|a| *a == flag).expect("flag present");
            let without: Vec<&str> = argv
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != at && *i != at + 1)
                .map(|(_, a)| *a)
                .collect();
            assert!(
                Cli::try_parse_from(&without).is_err(),
                "{flag} must be required"
            );
        }
    }

    /// The defaults the runbook documents, so a bare invocation is the
    /// intended run rather than whatever clap fell back to.
    #[test]
    fn sft_defaults_match_the_documented_cli() {
        let cfg = parse_sft(&minimal_sft_argv());
        assert_eq!(cfg.out, PathBuf::from("out-sft"));
        assert_eq!(cfg.gpus, 1);
        assert_eq!(cfg.device_batch, 8);
        assert_eq!(cfg.total_batch, 524_288);
        assert_eq!(cfg.num_iters, None, "absent means the whole epoch");
        assert_eq!(cfg.init_lr_frac, 0.8);
        assert_eq!(cfg.warmup_steps, 0);
        assert_eq!(cfg.warmdown_ratio, 0.5);
        assert_eq!(cfg.final_lr_frac, 0.0);
        assert_eq!(cfg.log_every, 10);
        assert_eq!(cfg.eval_every, 200);
        assert_eq!(cfg.eval_steps, 100);
        assert_eq!(cfg.mmlu_epochs, 3);
        assert_eq!(cfg.gsm8k_epochs, 4);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.validate(), Ok(()));
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
