use std::path::PathBuf;
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
    DEFAULT_SEQUENCE_LEN, Gpt, GptConfig, default_device,
};
use rs_nanogpt::tokenizer::{BpeTokenizer, BpeTokenizerTrainer};
use rs_nanogpt::train::{
    DEFAULT_FINAL_LR_FRAC, DEFAULT_WARMDOWN_RATIO, DEFAULT_WARMUP_STEPS, EvalContext, GroupLrs,
    TrainConfig, train,
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
            if let Err(err) = run_pretrain(args) {
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
    /// Total tokens per optimizer step. Must be a multiple of
    /// device_batch*seq_len; the quotient is the gradient-accumulation steps.
    #[arg(long, default_value_t = 16384)]
    total_batch: usize,
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
    /// Output dir; best checkpoint saved to <out>/best/.
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

    // Gradient accumulation: total_batch is reached by whole micro-batches.
    let micro = args.device_batch * args.sequence_len;
    if !args.total_batch.is_multiple_of(micro) {
        return Err(format!(
            "--total-batch ({}) must be a multiple of device_batch*seq_len ({micro})",
            args.total_batch
        )
        .into());
    }
    let grad_accum = args.total_batch / micro;
    if grad_accum == 0 {
        return Err(format!(
            "--total-batch ({}) must be at least device_batch*seq_len ({micro})",
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

fn run_pretrain(args: PretrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    validate_pretrain_args(&args)?;

    let micro = args.device_batch * args.sequence_len;
    let grad_accum = args.total_batch / micro;
    let train_cfg = TrainConfig {
        num_iters: args.num_iters,
        grad_accum,
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
    print_config_summary(&config);

    let device = default_device()?;

    let mut loader = DataLoader::open(
        &args.data,
        Split::Train,
        &tokenizer,
        args.device_batch,
        config.sequence_len,
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
    println!("\nmodel built on {device:?} | dtype F32 | {n_params} parameters");

    println!(
        "\ntraining: {} iters, total_batch {} tokens, device_batch {}, grad_accum {}",
        args.num_iters, args.total_batch, args.device_batch, grad_accum
    );
    let started_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
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
        grad_accum,
        tokens_per_step: args.total_batch, // == grad_accum * device_batch * seq_len
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
    write_run_json(&args.out.join("run.json"), &run_meta)?;
    let metrics = MetricsLogger::create(&args.out.join("metrics.jsonl"))?;

    let eval = EvalContext {
        val_batches: &val_batches,
        tokenizer: &tokenizer,
        token_bytes: &token_bytes,
        ckpt_root: &args.out,
        metrics: &metrics,
        eval_every: args.eval_every,
        sample_every: args.sample_every,
        sample_tokens: args.sample_tokens,
        sample_temperature: args.sample_temperature,
    };
    train(&model, &varmap, &mut loader, &train_cfg, &eval, &device)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
