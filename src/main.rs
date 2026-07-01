use std::path::PathBuf;
use std::process;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rs_nanogpt::data::{DataLoader, Split};
use rs_nanogpt::eval::tokenizer as tokenizer_eval;
use rs_nanogpt::model::{
    DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_NORM_EPS, DEFAULT_ROPE_BASE,
    DEFAULT_SEQUENCE_LEN, Gpt, GptConfig, default_device,
};
use rs_nanogpt::tokenizer::{BpeTokenizer, BpeTokenizerTrainer};
use rs_nanogpt::train::{
    DEFAULT_FINAL_LR_FRAC, DEFAULT_WARMDOWN_RATIO, DEFAULT_WARMUP_STEPS, GroupLrs, TrainConfig,
    train,
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
}

fn validate_pretrain_args(args: &PretrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.device_batch == 0 {
        return Err("--device-batch must be >= 1".into());
    }
    // sequence_len is also a GptConfig invariant (re-checked in GptConfig::validate);
    // it's validated here too so the micro-batch size below is nonzero.
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

    let mut loader = DataLoader::open(
        &args.data,
        Split::Train,
        &tokenizer,
        args.device_batch,
        config.sequence_len,
    )?;

    let device = default_device()?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Gpt::new(config, vb)?;
    let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("\nmodel built on {device:?}: {n_params} parameters");

    println!(
        "\ntraining: {} iters, total_batch {} tokens, device_batch {}, grad_accum {}",
        args.num_iters, args.total_batch, args.device_batch, grad_accum
    );
    train(&model, &varmap, &mut loader, &train_cfg, &device)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
        }
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
        let mut a = valid_args();
        a.device_batch = 0;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_zero_sequence_len() {
        let mut a = valid_args();
        a.sequence_len = 0;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_zero_num_iters() {
        let mut a = valid_args();
        a.num_iters = 0;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_zero_log_every() {
        let mut a = valid_args();
        a.log_every = 0;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_out_of_range_warmdown_ratio() {
        let mut a = valid_args();
        a.warmdown_ratio = 1.5;
        assert!(validate_pretrain_args(&a).is_err());
        a.warmdown_ratio = -0.1;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_out_of_range_final_lr_frac() {
        let mut a = valid_args();
        a.final_lr_frac = 2.0;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_total_batch_not_multiple_of_micro() {
        let mut a = valid_args();
        a.total_batch = 16384 + 1;
        assert!(validate_pretrain_args(&a).is_err());
    }

    #[test]
    fn rejects_total_batch_smaller_than_micro() {
        let mut a = valid_args();
        a.total_batch = 16384 / 2; // grad_accum would be 0
        assert!(validate_pretrain_args(&a).is_err());
    }
}
