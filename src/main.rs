use std::path::PathBuf;
use std::process;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rs_nanogpt::data::{DataLoader, Split};
use rs_nanogpt::eval::tokenizer as tokenizer_eval;
use rs_nanogpt::model::{
    DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_NORM_EPS, DEFAULT_ROPE_BASE,
    DEFAULT_SEQUENCE_LEN, Gpt, GptConfig, Reduction, cross_entropy, default_device,
};
use rs_nanogpt::tokenizer::{BpeTokenizer, BpeTokenizerTrainer};

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
    /// Rows per batch (B).
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    /// Number of batches to pull for the forward-only loss probe.
    #[arg(long, default_value_t = 20)]
    steps: usize,
}

fn run_pretrain(args: PretrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    let device = default_device()?;

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

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = Gpt::new(config.clone(), vb)?;
    let n_params: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("\nmodel built on {device:?}: {n_params} parameters");

    let mut loader = DataLoader::open(
        &args.data,
        Split::Train,
        &tokenizer,
        args.batch_size,
        config.sequence_len,
    )?;

    println!("\nforward-only loss probe ({} steps, no optimizer):", args.steps);
    for step in 0..args.steps {
        let batch = loader.next_batch(&device)?;
        let logits = model.forward(&batch.inputs)?;
        let loss = cross_entropy(&logits, &batch.targets, -1, Reduction::Mean)?
            .to_scalar::<f32>()?;
        println!("step {step}: loss {loss:.4}");
    }
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
