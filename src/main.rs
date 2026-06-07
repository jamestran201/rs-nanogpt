use std::path::PathBuf;
use std::process;

use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use clap::{Parser, Subcommand};
use rs_nanogpt::eval::tokenizer as tokenizer_eval;
use rs_nanogpt::model::{
    DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_NORM_EPS, DEFAULT_ROPE_BASE,
    DEFAULT_SEQUENCE_LEN, DEFAULT_VOCAB_SIZE, Gpt, GptConfig, default_device,
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
    Pretrain {
        /// Tokenizer vocabulary size.
        #[arg(long, default_value_t = DEFAULT_VOCAB_SIZE)]
        vocab_size: usize,
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
    },
}

fn main() -> std::io::Result<()> {
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
        Command::Pretrain {
            vocab_size,
            sequence_len,
            n_layer,
            n_head,
            n_embd,
            rope_base,
            norm_eps,
        } => {
            let config = GptConfig {
                vocab_size,
                sequence_len,
                n_layer,
                n_head,
                n_embd,
                rope_base,
                norm_eps,
            };
            if let Err(err) = config.validate() {
                eprintln!("invalid config: {err}");
                process::exit(1);
            }
            print_config_summary(&config);
            if let Err(err) = build_model(config) {
                eprintln!("failed to build model: {err}");
                process::exit(1);
            }
            // TODO(pretraining): add the transformer blocks, optimizer, data
            // loader, WSD schedule, and training/eval loop. See
            // writeups/pretraining-mvp-architecture.md.
            eprintln!("\nnote: training loop not yet implemented (model scaffold only).");
        }
    }
    Ok(())
}

/// Build the GPT model on the selected device with fresh-init weights, holding
/// its parameters in a `VarMap` (ready for the optimizer/checkpointing later).
fn build_model(config: GptConfig) -> candle_core::Result<()> {
    let device = default_device()?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = Gpt::new(config, vb)?;

    let n_params: usize = varmap
        .all_vars()
        .iter()
        .map(|v| v.elem_count())
        .sum();
    println!("\nmodel built on {device:?}: {n_params} parameters");
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
