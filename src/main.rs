use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};
use rs_nanogpt::eval::tokenizer as tokenizer_eval;
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
    }
    Ok(())
}
