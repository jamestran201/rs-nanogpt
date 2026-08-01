//! SFT task datasets, mirroring nanochat's `tasks/` package: the download
//! manifest behind `download-sft-data`, plus the parsers that turn the
//! downloaded files into [`Conversation`]s (smoltalk, mmlu, gsm8k,
//! custom_json). Spec: `writeups/sft-data-plan.md`.
//!
//! Parsers validate shape only — schema, types, role strings, choice arity,
//! JSON syntax. Conversation structure (alternation, system placement,
//! content kinds per role) is `render_conversation`'s job, so the rule lives
//! in one place. nanochat asserts it in both the parsers and the renderer; we
//! deliberately don't.

pub mod custom_json;
mod download;
pub mod gsm8k;
pub mod mixture;
pub mod mmlu;
pub mod pack;
pub mod smoltalk;

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use arrow_array::RecordBatch;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

use crate::tokenizer::{Conversation, Role};

pub use download::download_sft_data;

/// Which split of an SFT dataset to load. Distinct from the pretraining
/// `data::parquet::Split`, which selects shards by position; here train and
/// test are separate files named by the download manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Test,
}

impl Split {
    fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Test => "test",
        }
    }
}

pub(crate) fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// The sorted `{prefix}*.parquet` files in `dir`. No match is an error, not an
/// empty dataset: the file set is fixed, so absence means the data was never
/// downloaded (or landed under the wrong name) and should fail loudly.
fn dataset_files(dir: &Path, prefix: &str) -> io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.starts_with(prefix) && name.ends_with(".parquet")).then_some(path)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no {prefix}*.parquet files in {} — run download-sft-data",
                dir.display()
            ),
        ));
    }
    Ok(files)
}

fn open_parquet(path: &Path) -> io::Result<ParquetRecordBatchReader> {
    let file = fs::File::open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    ParquetRecordBatchReaderBuilder::try_new(file)
        .and_then(|b| b.build())
        .map_err(|e| invalid(format!("{}: {e}", path.display())))
}

/// Decode every `{dataset}-{split}-*.parquet` file in `dir`, in sorted file
/// order, by handing each record batch to `decode`. The `usize` argument is
/// the batch's first row index *within its file*, so row numbers in errors
/// match what a reader of that file would see.
fn load_dataset(
    dir: &Path,
    dataset: &str,
    split: Split,
    mut decode: impl FnMut(&RecordBatch, &Path, usize) -> io::Result<Vec<Conversation>>,
) -> io::Result<Vec<Conversation>> {
    let mut out = Vec::new();
    for path in dataset_files(dir, &format!("{dataset}-{}-", split.as_str()))? {
        let mut first_row = 0usize;
        for batch in open_parquet(&path)? {
            let batch = batch.map_err(|e| invalid(format!("{}: {e}", path.display())))?;
            out.extend(decode(&batch, &path, first_row)?);
            first_row += batch.num_rows();
        }
    }
    Ok(out)
}

fn role_from_str(s: &str) -> Option<Role> {
    match s {
        "system" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::test_support::{
        byte_tokenizer, write_gsm8k_shard, write_mmlu_shard, write_smoltalk_shard,
    };

    /// Every parser's output renders, including a smoltalk system-message row
    /// through the renderer's merge path.
    #[test]
    fn every_parser_output_renders() {
        let dir = tempfile::tempdir().unwrap();
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00000.parquet"),
            &[
                &[("system", "Be brief."), ("user", "Hi"), ("assistant", "Yo")],
                &[("user", "One"), ("assistant", "Two")],
            ],
        );
        write_mmlu_shard(
            &dir.path().join("mmlu-train-00000.parquet"),
            &[("What is 2+2?", &["3", "4", "5", "6"], 1)],
        );
        write_gsm8k_shard(
            &dir.path().join("gsm8k-train-00000.parquet"),
            &[("How many?", "It is <<1+2=3>>3.\n#### 3")],
        );
        let jsonl = dir.path().join("identity_conversations.jsonl");
        fs::File::create(&jsonl)
            .unwrap()
            .write_all(
                concat!(
                    r#"[{"role":"user","content":"Who are you?"},"#,
                    r#"{"role":"assistant","content":"nanochat"}]"#,
                    "\n"
                )
                .as_bytes(),
            )
            .unwrap();

        let mut all: Vec<Conversation> = Vec::new();
        all.extend(smoltalk::load(dir.path(), Split::Train).unwrap());
        all.extend(mmlu::load(dir.path(), Split::Train).unwrap());
        all.extend(gsm8k::load(dir.path(), Split::Train).unwrap());
        all.extend(custom_json::load(&jsonl).unwrap());
        assert_eq!(all.len(), 5);

        let tok = byte_tokenizer();
        for (i, conv) in all.iter().enumerate() {
            let rendered = tok
                .render_conversation(conv, 10_000)
                .unwrap_or_else(|e| panic!("conversation {i} failed to render: {e}"));
            assert!(!rendered.ids.is_empty());
        }
    }

    /// One JSONL line per identity-style conversation, as
    /// `identity_conversations.jsonl` ships.
    fn write_identity_jsonl(path: &Path, count: usize) {
        let mut file = fs::File::create(path).unwrap();
        for i in 0..count {
            writeln!(
                file,
                r#"[{{"role":"user","content":"Who are you? ({i})"}},{{"role":"assistant","content":"I am nanochat, a small language model."}}]"#
            )
            .unwrap();
        }
    }

    /// Capstone: every parser's output → mixture → packed epoch → batch →
    /// model → masked loss. Closes the loop the `sft` subcommand will run,
    /// mirroring `loader_feeds_model_and_loss_end_to_end` for pretraining.
    #[test]
    fn mixture_packs_into_batches_that_feed_model_and_loss() {
        use candle_core::Device;

        use crate::model::{Reduction, cross_entropy};
        use crate::test_support::tiny_gpt;
        use mixture::{Source, render_mixture};
        use pack::pack_epoch;

        let dir = tempfile::tempdir().unwrap();
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00000.parquet"),
            &[
                &[("system", "Be brief."), ("user", "Hi"), ("assistant", "Yo")],
                &[("user", "One"), ("assistant", "Two")],
            ],
        );
        write_mmlu_shard(
            &dir.path().join("mmlu-train-00000.parquet"),
            &[("What is 2+2?", &["3", "4", "5", "6"], 1)],
        );
        write_gsm8k_shard(
            &dir.path().join("gsm8k-train-00000.parquet"),
            &[("How many?", "It is <<1+2=3>>3.\n#### 3")],
        );
        let jsonl = dir.path().join("identity_conversations.jsonl");
        write_identity_jsonl(&jsonl, 4);

        // Rows of seq_len + 1, wide enough that no fixture is truncated (the
        // longest is MMLU's rendered prompt at ~121 byte-tokens), so every
        // conversation keeps its scored `<|assistant_end|>`.
        let (device_batch, seq_len) = (2usize, 128usize);
        let tok = byte_tokenizer();
        let sources = vec![
            Source {
                name: "smoltalk",
                conversations: smoltalk::load(dir.path(), Split::Train).unwrap(),
            },
            Source {
                name: "mmlu",
                conversations: mmlu::load(dir.path(), Split::Train).unwrap(),
            },
            Source {
                name: "gsm8k",
                conversations: gsm8k::load(dir.path(), Split::Train).unwrap(),
            },
            Source {
                name: "identity",
                conversations: custom_json::load(&jsonl).unwrap(),
            },
        ];

        let rendered = render_mixture(&tok, sources, seq_len + 1, 42).unwrap();
        assert!(rendered.iter().all(|r| r.ids.len() <= seq_len + 1));
        let epoch = pack_epoch(rendered, seq_len + 1, tok.bos_id());
        let num_batches = epoch.num_batches(1, device_batch);
        assert!(
            num_batches >= 1,
            "fixtures pack into {} rows, too few for a batch of {device_batch}",
            epoch.num_rows()
        );

        let (_vm, model) = tiny_gpt(tok.vocab_size(), seq_len);
        let (mut saw_ignored, mut saw_scored) = (false, false);
        for i in 0..num_batches {
            let batch = epoch.batch(i, 0, 1, device_batch, &Device::Cpu).unwrap();
            let logits = model.forward(&batch.inputs).unwrap();
            assert_eq!(logits.dims(), &[device_batch, seq_len, tok.vocab_size()]);

            let loss = cross_entropy(&logits, &batch.targets, -1, Reduction::Mean)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(loss.is_finite(), "expected finite loss, got {loss}");

            for target in batch
                .targets
                .flatten_all()
                .unwrap()
                .to_vec1::<i64>()
                .unwrap()
            {
                if target < 0 {
                    saw_ignored = true;
                } else {
                    saw_scored = true;
                }
            }
        }
        assert!(
            saw_ignored && saw_scored,
            "the epoch must contain both ignored and scored targets"
        );
    }

    // Smokes against the real files in `sft-data/`; run with
    // `cargo test -- --ignored`. Counts are per single local file — smoltalk
    // asserts `>=` because the box holds all 4 train shards.

    fn sft_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("sft-data")
    }

    /// Loading validates shape; rendering checks that real data also passes
    /// the structure checks.
    fn render_prefix(convs: &[Conversation], limit: usize) {
        let tok = byte_tokenizer();
        for (i, conv) in convs.iter().take(limit).enumerate() {
            tok.render_conversation(conv, 100_000)
                .unwrap_or_else(|e| panic!("conversation {i} failed to render: {e}"));
        }
    }

    #[test]
    #[ignore = "needs real data in sft-data/"]
    fn real_smoltalk_train() {
        let convs = smoltalk::load(&sft_dir(), Split::Train).unwrap();
        assert!(convs.len() >= 115_991, "got {}", convs.len());
        render_prefix(&convs, 10_000);
    }

    #[test]
    #[ignore = "needs real data in sft-data/"]
    fn real_mmlu_train() {
        let convs = mmlu::load(&sft_dir(), Split::Train).unwrap();
        assert_eq!(convs.len(), 99_842);
        render_prefix(&convs, convs.len());
    }

    #[test]
    #[ignore = "needs real data in sft-data/"]
    fn real_gsm8k_train() {
        let convs = gsm8k::load(&sft_dir(), Split::Train).unwrap();
        assert_eq!(convs.len(), 7_473);
        render_prefix(&convs, convs.len());
    }

    /// Real conversations through the whole phase-3 pipeline at the production
    /// row size, checking the conservation invariants and printing what the
    /// pack cost. Run with `cargo test --release -- --ignored --nocapture`.
    ///
    /// The pad fraction here is *not* the number the box will see: the byte
    /// tokenizer emits ~4× the tokens a trained vocab does, so most
    /// conversations render to exactly 2049 and fill rows perfectly — which is
    /// why the truncation count is printed alongside it.
    #[test]
    #[ignore = "needs real data in sft-data/"]
    fn real_mixture_packs_with_conservation() {
        use mixture::{Source, render_mixture};
        use pack::pack_epoch;

        const ROW_CAPACITY: usize = 2049; // seq_len 2048 + 1

        let mut conversations = smoltalk::load(&sft_dir(), Split::Train).unwrap();
        conversations.truncate(20_000);

        let dir = tempfile::tempdir().unwrap();
        let jsonl = dir.path().join("identity_conversations.jsonl");
        write_identity_jsonl(&jsonl, 1_000);
        let identity = custom_json::load(&jsonl).unwrap();

        let tok = byte_tokenizer();
        let sources = vec![
            Source {
                name: "smoltalk",
                conversations,
            },
            // Twice, as nanochat oversamples the identity file.
            Source {
                name: "identity",
                conversations: identity.clone(),
            },
            Source {
                name: "identity",
                conversations: identity,
            },
        ];
        let expected: usize = sources.iter().map(|s| s.conversations.len()).sum();

        let rendered = render_mixture(&tok, sources, ROW_CAPACITY, 42).unwrap();
        assert_eq!(rendered.len(), expected);
        let content: usize = rendered.iter().map(|r| r.ids.len()).sum();
        let truncated = rendered
            .iter()
            .filter(|r| r.ids.len() == ROW_CAPACITY)
            .count();

        let epoch = pack_epoch(rendered, ROW_CAPACITY, tok.bos_id());
        let stats = epoch.stats;
        assert_eq!(stats.conversations, expected, "no conversation may be lost");
        assert_eq!(stats.total_tokens, stats.rows * ROW_CAPACITY);
        assert_eq!(content + stats.pad_tokens, stats.total_tokens);
        assert_eq!(epoch.num_rows(), stats.rows);

        println!(
            "{stats:?}\npad {:.2}% | truncated at cap {truncated}/{expected} ({:.1}%)",
            100.0 * stats.pad_fraction(),
            100.0 * truncated as f64 / expected as f64,
        );
    }
}
