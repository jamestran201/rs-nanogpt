use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::train::{GroupLrs, PoolStats};

#[derive(Debug, Serialize)]
pub struct RunMeta {
    /// Which training phase produced this run: `"pretrain"` or `"sft"`.
    pub phase: &'static str,
    pub device: String,
    pub dtype: &'static str,
    pub started_at_unix: u64,
    pub n_params: usize,
    // model geometry
    pub vocab_size: usize,
    pub sequence_len: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    // optimization / batch geometry
    pub num_iters: usize,
    pub device_batch: usize,
    pub total_batch: usize,
    /// Data-parallel ranks (the CLI's `--gpus`); 1 for a single-process run.
    pub world_size: usize,
    pub grad_accum: usize,
    /// Global tokens per optimizer step (== `total_batch`), not the per-rank
    /// share, so runs at different GPU counts stay comparable.
    pub tokens_per_step: usize,
    pub embedding_lr: f64,
    pub unembedding_lr: f64,
    pub matrix_lr: f64,
    pub warmup_steps: usize,
    pub warmdown_ratio: f64,
    pub final_lr_frac: f64,
    // cadences
    pub log_every: usize,
    pub eval_every: usize,
    pub eval_steps: usize,
    pub sample_every: usize,
    /// Present only on an SFT run (`phase == "sft"`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sft: Option<SftRunMeta>,
}

/// The SFT-only half of a run's provenance: what the finetune was built from
/// and what the mixture packed to, so an `sft` run is reproducible from its
/// artifact alone. The shared `RunMeta` fields keep pretraining's conventions
/// (global `tokens_per_step`, post-`init_lr_frac` LRs) so the two phases stay
/// comparable.
#[derive(Debug, Serialize)]
pub struct SftRunMeta {
    pub base_checkpoint: String,
    pub seed: u64,
    pub mmlu_epochs: usize,
    pub gsm8k_epochs: usize,
    pub conversations: usize,
    pub rows: usize,
    pub pad_fraction: f64,
    pub scored_fraction: f64,
    pub val_rows: usize,
}

pub fn write_run_json(path: &Path, meta: &RunMeta) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let bytes = serde_json::to_vec_pretty(meta).expect("RunMeta serializes");
    std::fs::write(path, bytes)
}

/// Claim a fresh run directory `root/<id>`, walking `-2`, `-3`, … on
/// collision, and create it. Lives here rather than in the binary so both
/// training phases can reach it — `run.json` and `metrics.jsonl` are written
/// into what it returns.
pub fn unique_run_dir(root: &Path, id: &str) -> io::Result<PathBuf> {
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

#[derive(Debug, Serialize)]
pub struct MetricRecord {
    pub step: usize,
    pub kind: &'static str,
    pub elapsed_s: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_loss: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_matrix: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_embedding: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr_unembedding: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tok_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms_per_step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub val_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bpb: Option<f64>,
    // CUDA async-mempool sample (`kind: "mem"`), raw bytes so the JSONL stays
    // exact and the analysis picks its own units. Only present when
    // RS_NANOGPT_MEM_TRACE is on; see `crate::train::mem`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_used: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_used_high: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_reserved: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_reserved_high: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_dev_free: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct Throughput {
    pub tok_per_s: f64,
    pub ms_per_step: f64,
}

impl MetricRecord {
    pub fn train(
        step: usize,
        elapsed_s: f64,
        train_loss: f32,
        grad_norm: f64,
        lrs: GroupLrs,
        rate: Option<Throughput>,
    ) -> Self {
        Self {
            step,
            kind: "train",
            elapsed_s,
            train_loss: Some(train_loss),
            grad_norm: Some(grad_norm),
            lr_matrix: Some(lrs.matrix),
            lr_embedding: Some(lrs.embedding),
            lr_unembedding: Some(lrs.unembedding),
            tok_per_s: rate.map(|r| r.tok_per_s),
            ms_per_step: rate.map(|r| r.ms_per_step),
            val_loss: None,
            bpb: None,
            mem_used: None,
            mem_used_high: None,
            mem_reserved: None,
            mem_reserved_high: None,
            mem_dev_free: None,
        }
    }

    pub fn eval(step: usize, elapsed_s: f64, val_loss: f64, bpb: f64) -> Self {
        Self {
            step,
            kind: "eval",
            elapsed_s,
            train_loss: None,
            grad_norm: None,
            lr_matrix: None,
            lr_embedding: None,
            lr_unembedding: None,
            tok_per_s: None,
            ms_per_step: None,
            val_loss: Some(val_loss),
            bpb: Some(bpb),
            mem_used: None,
            mem_used_high: None,
            mem_reserved: None,
            mem_reserved_high: None,
            mem_dev_free: None,
        }
    }

    /// One CUDA memory-pool sample. `mem_reserved − mem_used` is the
    /// fragmentation the OOM investigation is chasing; the `_high` pair is the
    /// true intra-step peak since the previous sample reset them.
    pub fn mem(step: usize, elapsed_s: f64, stats: PoolStats) -> Self {
        Self {
            step,
            kind: "mem",
            elapsed_s,
            train_loss: None,
            grad_norm: None,
            lr_matrix: None,
            lr_embedding: None,
            lr_unembedding: None,
            tok_per_s: None,
            ms_per_step: None,
            val_loss: None,
            bpb: None,
            mem_used: Some(stats.used),
            mem_used_high: Some(stats.used_high),
            mem_reserved: Some(stats.reserved),
            mem_reserved_high: Some(stats.reserved_high),
            mem_dev_free: Some(stats.dev_free),
        }
    }
}

/// Append-only JSONL sink for `MetricRecord`s. Interior mutability lets it live
/// behind a shared `&` inside `EvalContext`. Best-effort: the first IO error warns
/// once to stderr and every failure is otherwise swallowed, so a lost metrics line
/// never aborts training.
pub struct MetricsLogger {
    sink: Option<RefCell<BufWriter<File>>>,
    warned: Cell<bool>,
}

impl MetricsLogger {
    pub fn create(path: &Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            sink: Some(RefCell::new(BufWriter::new(File::create(path)?))),
            warned: Cell::new(false),
        })
    }

    /// A no-op sink that discards every record. Non-master ranks in a
    /// multi-GPU run use this so `EvalContext` keeps its shape while only
    /// rank 0 writes metrics.jsonl.
    pub fn null() -> Self {
        Self {
            sink: None,
            warned: Cell::new(false),
        }
    }

    pub fn log(&self, rec: &MetricRecord) {
        if let Err(e) = self.try_log(rec)
            && !self.warned.replace(true)
        {
            eprintln!("warning: metrics logging failed ({e}); continuing without it");
        }
    }

    fn try_log(&self, rec: &MetricRecord) -> std::io::Result<()> {
        let Some(sink) = &self.sink else {
            return Ok(()); // null logger: silently discard
        };
        let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
        let mut w = sink.borrow_mut();
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        // Flush per line so a killed run keeps everything logged so far.
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::io::{BufRead, BufReader};

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

    #[test]
    fn metrics_logger_writes_train_and_eval_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.jsonl");
        {
            let logger = MetricsLogger::create(&path).unwrap();
            logger.log(&MetricRecord::train(
                10,
                1.5,
                3.25,
                0.42,
                GroupLrs {
                    embedding: 2e-3,
                    unembedding: 3e-4,
                    matrix: 1e-3,
                },
                Some(Throughput {
                    tok_per_s: 1000.0,
                    ms_per_step: 20.0,
                }),
            ));
            logger.log(&MetricRecord::eval(20, 2.0, 3.1, 1.4));
        } // drop flushes the BufWriter

        let lines: Vec<String> = BufReader::new(File::open(&path).unwrap())
            .lines()
            .map(|l| l.unwrap())
            .collect();
        assert_eq!(lines.len(), 2);

        let train: Value = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(train["kind"], "train");
        assert_eq!(train["step"], 10);
        assert!(train.get("train_loss").is_some());
        assert!(train.get("grad_norm").is_some());
        assert!(train.get("tok_per_s").is_some());
        // eval-only fields are omitted from a train record (skip_serializing_if).
        assert!(train.get("val_loss").is_none());
        assert!(train.get("bpb").is_none());

        let eval: Value = serde_json::from_str(&lines[1]).unwrap();
        assert_eq!(eval["kind"], "eval");
        assert!(eval.get("val_loss").is_some());
        assert!(eval.get("bpb").is_some());
        // train-only fields are omitted from an eval record.
        assert!(eval.get("train_loss").is_none());
        assert!(eval.get("grad_norm").is_none());
        assert!(eval.get("tok_per_s").is_none());
    }

    #[test]
    fn null_logger_discards_records_without_error() {
        let logger = MetricsLogger::null();
        logger.log(&MetricRecord::eval(1, 0.5, 3.0, 1.2));
        // Nothing observable to assert beyond "no panic, no warning state".
        assert!(!logger.warned.get());
    }

    #[test]
    fn train_record_omits_rate_fields_when_none() {
        let rec = MetricRecord::train(
            0,
            0.1,
            9.0,
            0.0,
            GroupLrs {
                embedding: 2e-3,
                unembedding: 3e-4,
                matrix: 1e-3,
            },
            None,
        );
        let v: Value = serde_json::to_value(&rec).unwrap();
        assert!(v.get("tok_per_s").is_none());
        assert!(v.get("ms_per_step").is_none());
        // required fields are still present even when the rate fields are absent.
        assert!(v.get("train_loss").is_some());
        assert_eq!(v["kind"], "train");
    }

    #[test]
    fn mem_record_carries_raw_bytes_and_omits_training_fields() {
        let rec = MetricRecord::mem(
            7,
            88.0,
            PoolStats {
                reserved: 61_440_000_000,
                reserved_high: 61_500_000_000,
                used: 13_280_000_000,
                used_high: 41_900_000_000,
                dev_free: 21_100_000_000,
                dev_total: 85_520_000_000,
            },
        );
        let v: Value = serde_json::to_value(&rec).unwrap();
        assert_eq!(v["kind"], "mem");
        assert_eq!(v["step"], 7);
        // Bytes, not GB: the analysis picks its own units and reserved − used
        // must stay exact.
        assert_eq!(v["mem_reserved"], 61_440_000_000u64);
        assert_eq!(v["mem_used"], 13_280_000_000u64);
        assert_eq!(v["mem_used_high"], 41_900_000_000u64);
        // dev_total is constant for a run and lives in run.json's territory,
        // so it is deliberately not repeated on every sample.
        assert!(v.get("mem_dev_total").is_none());
        assert!(v.get("train_loss").is_none());
        assert!(v.get("val_loss").is_none());
    }

    #[test]
    fn train_and_eval_records_omit_the_memory_fields() {
        let eval: Value = serde_json::to_value(MetricRecord::eval(1, 0.5, 3.0, 1.2)).unwrap();
        assert!(eval.get("mem_used").is_none());
        assert!(eval.get("mem_reserved").is_none());
    }

    #[test]
    fn run_meta_serializes_expected_keys() {
        let meta = RunMeta {
            phase: "pretrain",
            device: "Cpu".into(),
            dtype: "f32",
            started_at_unix: 1_700_000_000,
            n_params: 12_345,
            vocab_size: 512,
            sequence_len: 512,
            n_layer: 6,
            n_head: 6,
            n_embd: 384,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
            num_iters: 5000,
            device_batch: 32,
            total_batch: 16384,
            world_size: 1,
            grad_accum: 1,
            tokens_per_step: 16384,
            embedding_lr: 0.2,
            unembedding_lr: 0.004,
            matrix_lr: 0.003,
            warmup_steps: 40,
            warmdown_ratio: 0.65,
            final_lr_frac: 0.05,
            log_every: 10,
            eval_every: 250,
            eval_steps: 20,
            sample_every: 0,
            sft: None,
        };
        let v: Value = serde_json::to_value(&meta).unwrap();
        assert_eq!(v["device"], "Cpu");
        assert_eq!(v["dtype"], "f32");
        assert_eq!(v["n_params"], 12_345);
        assert_eq!(v["tokens_per_step"], 16384);
        assert_eq!(v["world_size"], 1);
        assert_eq!(v["phase"], "pretrain");
        // A pretrain run carries no SFT block at all, rather than a null one.
        assert!(v.get("sft").is_none());
        // Git-commit provenance is intentionally out of scope for now.
        assert!(v.get("git_commit").is_none());
    }

    /// An SFT run's nested block serializes inline under `sft`.
    #[test]
    fn run_meta_carries_the_sft_block_when_present() {
        let v: Value = serde_json::to_value(SftRunMeta {
            base_checkpoint: "out-d24/1752192000/best".into(),
            seed: 42,
            mmlu_epochs: 3,
            gsm8k_epochs: 4,
            conversations: 800_000,
            rows: 150_000,
            pad_fraction: 0.07,
            scored_fraction: 0.38,
            val_rows: 1_600,
        })
        .unwrap();
        assert_eq!(v["base_checkpoint"], "out-d24/1752192000/best");
        assert_eq!(v["mmlu_epochs"], 3);
        assert_eq!(v["val_rows"], 1_600);
    }
}
