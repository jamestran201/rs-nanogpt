use std::cell::{Cell, RefCell};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::train::GroupLrs;

#[derive(Debug, Serialize)]
pub struct RunMeta {
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
    pub grad_accum: usize,
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
}

pub fn write_run_json(path: &Path, meta: &RunMeta) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let bytes = serde_json::to_vec_pretty(meta).expect("RunMeta serializes");
    std::fs::write(path, bytes)
}

/// Format a Unix timestamp (seconds) as `YYYY-MM-DD_HH-MM-SS` in UTC — used to
/// name a per-run output directory. Filesystem-safe (no colons) and lexically
/// sortable, so run folders sort chronologically. UTC keeps it independent of
/// the box's timezone; the calendar split is Howard Hinnant's civil-from-days
/// algorithm, so it needs no timezone crate.
pub fn run_timestamp(unix_secs: u64) -> String {
    let secs = unix_secs as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (sod / 3600, (sod % 3600) / 60, sod % 60);

    // civil_from_days: days since 1970-01-01 -> (year, month, day) in UTC.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day-of-era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day-of-year (Mar-based), [0, 365]
    let mp = (5 * doy + 2) / 153; // month, Mar=0 .. Feb=11
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    format!("{year:04}-{m:02}-{d:02}_{hh:02}-{mm:02}-{ss:02}")
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
        }
    }
}

/// Append-only JSONL sink for `MetricRecord`s. Interior mutability lets it live
/// behind a shared `&` inside `EvalContext`. Best-effort: the first IO error warns
/// once to stderr and every failure is otherwise swallowed, so a lost metrics line
/// never aborts training.
pub struct MetricsLogger {
    sink: RefCell<BufWriter<File>>,
    warned: Cell<bool>,
}

impl MetricsLogger {
    pub fn create(path: &Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Ok(Self {
            sink: RefCell::new(BufWriter::new(File::create(path)?)),
            warned: Cell::new(false),
        })
    }

    pub fn log(&self, rec: &MetricRecord) {
        if let Err(e) = self.try_log(rec)
            && !self.warned.replace(true)
        {
            eprintln!("warning: metrics logging failed ({e}); continuing without it");
        }
    }

    fn try_log(&self, rec: &MetricRecord) -> std::io::Result<()> {
        let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
        let mut w = self.sink.borrow_mut();
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
    fn run_timestamp_formats_utc() {
        // Epoch and a known second: 1_700_000_000 == 2023-11-14 22:13:20 UTC.
        assert_eq!(run_timestamp(0), "1970-01-01_00-00-00");
        assert_eq!(run_timestamp(1_700_000_000), "2023-11-14_22-13-20");
        // Leap day boundary: 2024-02-29 is a valid date the algorithm must land on.
        assert_eq!(run_timestamp(1_709_208_000), "2024-02-29_12-00-00");
    }

    #[test]
    fn run_timestamp_is_sortable_and_filesystem_safe() {
        let earlier = run_timestamp(1_700_000_000);
        let later = run_timestamp(1_700_003_600); // +1h
        assert!(earlier < later, "timestamps must sort chronologically");
        assert!(!earlier.contains(':'), "no colons (Windows-unsafe / awkward)");
        assert!(!earlier.contains('/'), "no path separators");
    }

    #[test]
    fn run_meta_serializes_expected_keys() {
        let meta = RunMeta {
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
        };
        let v: Value = serde_json::to_value(&meta).unwrap();
        assert_eq!(v["device"], "Cpu");
        assert_eq!(v["dtype"], "f32");
        assert_eq!(v["n_params"], 12_345);
        assert_eq!(v["tokens_per_step"], 16384);
        // Git-commit provenance is intentionally out of scope for now.
        assert!(v.get("git_commit").is_none());
    }
}
