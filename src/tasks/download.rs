//! The SFT data manifest: the fixed set of files chat-SFT trains and
//! validates on, fetched with the generic downloader. Ports the data-fetch
//! side of nanochat's `tasks/` package (which delegates to HuggingFace
//! `datasets`) plus the identity-file `curl` at `speedrun.sh:82`.

use std::io;
use std::path::Path;

use crate::data::{FileJob, Summary, download_files};

const HF: &str = "https://huggingface.co/datasets";

const IDENTITY_URL: &str =
    "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl";

/// `(repo path under huggingface.co/datasets, local filename)`. All three
/// repos ship native parquet, so these are plain `resolve/main` file URLs.
/// Hardcoded like `MAX_SHARD`: if a repo is re-sharded upstream the download
/// 404s loudly and this table needs updating. Local names are dataset-prefixed
/// so the parsers can find each dataset+split by filename prefix.
const SFT_FILES: &[(&str, &str)] = &[
    // SmolTalk (~922 MB train + 48 MB test): 460K/24K general conversations.
    (
        "HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00000-of-00004.parquet",
        "smoltalk-train-00000.parquet",
    ),
    (
        "HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00001-of-00004.parquet",
        "smoltalk-train-00001.parquet",
    ),
    (
        "HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00002-of-00004.parquet",
        "smoltalk-train-00002.parquet",
    ),
    (
        "HuggingFaceTB/smol-smoltalk/resolve/main/data/train-00003-of-00004.parquet",
        "smoltalk-train-00003.parquet",
    ),
    (
        "HuggingFaceTB/smol-smoltalk/resolve/main/data/test-00000-of-00001.parquet",
        "smoltalk-test-00000.parquet",
    ),
    // MMLU (~48 MB train + 3.5 MB test): 100K/14K multiple-choice questions;
    // auxiliary_train is the split nanochat trains on.
    (
        "cais/mmlu/resolve/main/all/auxiliary_train-00000-of-00001.parquet",
        "mmlu-train-00000.parquet",
    ),
    (
        "cais/mmlu/resolve/main/all/test-00000-of-00001.parquet",
        "mmlu-test-00000.parquet",
    ),
    // GSM8K (~2.3 MB train + 0.4 MB test): 7.5K/1.3K grade-school math
    // problems with <<expr=result>> calculator (tool-call) annotations.
    (
        "openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet",
        "gsm8k-train-00000.parquet",
    ),
    (
        "openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
        "gsm8k-test-00000.parquet",
    ),
];

fn sft_jobs() -> Vec<FileJob> {
    let mut jobs: Vec<FileJob> = SFT_FILES
        .iter()
        .map(|(path, filename)| FileJob {
            url: format!("{HF}/{path}"),
            filename: (*filename).to_string(),
        })
        .collect();
    // 1000 synthetic conversations that give the model its nanochat persona.
    jobs.push(FileJob {
        url: IDENTITY_URL.to_string(),
        filename: "identity_conversations.jsonl".to_string(),
    });
    jobs
}

/// Fetch every SFT data file (~1.03 GB, dominated by SmolTalk) into `dir`.
/// Already-present files are skipped, so a rerun resumes after interruption.
pub fn download_sft_data(dir: &Path, workers: usize) -> io::Result<Summary> {
    download_files(dir, &sft_jobs(), workers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_is_ten_unique_wellformed_files() {
        let jobs = sft_jobs();
        assert_eq!(jobs.len(), 10);

        let mut names: Vec<&str> = jobs.iter().map(|j| j.filename.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 10, "local filenames must be unique");

        let mut urls: Vec<&str> = jobs.iter().map(|j| j.url.as_str()).collect();
        urls.sort_unstable();
        urls.dedup();
        assert_eq!(urls.len(), 10, "URLs must be unique");

        for j in &jobs {
            assert!(j.url.starts_with("https://"), "bad URL: {}", j.url);
            assert!(
                !j.filename.contains('/'),
                "filename must be a bare name: {}",
                j.filename
            );
        }
    }

    /// The per-dataset filename prefixes the parsers will glob by.
    #[test]
    fn manifest_covers_every_dataset_split() {
        let jobs = sft_jobs();
        let count = |prefix: &str| {
            jobs.iter()
                .filter(|j| j.filename.starts_with(prefix) && j.filename.ends_with(".parquet"))
                .count()
        };
        assert_eq!(count("smoltalk-train-"), 4);
        assert_eq!(count("smoltalk-test-"), 1);
        assert_eq!(count("mmlu-train-"), 1);
        assert_eq!(count("mmlu-test-"), 1);
        assert_eq!(count("gsm8k-train-"), 1);
        assert_eq!(count("gsm8k-test-"), 1);
        assert!(
            jobs.iter()
                .any(|j| j.filename == "identity_conversations.jsonl")
        );
    }
}
