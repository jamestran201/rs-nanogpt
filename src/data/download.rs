use std::fs;
use std::io;
use std::path::Path;
use std::thread;
use std::time::Duration;

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use ureq::Agent;

pub const BASE_URL: &str =
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main";

/// Highest shard index in the dataset: the last file is `shard_06542.parquet`,
/// so there are 6543 shards (indices `0..=MAX_SHARD`).
pub const MAX_SHARD: usize = 6542;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const RECV_RESPONSE_TIMEOUT: Duration = Duration::from_secs(30);
const RECV_BODY_TIMEOUT: Duration = Duration::from_secs(300);

fn shard_filename(index: usize) -> String {
    format!("shard_{index:05}.parquet")
}

struct RetryPolicy {
    max_attempts: u32,
    backoff_base: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            backoff_base: Duration::from_secs(1),
        }
    }
}

enum Outcome {
    Downloaded,
    Skipped,
    Failed,
}

pub struct Summary {
    pub requested: usize,
    pub downloaded: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// One file to fetch: an absolute `url` and the local `filename` it is saved
/// under. The two need not agree — the SFT manifest renames upstream
/// `train-00000-of-00004.parquet` to `smoltalk-train-00000.parquet`.
#[derive(Debug)]
pub struct FileJob {
    pub url: String,
    pub filename: String,
}

fn shard_indices(start: usize, num: usize, val_shard: Option<usize>) -> Vec<usize> {
    let range = start..start + num;
    let mut indices: Vec<usize> = range.clone().collect();
    if let Some(v) = val_shard
        && !range.contains(&v)
    {
        indices.push(v);
    }
    indices
}

fn build_agent() -> Agent {
    Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RECV_RESPONSE_TIMEOUT))
        .timeout_recv_body(Some(RECV_BODY_TIMEOUT))
        .build()
        .into()
}

fn try_download(agent: &Agent, url: &str, tmp_path: &Path, final_path: &Path) -> io::Result<()> {
    let response = agent.get(url).call().map_err(io::Error::other)?;
    let mut reader = response.into_body().into_reader();
    let mut tmp = fs::File::create(tmp_path)?;
    io::copy(&mut reader, &mut tmp)?;
    tmp.sync_all()?;
    drop(tmp);
    fs::rename(tmp_path, final_path)
}

fn download_one(agent: &Agent, dir: &Path, job: &FileJob, policy: &RetryPolicy) -> Outcome {
    let filename = job.filename.as_str();
    let final_path = dir.join(filename);
    if final_path.exists() {
        println!("skip     {filename} (already present)");
        return Outcome::Skipped;
    }

    let tmp_path = dir.join(format!("{filename}.tmp"));

    println!("get      {filename}");
    for attempt in 1..=policy.max_attempts {
        match try_download(agent, &job.url, &tmp_path, &final_path) {
            Ok(()) => {
                println!("done     {filename}");
                return Outcome::Downloaded;
            }
            Err(e) => {
                // Drop any partial file so a retry (or a later run) starts clean.
                let _ = fs::remove_file(&tmp_path);
                if attempt < policy.max_attempts {
                    let wait = policy.backoff_base * 2u32.pow(attempt);
                    eprintln!(
                        "retry    {filename}: attempt {attempt}/{} failed ({e}); waiting {}s",
                        policy.max_attempts,
                        wait.as_secs()
                    );
                    if !wait.is_zero() {
                        thread::sleep(wait);
                    }
                } else {
                    eprintln!(
                        "failed   {filename}: {e} (after {} attempts)",
                        policy.max_attempts
                    );
                }
            }
        }
    }
    Outcome::Failed
}

/// Fetch ClimbMix pretraining shards `[start, start+num)` plus the pinned val
/// shard, via [`download_files`].
pub fn download_shards(
    dir: &Path,
    start: usize,
    num: usize,
    val_shard: Option<usize>,
    workers: usize,
    base_url: &str,
) -> io::Result<Summary> {
    let jobs: Vec<FileJob> = shard_indices(start, num, val_shard)
        .into_iter()
        .map(|index| {
            let filename = shard_filename(index);
            FileJob {
                url: format!("{}/{}", base_url.trim_end_matches('/'), filename),
                filename,
            }
        })
        .collect();
    download_files(dir, &jobs, workers)
}

/// Fetch `jobs` into `dir` in parallel (`workers` threads), skipping files
/// already present. Each file streams to `<filename>.tmp` and is renamed into
/// place only when complete, so an interrupted run never leaves a truncated
/// file under its final name. Failures retry with exponential backoff before
/// counting toward `Summary::failed`.
pub fn download_files(dir: &Path, jobs: &[FileJob], workers: usize) -> io::Result<Summary> {
    download_files_with_policy(dir, jobs, workers, &RetryPolicy::default())
}

/// [`download_files`] with an explicit retry policy, so tests can hit the
/// failure path without the default's ~30s of backoff sleeps.
fn download_files_with_policy(
    dir: &Path,
    jobs: &[FileJob],
    workers: usize,
    policy: &RetryPolicy,
) -> io::Result<Summary> {
    fs::create_dir_all(dir)?;

    let requested = jobs.len();
    println!(
        "downloading {requested} file(s) to {} with {workers} worker(s)",
        dir.display()
    );

    let agent = build_agent();
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(io::Error::other)?;

    let outcomes: Vec<Outcome> = pool.install(|| {
        jobs.par_iter()
            .map(|job| download_one(&agent, dir, job, policy))
            .collect()
    });

    let mut summary = Summary {
        requested,
        downloaded: 0,
        skipped: 0,
        failed: 0,
    };
    for outcome in outcomes {
        match outcome {
            Outcome::Downloaded => summary.downloaded += 1,
            Outcome::Skipped => summary.skipped += 1,
            Outcome::Failed => summary.failed += 1,
        }
    }

    println!(
        "downloaded {}/{} ({} skipped, {} failed) -> {}",
        summary.downloaded,
        summary.requested,
        summary.skipped,
        summary.failed,
        dir.display()
    );
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    /// A throwaway HTTP/1.1 server that serves a fixed `path -> bytes` table and
    /// 404s everything else, so tests exercise the real `ureq` path without the
    /// network. Returns its base URL (e.g. `http://127.0.0.1:54321`); the
    /// accept-loop thread is detached and dies with the test process.
    fn serve(routes: Vec<(String, Vec<u8>)>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                // GET requests are small; the request line (with the path) is in
                // the first read, which is all we need to route.
                let mut buf = [0u8; 2048];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("");
                match routes.iter().find(|(p, _)| p == path) {
                    Some((_, bytes)) => {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            bytes.len()
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(bytes);
                    }
                    None => {
                        let _ = stream.write_all(
                            b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        );
                    }
                }
            }
        });
        format!("http://{addr}")
    }

    fn fast_policy() -> RetryPolicy {
        // No real waiting in tests; keep a retry so the loop structure is exercised.
        RetryPolicy {
            max_attempts: 2,
            backoff_base: Duration::ZERO,
        }
    }

    fn job(url: String, filename: &str) -> FileJob {
        FileJob {
            url,
            filename: filename.to_string(),
        }
    }

    #[test]
    fn shard_filename_zero_pads_to_five_digits() {
        assert_eq!(shard_filename(0), "shard_00000.parquet");
        assert_eq!(shard_filename(42), "shard_00042.parquet");
        assert_eq!(shard_filename(MAX_SHARD), "shard_06542.parquet");
    }

    #[test]
    fn shard_indices_appends_pinned_val_and_dedups() {
        assert_eq!(shard_indices(5, 3, Some(6542)), vec![5, 6, 7, 6542]);
        assert_eq!(shard_indices(5, 3, None), vec![5, 6, 7]);
        // val index already inside [0, 2) must not be duplicated
        assert_eq!(shard_indices(0, 2, Some(1)), vec![0, 1]);
    }

    #[test]
    fn download_one_writes_file_and_removes_tmp() {
        let dir = tempfile::tempdir().unwrap();
        let base = serve(vec![(
            "/shard_00005.parquet".into(),
            b"hello-shard".to_vec(),
        )]);
        let agent = build_agent();

        let j = job(format!("{base}/shard_00005.parquet"), "shard_00005.parquet");
        let outcome = download_one(&agent, dir.path(), &j, &fast_policy());
        assert!(matches!(outcome, Outcome::Downloaded));

        let path = dir.path().join("shard_00005.parquet");
        assert_eq!(fs::read(&path).unwrap(), b"hello-shard");
        assert!(!dir.path().join("shard_00005.parquet.tmp").exists());
    }

    #[test]
    fn download_one_skips_existing_without_fetching() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard_00005.parquet");
        fs::write(&path, b"original").unwrap();
        // Server would serve *different* bytes; a skip must not overwrite.
        let base = serve(vec![("/shard_00005.parquet".into(), b"NEW".to_vec())]);
        let agent = build_agent();

        let j = job(format!("{base}/shard_00005.parquet"), "shard_00005.parquet");
        let outcome = download_one(&agent, dir.path(), &j, &RetryPolicy::default());
        assert!(matches!(outcome, Outcome::Skipped));
        assert_eq!(fs::read(&path).unwrap(), b"original");
    }

    #[test]
    fn download_one_fails_after_retries_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let base = serve(vec![]); // 404 everything
        let agent = build_agent();

        let j = job(format!("{base}/shard_00005.parquet"), "shard_00005.parquet");
        let outcome = download_one(&agent, dir.path(), &j, &fast_policy());
        assert!(matches!(outcome, Outcome::Failed));
        assert!(!dir.path().join("shard_00005.parquet").exists());
        assert!(!dir.path().join("shard_00005.parquet.tmp").exists());
    }

    /// One job per outcome kind: a renamed fresh file (downloaded), a
    /// pre-existing one the server's newer bytes must not overwrite (skipped),
    /// and a 404 (failed).
    #[test]
    fn download_files_saves_under_local_names_and_counts_outcomes() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("already.bin"), b"original").unwrap();
        let base = serve(vec![
            (
                "/data/train-00000-of-00004.parquet".into(),
                b"fresh".to_vec(),
            ),
            ("/already.bin".into(), b"NEW".to_vec()),
        ]);

        let jobs = vec![
            job(
                format!("{base}/data/train-00000-of-00004.parquet"),
                "smoltalk-train-00000.parquet",
            ),
            job(format!("{base}/already.bin"), "already.bin"),
            job(format!("{base}/missing.bin"), "missing.bin"),
        ];
        let summary = download_files_with_policy(dir.path(), &jobs, 2, &fast_policy()).unwrap();

        assert_eq!(summary.requested, 3);
        assert_eq!(summary.downloaded, 1);
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.failed, 1);
        assert_eq!(
            fs::read(dir.path().join("smoltalk-train-00000.parquet")).unwrap(),
            b"fresh"
        );
        assert_eq!(
            fs::read(dir.path().join("already.bin")).unwrap(),
            b"original"
        );
        assert!(!dir.path().join("missing.bin").exists());
    }

    #[test]
    fn download_shards_fetches_range_plus_pinned_val() {
        let dir = tempfile::tempdir().unwrap();
        let base = serve(vec![
            ("/shard_00005.parquet".into(), b"a".to_vec()),
            ("/shard_00006.parquet".into(), b"b".to_vec()),
            ("/shard_00007.parquet".into(), b"c".to_vec()),
            ("/shard_06542.parquet".into(), b"v".to_vec()),
        ]);

        let summary = download_shards(dir.path(), 5, 3, Some(6542), 2, &base).unwrap();
        assert_eq!(summary.requested, 4);
        assert_eq!(summary.downloaded, 4);
        assert_eq!(summary.failed, 0);
        for name in [
            "shard_00005.parquet",
            "shard_00006.parquet",
            "shard_00007.parquet",
            "shard_06542.parquet",
        ] {
            assert!(dir.path().join(name).exists(), "missing {name}");
        }
    }
}
