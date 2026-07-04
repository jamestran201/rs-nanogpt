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

fn download_one(
    agent: &Agent,
    dir: &Path,
    index: usize,
    base_url: &str,
    policy: &RetryPolicy,
) -> Outcome {
    let filename = shard_filename(index);
    let final_path = dir.join(&filename);
    if final_path.exists() {
        println!("skip     {filename} (already present)");
        return Outcome::Skipped;
    }

    let url = format!("{}/{}", base_url.trim_end_matches('/'), filename);
    let tmp_path = dir.join(format!("{filename}.tmp"));

    println!("get      {filename}");
    for attempt in 1..=policy.max_attempts {
        match try_download(agent, &url, &tmp_path, &final_path) {
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

pub fn download_shards(
    dir: &Path,
    start: usize,
    num: usize,
    val_shard: Option<usize>,
    workers: usize,
    base_url: &str,
) -> io::Result<Summary> {
    fs::create_dir_all(dir)?;

    let indices = shard_indices(start, num, val_shard);
    let requested = indices.len();
    println!(
        "downloading {requested} shard(s) to {} with {workers} worker(s)",
        dir.display()
    );

    let agent = build_agent();
    let policy = RetryPolicy::default();
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(io::Error::other)?;

    let outcomes: Vec<Outcome> = pool.install(|| {
        indices
            .par_iter()
            .map(|&index| download_one(&agent, dir, index, base_url, &policy))
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

        let outcome = download_one(&agent, dir.path(), 5, &base, &fast_policy());
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

        let outcome = download_one(&agent, dir.path(), 5, &base, &RetryPolicy::default());
        assert!(matches!(outcome, Outcome::Skipped));
        assert_eq!(fs::read(&path).unwrap(), b"original");
    }

    #[test]
    fn download_one_fails_after_retries_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let base = serve(vec![]); // 404 everything
        let agent = build_agent();

        let outcome = download_one(&agent, dir.path(), 5, &base, &fast_policy());
        assert!(matches!(outcome, Outcome::Failed));
        assert!(!dir.path().join("shard_00005.parquet").exists());
        assert!(!dir.path().join("shard_00005.parquet.tmp").exists());
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
