# rs-nanogpt

## Downloading the dataset

Download pretraining shards from the [ClimbMix-400B](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) dataset into a local directory. Each shard is `shard_NNNNN.parquet` (~100 MB, ~250M characters); the dataset has 6543 shards (indices 0–6542).

```sh
cargo run --release -- download-data \
    --out data \
    --num 170
```

This fetches the first 170 shards (`shard_00000`–`shard_00169`) as training data, plus the pinned validation shard `shard_06542`. Because that shard sorts last by filename, the `pretrain`/`train-tokenizer` commands automatically hold it out as the validation split while keeping the whole `--start`/`--num` range as training data. Downloads run in parallel, skip files already on disk (so re-running resumes), and stream to a temp file before an atomic rename, so an interrupted run never leaves a half-written shard.

To extend a partial download later, move `--start` forward — the pinned val shard is already present and is skipped:

```sh
cargo run --release -- download-data --start 170 --num 30
```

Flags:

| Flag | Description |
|---|---|
| `--out` | Directory shards are written to (created if missing). Default: `data`. |
| `--start` | First shard index to download. Default: 0. |
| `--num` | Number of shards to download in total, starting at `--start`. Required. |
| `--workers` | Number of parallel downloads. Default: 4. |
| `--val-shard` | Validation shard to also fetch; it pins the val split. Pass `none` to download only the `--start`/`--num` range. Default: 6542. |
| `--base-url` | Base URL the shards are fetched from. Defaults to the ClimbMix-400B host. |

Run `cargo run -- download-data --help` for the full flag list.

## Training a BPE tokenizer

Train a tokenizer from a directory of parquet files (each containing a `text` column) and write a tiktoken-format vocabulary to disk:

```sh
cargo run --release -- train-tokenizer \
    --corpus data \
    --output vocab.txt \
    --vocab-size 32768 \
    --max-chars 2000000000 \
    --doc-cap 10000
```

Flags:

| Flag | Description |
|---|---|
| `--corpus` | Directory of `.parquet` files. Each file must have a `text` column. |
| `--output` | Path where the vocabulary will be written. |
| `--vocab-size` | Target vocabulary size. Must be at least 256. Default: 512. |
| `--max-chars` | Maximum number of bytes to read from the corpus. |
| `--doc-cap` | Maximum bytes per document; longer documents are truncated at a UTF-8 char boundary so a few unusually long documents can't dominate BPE pair statistics. Default: 10000. |

The output file is in tiktoken format — one token per line, `<base64-encoded-bytes> <rank>`, with the 256 single-byte tokens at ranks 0–255 and learned merges at ranks 256+.

Run `cargo run -- train-tokenizer --help` for the full flag list.

## Evaluating a tokenizer

Load a trained vocabulary, encode and decode a small set of built-in text fixtures, and print compression ratios:

```sh
cargo run --release -- eval-tokenizer --vocab target/vocab.txt
```

Example output:

```
fixture         bytes     tokens    bytes/token  round_trip
en               2431       1189          2.045          ok
korean            585        583          1.003          ok
code              372        309          1.204          ok
```

Each fixture is encoded then decoded; `round_trip` is `ok` when the decoded bytes match the input exactly. `bytes/token` is the compression ratio — higher is better.

The fixtures (English prose, Korean, and a code snippet) are embedded into the binary and live under `tests/fixtures/eval/`.

## Pretraining a model

Train a GPT model on a directory of parquet shards, using a trained tokenizer to encode the text and to size the model's vocabulary:

```sh
cargo run --release -- pretrain \
    --data data \
    --vocab vocab.txt \
    --num-iters 5000 \
    --out out
```

The shards in `--data` are split by filename: the last shard (sorted by name) is held out as the validation set and the rest form the training set. `--vocab` is the tiktoken-format file produced by `train-tokenizer`; it doubles as the tokenizer and fixes the model's embedding / `lm_head` size, so there is no separate `--vocab-size` flag.

The defaults describe a "Mac smoke" configuration — a depth-6, 384-wide model at sequence length 512 — small enough to run on a laptop.

Key flags:

| Flag | Description |
|---|---|
| `--data` | Directory of `.parquet` shards. The last shard (sorted by name) is the validation split; the rest are training data. |
| `--vocab` | Tiktoken-format vocabulary from `train-tokenizer`. Sets both the tokenizer and the model's vocab size. |
| `--num-iters` | Number of optimizer steps (training horizon). Default: 5000. |
| `--device-batch` | Rows per forward pass (B). Memory-limited; peak memory scales linearly with it (attention is chunked/flash, so no B·T² term) — reduce if you run out of memory. Default: 32. |
| `--total-batch` | Tokens per optimizer step. Must be a multiple of `device_batch × sequence_len`; the quotient is the gradient-accumulation count. Default: 16384. |
| `--sequence-len` | Context length in tokens (T). Default: 512. |
| `--n-layer` / `--n-head` / `--n-embd` | Transformer depth, attention heads, and residual width (`n_embd` must be divisible by `n_head`). Defaults: 6 / 6 / 384. |
| `--embedding-lr` / `--unembedding-lr` / `--matrix-lr` | Separate AdamW learning rates for the token embedding, the unembedding (`lm_head`), and the transformer block matrices. Defaults: 0.2 / 0.004 / 0.003. |
| `--eval-every` | Compute validation loss/bpb every N steps and checkpoint the best model (0 disables). Default: 250. |
| `--sample-every` | Generate sample text from fixed prompts every N steps (0 disables). Default: 0. |
| `--out` | Output directory for telemetry and checkpoints. Default: `out`. |

Run `cargo run -- pretrain --help` for the full flag list (LR warmup/warmdown schedule, eval/sample sizing, RoPE base, RMSNorm epsilon).

During training the loop prints a per-step line and, at the `--eval-every` cadence, a validation line (loss/grad-norm/throughput values below are illustrative):

```
step      0  val_loss 9.9821  bpb 2.8410
step      0/5000 | loss 10.4213 | gnorm 1.284 | t+00:00:02
step     10/5000 | loss 9.6042 | gnorm 0.911 | lr m=7.50e-4 e=5.00e-2 u=1.00e-3 | 5300 tok/s | 31 ms/step | t+00:00:03 | eta 00:02:34
step   1000/5000 | loss 4.2170 | gnorm 0.318 | lr m=3.00e-3 e=2.00e-1 u=4.00e-3 | 5300 tok/s | 31 ms/step | t+00:00:33 | eta 00:02:10
step   1000  val_loss 4.0521  bpb 1.1874
```

`gnorm` is the pre-optimizer gradient L2 norm (an early-warning signal for divergence); `lr m/e/u` are the current matrix / embedding / unembedding learning rates after the schedule multiplier; `bpb` is bits-per-byte on the validation set. A non-finite loss aborts the run.

Each run writes three things under `--out`:

| Path | Contents |
|---|---|
| `run.json` | Run metadata: model config, batch sizing, learning rates, and start time. |
| `metrics.jsonl` | Append-only per-step / per-eval telemetry (loss, grad norm, throughput, val bpb). |
| `best/` | Checkpoint of the lowest-val-bpb model so far (`model.safetensors` + `meta.txt`), overwritten each time a new best is reached. |
