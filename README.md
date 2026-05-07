# rs-nanogpt

## Training a BPE tokenizer

Train a tokenizer from a directory of parquet files (each containing a `text` column) and write a tiktoken-format vocabulary to disk:

```sh
cargo run --release -- train-tokenizer \
    --corpus data \
    --output vocab.txt \
    --vocab-size 512 \
    --max-chars 10000
```

Flags:

| Flag | Description |
|---|---|
| `--corpus` | Directory of `.parquet` files. Each file must have a `text` column. |
| `--output` | Path where the vocabulary will be written. |
| `--vocab-size` | Target vocabulary size. Must be at least 256. Default: 512. |
| `--max-chars` | Maximum number of bytes to read from the corpus. |

The output file is in tiktoken format — one token per line, `<base64-encoded-bytes> <rank>`, with the 256 single-byte tokens at ranks 1–256 and learned merges at ranks 257+.

Run `cargo run -- train-tokenizer --help` for the full flag list.
