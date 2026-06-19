# rs-nanogpt

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
