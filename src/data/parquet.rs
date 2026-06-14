use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use arrow_array::Array;
use arrow_array::cast::AsArray;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
}

pub(crate) fn list_shards(dir: &Path) -> io::Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotADirectory,
            format!("{} is not a directory", dir.display()),
        ));
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let extension = path.extension().and_then(|e| e.to_str());
            (extension == Some("parquet")).then_some(path)
        })
        .collect();
    files.sort();
    Ok(files)
}

pub(crate) fn shards_for_split(all: &[PathBuf], split: Split) -> io::Result<Vec<PathBuf>> {
    if all.len() < 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "need at least 2 shards to form train/val splits, found {}",
                all.len()
            ),
        ));
    }
    let chosen = match split {
        Split::Train => &all[..all.len() - 1],
        Split::Val => &all[all.len() - 1..],
    };
    Ok(chosen.to_vec())
}

pub(crate) struct ShardTextIter {
    files: std::vec::IntoIter<PathBuf>,
    state: State,
}

enum State {
    NeedFile,
    NeedBatch(ParquetRecordBatchReader),
    InBatch {
        reader: ParquetRecordBatchReader,
        batch: arrow_array::RecordBatch,
        row_idx: usize,
    },
    Done,
}

impl ShardTextIter {
    pub(crate) fn new(shards: Vec<PathBuf>) -> Self {
        Self {
            files: shards.into_iter(),
            state: State::NeedFile,
        }
    }
}

fn next_text_in_batch(
    batch: &arrow_array::RecordBatch,
    start: usize,
) -> io::Result<(Option<String>, usize)> {
    let Some(col) = batch.column_by_name("text") else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing 'text' column",
        ));
    };
    let strings = col.as_string::<i32>();
    let mut i = start;
    while i < strings.len() {
        let idx = i;
        i += 1;
        if strings.is_null(idx) {
            continue;
        }
        return Ok((Some(strings.value(idx).to_string()), i));
    }
    Ok((None, i))
}

impl Iterator for ShardTextIter {
    type Item = io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match std::mem::replace(&mut self.state, State::Done) {
                State::Done => return None,

                State::NeedFile => {
                    let path = self.files.next()?;
                    let file = match fs::File::open(&path) {
                        Ok(f) => f,
                        Err(e) => return Some(Err(e)),
                    };
                    let reader = match ParquetRecordBatchReaderBuilder::try_new(file)
                        .and_then(|b| b.build())
                    {
                        Ok(r) => r,
                        Err(e) => return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e))),
                    };
                    self.state = State::NeedBatch(reader);
                }

                State::NeedBatch(mut reader) => match reader.next() {
                    Some(Ok(batch)) => {
                        self.state = State::InBatch {
                            reader,
                            batch,
                            row_idx: 0,
                        };
                    }
                    Some(Err(e)) => {
                        return Some(Err(io::Error::new(io::ErrorKind::InvalidData, e)));
                    }
                    None => self.state = State::NeedFile,
                },

                State::InBatch {
                    reader,
                    batch,
                    row_idx,
                } => match next_text_in_batch(&batch, row_idx) {
                    Err(e) => return Some(Err(e)),
                    Ok((Some(text), next_idx)) => {
                        self.state = State::InBatch {
                            reader,
                            batch,
                            row_idx: next_idx,
                        };
                        return Some(Ok(text));
                    }
                    Ok((None, _)) => self.state = State::NeedBatch(reader),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use parquet::arrow::ArrowWriter;

    fn paths(names: &[&str]) -> Vec<PathBuf> {
        names.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn split_selects_all_but_last_and_last() {
        let all = paths(&["a.parquet", "b.parquet", "c.parquet"]);
        assert_eq!(
            shards_for_split(&all, Split::Train).unwrap(),
            paths(&["a.parquet", "b.parquet"])
        );
        assert_eq!(
            shards_for_split(&all, Split::Val).unwrap(),
            paths(&["c.parquet"])
        );
    }

    #[test]
    fn split_requires_at_least_two_shards() {
        let one = paths(&["only.parquet"]);
        let err = shards_for_split(&one, Split::Train).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    fn write_parquet(path: &Path, texts: Vec<Option<&str>>) {
        let arr: ArrayRef = Arc::new(StringArray::from(texts));
        let batch = RecordBatch::try_from_iter([("text", arr)]).unwrap();
        let file = fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn list_shards_returns_sorted_parquet_only() {
        let dir = tempfile::tempdir().unwrap();
        write_parquet(&dir.path().join("b.parquet"), vec![Some("x")]);
        write_parquet(&dir.path().join("a.parquet"), vec![Some("y")]);
        fs::write(dir.path().join("notes.txt"), "ignore me").unwrap();

        let shards = list_shards(dir.path()).unwrap();
        let names: Vec<_> = shards
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["a.parquet", "b.parquet"]);
    }

    #[test]
    fn list_shards_errors_on_non_directory() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let err = list_shards(f.path()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotADirectory);
    }

    #[test]
    fn shard_text_iter_streams_all_rows_in_order_skipping_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let s0 = dir.path().join("0.parquet");
        let s1 = dir.path().join("1.parquet");
        write_parquet(&s0, vec![Some("alpha"), None, Some("beta")]);
        write_parquet(&s1, vec![Some("gamma")]);

        let docs: Vec<String> = ShardTextIter::new(vec![s0, s1])
            .collect::<io::Result<_>>()
            .unwrap();
        assert_eq!(docs, vec!["alpha", "beta", "gamma"]);
    }
}
