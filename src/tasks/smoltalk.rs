//! SmolTalk (`HuggingFaceTB/smol-smoltalk`) — the general conversational
//! dataset, 460K train / 24K test rows. Port of nanochat's
//! `tasks/smoltalk.py`: each row's `messages` column (arrow
//! `list<struct<content, role>>`) becomes one [`Conversation`]; the `source`
//! column is ignored.

use std::io;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ListArray, RecordBatch, StringArray, StructArray};

use super::{Split, invalid, load_dataset, role_from_str};
use crate::tokenizer::{Content, Conversation, Message};

/// Load every `smoltalk-{split}-*.parquet` file in `dir`. Errors carry file +
/// row context; any null is data corruption.
pub fn load(dir: &Path, split: Split) -> io::Result<Vec<Conversation>> {
    load_dataset(dir, "smoltalk", split, decode_batch)
}

/// The `messages` column decoded once per batch: the list array plus its
/// struct children as plain string arrays.
struct MessagesColumn<'a> {
    lists: &'a ListArray,
    structs: &'a StructArray,
    roles: &'a StringArray,
    contents: &'a StringArray,
}

fn messages_column<'a>(batch: &'a RecordBatch, path: &Path) -> io::Result<MessagesColumn<'a>> {
    let schema_err = |what: &str| invalid(format!("{}: {what}", path.display()));
    let lists = batch
        .column_by_name("messages")
        .ok_or_else(|| schema_err("missing 'messages' column"))?
        .as_list_opt::<i32>()
        .ok_or_else(|| schema_err("'messages' column is not a list"))?;
    let structs = lists
        .values()
        .as_struct_opt()
        .ok_or_else(|| schema_err("'messages' items are not structs"))?;
    let child = |name: &str| {
        structs
            .column_by_name(name)
            .and_then(|c| c.as_string_opt::<i32>())
            .ok_or_else(|| schema_err(&format!("'messages' structs lack a string '{name}' field")))
    };
    Ok(MessagesColumn {
        lists,
        structs,
        roles: child("role")?,
        contents: child("content")?,
    })
}

fn decode_batch(
    batch: &RecordBatch,
    path: &Path,
    first_row: usize,
) -> io::Result<Vec<Conversation>> {
    let col = messages_column(batch, path)?;
    let offsets = col.lists.value_offsets();
    let mut out = Vec::with_capacity(col.lists.len());
    for i in 0..col.lists.len() {
        let row_err =
            |what: String| invalid(format!("{}: row {}: {what}", path.display(), first_row + i));
        if col.lists.is_null(i) {
            return Err(row_err("null 'messages' entry".into()));
        }
        let (start, end) = (offsets[i] as usize, offsets[i + 1] as usize);
        let mut messages = Vec::with_capacity(end - start);
        for j in start..end {
            if col.structs.is_null(j) || col.roles.is_null(j) || col.contents.is_null(j) {
                return Err(row_err(format!("null message at index {}", j - start)));
            }
            let role_str = col.roles.value(j);
            let role = role_from_str(role_str)
                .ok_or_else(|| row_err(format!("unknown role {role_str:?}")))?;
            messages.push(Message {
                role,
                content: Content::Text(col.contents.value(j).to_string()),
            });
        }
        out.push(Conversation { messages });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_smoltalk_shard;
    use crate::tokenizer::Role;

    /// A system+user+assistant row — the real-data shape — decodes to the
    /// exact `Conversation`.
    #[test]
    fn decodes_exact_conversation() {
        let dir = tempfile::tempdir().unwrap();
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00000.parquet"),
            &[&[("system", "S"), ("user", "U"), ("assistant", "A")]],
        );

        let convs = load(dir.path(), Split::Train).unwrap();
        assert_eq!(convs.len(), 1);
        let text = |role, s: &str| Message {
            role,
            content: Content::Text(s.into()),
        };
        assert_eq!(
            convs[0],
            Conversation {
                messages: vec![
                    text(Role::System, "S"),
                    text(Role::User, "U"),
                    text(Role::Assistant, "A"),
                ]
            }
        );
    }

    /// Multiple files load in sorted filename order, and only files matching
    /// the requested split's prefix are read.
    #[test]
    fn walks_split_files_sorted() {
        let dir = tempfile::tempdir().unwrap();
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00001.parquet"),
            &[&[("user", "second")]],
        );
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00000.parquet"),
            &[&[("user", "first")]],
        );
        write_smoltalk_shard(
            &dir.path().join("smoltalk-test-00000.parquet"),
            &[&[("user", "test split")]],
        );

        let convs = load(dir.path(), Split::Train).unwrap();
        let texts: Vec<_> = convs
            .iter()
            .map(|c| match &c.messages[0].content {
                Content::Text(t) => t.as_str(),
                Content::Parts(_) => unreachable!(),
            })
            .collect();
        assert_eq!(texts, ["first", "second"]);
    }

    /// The error names the file and the 0-based row within it.
    #[test]
    fn unknown_role_errors_with_context() {
        let dir = tempfile::tempdir().unwrap();
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00000.parquet"),
            &[&[("user", "ok")], &[("tool", "nope")]],
        );

        let err = load(dir.path(), Split::Train).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        for needle in ["smoltalk-train-00000.parquet", "row 1", "\"tool\""] {
            assert!(
                msg.contains(needle),
                "error {msg:?} should contain {needle:?}"
            );
        }
    }

    /// No matching files is a loud NotFound, not an empty dataset.
    #[test]
    fn missing_split_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_smoltalk_shard(
            &dir.path().join("smoltalk-train-00000.parquet"),
            &[&[("user", "u")]],
        );
        let err = load(dir.path(), Split::Test).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
