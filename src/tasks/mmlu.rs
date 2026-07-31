//! MMLU (`cais/mmlu`) — multiple-choice questions, 100K auxiliary-train /
//! 14K test rows. Port of nanochat's `tasks/mmlu.py` plus `render_mc`
//! (`tasks/common.py:112`): each row becomes a two-message [`Conversation`],
//! the rendered question as the user turn and the answer letter as the
//! assistant turn. The `subject` column is eval-only and ignored.

use std::io;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};

use super::{Split, invalid, load_dataset};
use crate::tokenizer::{Content, Conversation, Message, Role};

const LETTERS: [&str; 4] = ["A", "B", "C", "D"];

/// nanochat's multiple-choice prompt format, ported verbatim. Both details are
/// small-model token-binding tricks: the letter comes *after* the choice, and
/// nothing separates it from the `=`, so the prompt's "A" is the same token as
/// the assistant's reply "A" (" A" would be a different one).
fn render_mc(question: &str, choices: &[&str]) -> String {
    let mut query = format!("Multiple Choice question: {question}\n");
    for (choice, letter) in choices.iter().zip(LETTERS) {
        query.push_str(&format!("- {choice}={letter}\n"));
    }
    query.push_str("\nRespond only with the letter of the correct answer.");
    query
}

/// Load every `mmlu-{split}-*.parquet` file in `dir`. Validates 4 choices and
/// `answer ∈ 0..=3` per row; errors carry file + row context.
pub fn load(dir: &Path, split: Split) -> io::Result<Vec<Conversation>> {
    load_dataset(dir, "mmlu", split, decode_batch)
}

fn decode_batch(
    batch: &RecordBatch,
    path: &Path,
    first_row: usize,
) -> io::Result<Vec<Conversation>> {
    let schema_err = |what: String| invalid(format!("{}: {what}", path.display()));
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .ok_or_else(|| schema_err(format!("missing '{name}' column")))
    };
    let questions = column("question")?
        .as_string_opt::<i32>()
        .ok_or_else(|| schema_err("'question' column is not a string".into()))?;
    let choice_lists = column("choices")?
        .as_list_opt::<i32>()
        .ok_or_else(|| schema_err("'choices' column is not a list".into()))?;
    let choice_values = choice_lists
        .values()
        .as_string_opt::<i32>()
        .ok_or_else(|| schema_err("'choices' items are not strings".into()))?;
    let answers = column("answer")?
        .as_primitive_opt::<Int64Type>()
        .ok_or_else(|| schema_err("'answer' column is not int64".into()))?;

    let offsets = choice_lists.value_offsets();
    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let row_err =
            |what: String| invalid(format!("{}: row {}: {what}", path.display(), first_row + i));
        if questions.is_null(i) || choice_lists.is_null(i) || answers.is_null(i) {
            return Err(row_err("null field".into()));
        }
        let (start, end) = (offsets[i] as usize, offsets[i + 1] as usize);
        let mut choices = Vec::with_capacity(end - start);
        for j in start..end {
            if choice_values.is_null(j) {
                return Err(row_err("null choice".into()));
            }
            choices.push(choice_values.value(j));
        }
        if choices.len() != 4 {
            return Err(row_err(format!(
                "expected 4 choices, got {}",
                choices.len()
            )));
        }
        let answer = answers.value(i);
        let letter = usize::try_from(answer)
            .ok()
            .and_then(|a| LETTERS.get(a))
            .ok_or_else(|| row_err(format!("answer {answer} out of range 0..=3")))?;
        out.push(Conversation {
            messages: vec![
                Message {
                    role: Role::User,
                    content: Content::Text(render_mc(questions.value(i), &choices)),
                },
                Message {
                    role: Role::Assistant,
                    content: Content::Text((*letter).to_string()),
                },
            ],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_mmlu_shard;

    fn write_row(dir: &Path, choices: &[&str], answer: i64) {
        write_mmlu_shard(
            &dir.join("mmlu-train-00000.parquet"),
            &[("What is 2+2?", choices, answer)],
        );
    }

    /// The user message matches a hand-written `render_mc` output exactly,
    /// and `answer=1` maps to the letter "B".
    #[test]
    fn renders_mc_verbatim() {
        let dir = tempfile::tempdir().unwrap();
        write_row(dir.path(), &["3", "4", "5", "6"], 1);

        let convs = load(dir.path(), Split::Train).unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages[0].content,
            Content::Text(
                "Multiple Choice question: What is 2+2?\n\
                 - 3=A\n\
                 - 4=B\n\
                 - 5=C\n\
                 - 6=D\n\
                 \nRespond only with the letter of the correct answer."
                    .into()
            )
        );
        assert_eq!(convs[0].messages[1].content, Content::Text("B".into()));
    }

    #[test]
    fn wrong_choice_arity_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_row(dir.path(), &["3", "4", "5"], 0);

        let err = load(dir.path(), Split::Train).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("row 0") && msg.contains("expected 4 choices, got 3"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn out_of_range_answer_errors() {
        let dir = tempfile::tempdir().unwrap();
        write_row(dir.path(), &["3", "4", "5", "6"], 4);

        let err = load(dir.path(), Split::Train).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("answer 4 out of range"),
            "unexpected error: {err}"
        );
    }
}
