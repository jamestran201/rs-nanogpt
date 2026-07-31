//! GSM8K (`openai/gsm8k`, `main` subset) — grade-school math word problems,
//! 7.5K train / 1.3K test rows. Port of nanochat's `tasks/gsm8k.py:52-85`:
//! the answer text carries `<<expr=result>>` calculator annotations, which
//! become [`Part::Python`] + [`Part::PythonOutput`] tool-call spans in a
//! structured assistant message — the dataset that teaches the model to emit
//! tool calls. The trailing `#### N` answer marker stays in the final text
//! part; `extract_answer`/`evaluate` are eval-only and not ported.

use std::io;
use std::path::Path;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, StringArray};
use fancy_regex::Regex;

use super::{Split, invalid, load_dataset};
use crate::tokenizer::{Content, Conversation, Message, Part, Role};

/// Split an answer on `<<expr=result>>` calculator calls. Text between calls
/// becomes `Part::Text`; each call's inner text splits on the *last* `=` into
/// expression and result (no `=` → the whole inner text is the expression,
/// empty output). Unlike nanochat we drop empty text segments, which Python's
/// `re.split` keeps; they encode to zero tokens, so renders match either way.
fn split_answer(call_re: &Regex, answer: &str) -> io::Result<Vec<Part>> {
    let mut parts = Vec::new();
    let mut last_end = 0usize;
    let push_text = |parts: &mut Vec<Part>, text: &str| {
        if !text.is_empty() {
            parts.push(Part::Text(text.to_string()));
        }
    };
    for hit in call_re.find_iter(answer) {
        let hit = hit.map_err(|e| invalid(format!("calculator-call regex failed: {e}")))?;
        push_text(&mut parts, &answer[last_end..hit.start()]);
        let inner = &answer[hit.start() + 2..hit.end() - 2];
        let (expr, result) = inner.rsplit_once('=').unwrap_or((inner, ""));
        parts.push(Part::Python(expr.to_string()));
        parts.push(Part::PythonOutput(result.to_string()));
        last_end = hit.end();
    }
    push_text(&mut parts, &answer[last_end..]);
    Ok(parts)
}

/// Load every `gsm8k-{split}-*.parquet` file in `dir`. Each row's question
/// becomes the user turn and its annotated answer the structured assistant
/// turn.
pub fn load(dir: &Path, split: Split) -> io::Result<Vec<Conversation>> {
    let call_re = Regex::new(r"<<[^>]+>>").expect("valid literal regex");
    load_dataset(dir, "gsm8k", split, |batch, path, first_row| {
        decode_batch(batch, &call_re, path, first_row)
    })
}

fn decode_batch(
    batch: &RecordBatch,
    call_re: &Regex,
    path: &Path,
    first_row: usize,
) -> io::Result<Vec<Conversation>> {
    let string_column = |name: &str| -> io::Result<&StringArray> {
        batch
            .column_by_name(name)
            .and_then(|c| c.as_string_opt::<i32>())
            .ok_or_else(|| {
                invalid(format!(
                    "{}: missing string '{name}' column",
                    path.display()
                ))
            })
    };
    let questions = string_column("question")?;
    let answers = string_column("answer")?;
    let mut out = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let row_err =
            |what: &str| invalid(format!("{}: row {}: {what}", path.display(), first_row + i));
        if questions.is_null(i) || answers.is_null(i) {
            return Err(row_err("null field"));
        }
        let parts = split_answer(call_re, answers.value(i)).map_err(|e| row_err(&e.to_string()))?;
        out.push(Conversation {
            messages: vec![
                Message {
                    role: Role::User,
                    content: Content::Text(questions.value(i).to_string()),
                },
                Message {
                    role: Role::Assistant,
                    content: Content::Parts(parts),
                },
            ],
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::write_gsm8k_shard;

    fn parts(answer: &str) -> Vec<Part> {
        let re = Regex::new(r"<<[^>]+>>").unwrap();
        split_answer(&re, answer).unwrap()
    }

    fn text(s: &str) -> Part {
        Part::Text(s.into())
    }
    fn python(s: &str) -> Part {
        Part::Python(s.into())
    }
    fn output(s: &str) -> Part {
        Part::PythonOutput(s.into())
    }

    /// The `gsm8k.py:8-12` docstring example, asserted part by part.
    #[test]
    fn docstring_example_splits_exactly() {
        let answer = "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n\
                      Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n\
                      #### 10";
        assert_eq!(
            parts(answer),
            vec![
                text("Weng earns 12/60 = $"),
                python("12/60"),
                output("0.2"),
                text("0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $"),
                python("0.2*50"),
                output("10"),
                text("10.\n#### 10"),
            ]
        );
    }

    /// A call without `=` keeps the whole inner text as the expression and
    /// yields an empty output.
    #[test]
    fn call_without_equals() {
        assert_eq!(
            parts("a<<1+1>>b"),
            vec![text("a"), python("1+1"), output(""), text("b")]
        );
    }

    #[test]
    fn no_calls_single_text_part() {
        assert_eq!(
            parts("just prose\n#### 3"),
            vec![text("just prose\n#### 3")]
        );
    }

    /// A leading `<<` produces no empty text part before the call — the
    /// divergence from Python's `re.split`.
    #[test]
    fn leading_call_skips_empty_prefix() {
        assert_eq!(
            parts("<<2*3=6>>6 total"),
            vec![python("2*3"), output("6"), text("6 total")]
        );
    }

    /// Through the parquet reader: question → user text, answer → assistant
    /// parts.
    #[test]
    fn loads_conversation_shape() {
        let dir = tempfile::tempdir().unwrap();
        write_gsm8k_shard(
            &dir.path().join("gsm8k-train-00000.parquet"),
            &[("How many?", "It is <<1+2=3>>3.\n#### 3")],
        );

        let convs = load(dir.path(), Split::Train).unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(
            convs[0].messages[0].content,
            Content::Text("How many?".into())
        );
        assert_eq!(
            convs[0].messages[1].content,
            Content::Parts(vec![
                text("It is "),
                python("1+2"),
                output("3"),
                text("3.\n#### 3"),
            ])
        );
    }
}
