//! Conversations from a JSONL file, one JSON array of
//! `{"role": …, "content": …}` objects per line. Port of nanochat's
//! `tasks/customjson.py`. The only production input is the 1000 identity
//! conversations (`identity_conversations.jsonl`) that give the model its
//! persona.

use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use serde::Deserialize;

use super::{invalid, role_from_str};
use crate::tokenizer::{Content, Conversation, Message};

/// The on-disk message shape. Private so the public conversation types stay
/// serde-free; extra JSON fields are ignored, missing ones are errors.
#[derive(Deserialize)]
struct RawMessage {
    role: String,
    content: String,
}

/// Load every conversation in the JSONL file at `path`. Blank lines are
/// skipped; errors name the 1-based line number.
pub fn load(path: &Path) -> io::Result<Vec<Conversation>> {
    let file = fs::File::open(path)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let context = |what: String| format!("{}: line {}: {what}", path.display(), idx + 1);
        let line_err = |what: String| invalid(context(what));
        // A read error here is usually invalid UTF-8; keep its kind but name
        // the line, like every other error in this loop.
        let line = line.map_err(|e| io::Error::new(e.kind(), context(e.to_string())))?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: Vec<RawMessage> =
            serde_json::from_str(&line).map_err(|e| line_err(format!("invalid JSON: {e}")))?;
        let messages = raw
            .into_iter()
            .map(|m| {
                let role = role_from_str(&m.role)
                    .ok_or_else(|| line_err(format!("unknown role {:?}", m.role)))?;
                Ok(Message {
                    role,
                    content: Content::Text(m.content),
                })
            })
            .collect::<io::Result<Vec<Message>>>()?;
        out.push(Conversation { messages });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::tokenizer::Role;

    fn jsonl(contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn loads_two_conversations() {
        let f = jsonl(concat!(
            r#"[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]"#,
            "\n",
            r#"[{"role":"user","content":"Who are you?"},{"role":"assistant","content":"nanochat"}]"#,
            "\n",
        ));
        let convs = load(f.path()).unwrap();
        assert_eq!(convs.len(), 2);
        assert_eq!(convs[0].messages.len(), 2);
        assert_eq!(convs[0].messages[0].role, Role::User);
        assert_eq!(convs[0].messages[1].content, Content::Text("Hello".into()));
        assert_eq!(
            convs[1].messages[1].content,
            Content::Text("nanochat".into())
        );
    }

    #[test]
    fn malformed_line_errors_with_line_number() {
        let f = jsonl(concat!(
            r#"[{"role":"user","content":"ok"}]"#,
            "\n",
            "not json\n",
        ));
        let err = load(f.path()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("line 2") && msg.contains("invalid JSON"),
            "unexpected error: {msg}"
        );
    }

    /// Blank lines are skipped but still count toward the reported line
    /// number.
    #[test]
    fn blank_lines_skipped() {
        let f = jsonl(concat!(
            "\n",
            r#"[{"role":"user","content":"a"}]"#,
            "\n",
            "   \n",
            r#"[{"role":"user","content":"b"}]"#,
            "\n",
        ));
        assert_eq!(load(f.path()).unwrap().len(), 2);

        let f = jsonl(concat!("\n", "   \n", "not json\n"));
        let msg = load(f.path()).unwrap_err().to_string();
        assert!(msg.contains("line 3"), "unexpected error: {msg}");
    }

    #[test]
    fn unknown_role_errors_with_line_number() {
        let f = jsonl(concat!(r#"[{"role":"tool","content":"x"}]"#, "\n"));
        let err = load(f.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("line 1") && msg.contains("\"tool\""),
            "unexpected error: {msg}"
        );
    }
}
