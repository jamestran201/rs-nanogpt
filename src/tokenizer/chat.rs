//! Chat conversation rendering for SFT: a [`Conversation`] becomes token ids
//! plus a loss mask. Port of nanochat's `render_conversation`
//! (`nanochat/tokenizer.py:266`).
//!
//! The mask is true exactly on the tokens the model must produce at inference
//! time: assistant text, `<|python_*|>` tool-call spans, and
//! `<|assistant_end|>` (stop-token supervision). Everything the user or
//! harness supplies — BOS, user turns, `<|assistant_start|>`, tool outputs —
//! stays unscored context, still fed through the forward pass as input. The
//! SFT packer later turns unscored positions into `-1` (ignore_index) targets.

use std::io;

use super::bpe::BpeTokenizer;
use super::shared::TokenId;

/// Who authored a message. `System` is legal only as the first message: it has
/// no delimiter tokens and merges into the following user turn at render time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// One piece of a structured assistant message. The shape comes from GSM8K,
/// whose `<<expr=result>>` calculator annotations split into a `Python` call
/// and its `PythonOutput` (`tasks/gsm8k.py:60-76`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Part {
    /// Assistant prose (scored).
    Text(String),
    /// A tool call the model emits, wrapped in `<|python_start|>` /
    /// `<|python_end|>` (scored — the model decides to call the tool).
    Python(String),
    /// The interpreter's reply, wrapped in `<|output_start|>` /
    /// `<|output_end|>` (unscored — the harness injects it at inference).
    PythonOutput(String),
}

/// Message payload: plain text, or a sequence of parts (assistant only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

/// A chat conversation: an optional leading system message, then strictly
/// alternating user/assistant messages starting with user. May end on either
/// role (ending on user is the inference prefill shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conversation {
    pub messages: Vec<Message>,
}

/// A rendered conversation. `ids` and `mask` have equal length; `mask[i]` is
/// true when the loss scores `ids[i]` as a prediction target.
#[derive(Debug, PartialEq, Eq)]
pub struct RenderedConversation {
    pub ids: Vec<TokenId>,
    pub mask: Vec<bool>,
}

/// The chat special ids, resolved once per render. Infallible: every
/// `BpeTokenizer` reserves the whole `shared::SPECIAL_TOKENS` block above its
/// learned merges.
struct ChatSpecials {
    user_start: TokenId,
    user_end: TokenId,
    assistant_start: TokenId,
    assistant_end: TokenId,
    python_start: TokenId,
    python_end: TokenId,
    output_start: TokenId,
    output_end: TokenId,
}

impl ChatSpecials {
    fn resolve(tok: &BpeTokenizer) -> Self {
        let id = |name: &str| {
            tok.special_id(name)
                .expect("chat special tokens are reserved in every vocabulary")
        };
        Self {
            user_start: id("<|user_start|>"),
            user_end: id("<|user_end|>"),
            assistant_start: id("<|assistant_start|>"),
            assistant_end: id("<|assistant_end|>"),
            python_start: id("<|python_start|>"),
            python_end: id("<|python_end|>"),
            output_start: id("<|output_start|>"),
            output_end: id("<|output_end|>"),
        }
    }
}

/// Accumulates the parallel `ids`/`mask` streams during a render.
struct Renderer<'a> {
    tok: &'a BpeTokenizer,
    sp: ChatSpecials,
    ids: Vec<TokenId>,
    mask: Vec<bool>,
}

impl Renderer<'_> {
    fn push(&mut self, id: TokenId, scored: bool) {
        self.ids.push(id);
        self.mask.push(scored);
    }

    fn text(&mut self, text: &str, scored: bool) {
        let ids = self.tok.encode(text);
        self.mask.extend(std::iter::repeat_n(scored, ids.len()));
        self.ids.extend(ids);
    }

    fn user_turn(&mut self, text: &str) {
        self.push(self.sp.user_start, false);
        self.text(text, false);
        self.push(self.sp.user_end, false);
    }

    fn assistant_turn(&mut self, content: &Content) {
        // The start token primes the reply at inference, so it is never a
        // target; the end token is how the model yields the turn, so it
        // always is.
        self.push(self.sp.assistant_start, false);
        match content {
            Content::Text(t) => self.text(t, true),
            Content::Parts(parts) => {
                for part in parts {
                    match part {
                        Part::Text(t) => self.text(t, true),
                        Part::Python(t) => {
                            self.push(self.sp.python_start, true);
                            self.text(t, true);
                            self.push(self.sp.python_end, true);
                        }
                        Part::PythonOutput(t) => {
                            self.push(self.sp.output_start, false);
                            self.text(t, false);
                            self.push(self.sp.output_end, false);
                        }
                    }
                }
            }
        }
        self.push(self.sp.assistant_end, true);
    }

    fn finish(mut self, max_tokens: usize) -> RenderedConversation {
        self.ids.truncate(max_tokens);
        self.mask.truncate(max_tokens);
        RenderedConversation {
            ids: self.ids,
            mask: self.mask,
        }
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

impl BpeTokenizer {
    /// Render a conversation into token ids plus a loss mask, both truncated
    /// to `max_tokens`. Truncation may cut mid-assistant-span, leaving no
    /// `<|assistant_end|>` — accepted, matching nanochat (`tokenizer.py:347`).
    ///
    /// `InvalidData` errors are data-corruption guards ported from nanochat's
    /// asserts: empty conversation, a leading system message not followed by a
    /// user message, broken user/assistant alternation, non-text user/system
    /// content. Indices they report are indices into `conv.messages`.
    pub fn render_conversation(
        &self,
        conv: &Conversation,
        max_tokens: usize,
    ) -> io::Result<RenderedConversation> {
        let messages = &conv.messages;
        if messages.is_empty() {
            return Err(invalid("conversation has no messages"));
        }

        // A leading system message renders as "{system}\n\n{user}" inside the
        // first user turn (tokenizer.py:283).
        let (merged_first_user, rest, index_offset) = if messages[0].role == Role::System {
            let Content::Text(sys) = &messages[0].content else {
                return Err(invalid("system message content must be text"));
            };
            match messages.get(1) {
                None => {
                    return Err(invalid(
                        "system message must be followed by a user message, \
                         but the conversation ends after it",
                    ));
                }
                Some(m) if m.role != Role::User => {
                    return Err(invalid(format!(
                        "system message must be followed by a user message, got {}",
                        role_name(m.role)
                    )));
                }
                Some(m) => {
                    let Content::Text(user) = &m.content else {
                        return Err(invalid("message 1: user message content must be text"));
                    };
                    (Some(format!("{sys}\n\n{user}")), &messages[2..], 1usize)
                }
            }
        } else {
            (None, &messages[..], 0)
        };

        let mut r = Renderer {
            tok: self,
            sp: ChatSpecials::resolve(self),
            ids: Vec::new(),
            mask: Vec::new(),
        };
        r.push(self.bos_id(), false);

        // `logical` walks the post-merge alternation, so errors add
        // `index_offset` to recover the caller's message index.
        let mut logical = 0usize;
        if let Some(text) = &merged_first_user {
            r.user_turn(text);
            logical = 1;
        }
        for msg in rest {
            let expect_user = logical.is_multiple_of(2);
            let expected = if expect_user {
                Role::User
            } else {
                Role::Assistant
            };
            if msg.role != expected {
                return Err(invalid(format!(
                    "message {}: expected {}, got {}",
                    logical + index_offset,
                    role_name(expected),
                    role_name(msg.role)
                )));
            }
            if expect_user {
                let Content::Text(text) = &msg.content else {
                    return Err(invalid(format!(
                        "message {}: user message content must be text",
                        logical + index_offset
                    )));
                };
                r.user_turn(text);
            } else {
                r.assistant_turn(&msg.content);
            }
            logical += 1;
        }

        Ok(r.finish(max_tokens))
    }

    /// Append one user turn plus the `<|assistant_start|>` that primes the
    /// reply: `<|user_start|>{text}<|user_end|><|assistant_start|>`. The
    /// incremental, inference-time counterpart of
    /// [`render_conversation`](Self::render_conversation), for callers that
    /// build a conversation turn by turn (`chat_cli.py:72-77`) rather than
    /// re-rendering a message list.
    ///
    /// The wire format has one owner: this must agree with
    /// `render_conversation` on the prefill shape, which
    /// `push_user_turn_matches_render_conversation` pins.
    pub fn push_user_turn(&self, ids: &mut Vec<TokenId>, text: &str) {
        let sp = ChatSpecials::resolve(self);
        ids.push(sp.user_start);
        ids.extend(self.encode(text));
        ids.push(sp.user_end);
        ids.push(sp.assistant_start);
    }

    /// The ids that end an assistant turn at inference: `<|assistant_end|>`
    /// (the trained stop token) and `<|bos|>` (a document boundary the model
    /// should never emit mid-reply; nanochat treats it as terminal too,
    /// `engine.py:254`).
    pub fn assistant_stop_ids(&self) -> [TokenId; 2] {
        [ChatSpecials::resolve(self).assistant_end, self.bos_id()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::byte_tokenizer;

    // byte_tokenizer(): byte value b = token id b, specials at 256-264 in
    // SPECIAL_TOKENS order — bos=256, user_start=257, user_end=258,
    // assistant_start=259, assistant_end=260, python_start=261,
    // python_end=262, output_start=263, output_end=264.

    fn user(text: &str) -> Message {
        Message {
            role: Role::User,
            content: Content::Text(text.into()),
        }
    }

    fn assistant(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: Content::Text(text.into()),
        }
    }

    fn system(text: &str) -> Message {
        Message {
            role: Role::System,
            content: Content::Text(text.into()),
        }
    }

    fn conv(messages: Vec<Message>) -> Conversation {
        Conversation { messages }
    }

    fn render(c: &Conversation) -> RenderedConversation {
        byte_tokenizer().render_conversation(c, 10_000).unwrap()
    }

    fn assert_render_err(c: &Conversation, needle: &str) {
        let err = byte_tokenizer().render_conversation(c, 10_000).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains(needle),
            "error {msg:?} should contain {needle:?}"
        );
    }

    #[test]
    fn two_message_layout_exact() {
        let r = render(&conv(vec![user("Hi"), assistant("Yo")]));
        //                bos  us    H   i   ue   as    Y   o   ae
        assert_eq!(r.ids, [256, 257, 72, 105, 258, 259, 89, 111, 260]);
        assert_eq!(
            r.mask,
            [false, false, false, false, false, false, true, true, true]
        );
    }

    /// The call span is scored; the output span is not, since it comes from
    /// the interpreter at inference time.
    #[test]
    fn parts_render_and_mask() {
        let c = conv(vec![
            user("q"),
            Message {
                role: Role::Assistant,
                content: Content::Parts(vec![
                    Part::Text("a".into()),
                    Part::Python("b".into()),
                    Part::PythonOutput("c".into()),
                ]),
            },
        ]);
        let r = render(&c);
        //                bos  us    q   ue   as   a   ps   b   pe   os   c   oe   ae
        assert_eq!(
            r.ids,
            [256, 257, 113, 258, 259, 97, 261, 98, 262, 263, 99, 264, 260]
        );
        assert_eq!(
            r.mask,
            [
                false, false, false, false, false, true, true, true, true, false, false, false,
                true
            ]
        );
    }

    /// User turns stay unscored after the first exchange too.
    #[test]
    fn multi_turn_mask_pattern() {
        let c = conv(vec![user("a"), assistant("b"), user("c"), assistant("d")]);
        let r = render(&c);
        //                bos  us   a   ue   as   b   ae   us   c   ue   as   d    ae
        assert_eq!(
            r.ids,
            [256, 257, 97, 258, 259, 98, 260, 257, 99, 258, 259, 100, 260]
        );
        assert_eq!(
            r.mask,
            [
                false, false, false, false, false, true, true, false, false, false, false, true,
                true
            ]
        );
    }

    /// A leading system message renders exactly as if its text were already
    /// joined to the first user message with "\n\n" (tokenizer.py:283-291).
    #[test]
    fn system_merge_matches_premerged() {
        let merged = render(&conv(vec![system("S"), user("U"), assistant("A")]));
        let premerged = render(&conv(vec![user("S\n\nU"), assistant("A")]));
        assert_eq!(merged.ids, premerged.ids);
        assert_eq!(merged.mask, premerged.mask);
    }

    #[test]
    fn truncation_caps_both_vectors() {
        let c = conv(vec![user("Hi"), assistant("Yo")]);
        let full = render(&c);
        let cut = byte_tokenizer().render_conversation(&c, 4).unwrap();
        assert_eq!(cut.ids.len(), 4);
        assert_eq!(cut.mask.len(), 4);
        assert_eq!(cut.ids[..], full.ids[..4]);
        assert_eq!(cut.mask[..], full.mask[..4]);
    }

    /// The prefill shape the chat CLI will use: nothing to score, because it
    /// appends `<|assistant_start|>` and lets the model continue.
    #[test]
    fn ends_on_user_renders_all_unmasked() {
        let r = render(&conv(vec![user("Hi")]));
        assert_eq!(r.ids, [256, 257, 72, 105, 258]);
        assert!(r.mask.iter().all(|&m| !m));
    }

    /// The incremental builder must agree with the whole-conversation
    /// renderer: `[bos] ++ push_user_turn(t)` is the render of a
    /// conversation ending on that user turn, plus the `<|assistant_start|>`
    /// that primes the reply. Checked after an assistant turn too, since a
    /// REPL keeps appending to the same id stream.
    #[test]
    fn push_user_turn_matches_render_conversation() {
        let tok = byte_tokenizer();
        let assistant_start = tok.special_id("<|assistant_start|>").unwrap();

        let mut ids = vec![tok.bos_id()];
        tok.push_user_turn(&mut ids, "Hi");
        let mut want = render(&conv(vec![user("Hi")])).ids;
        want.push(assistant_start);
        assert_eq!(ids, want);

        // The model's reply enters the stream as ids; the next push must then
        // match a full re-render of all three messages.
        ids.extend(tok.encode("Yo"));
        ids.push(tok.special_id("<|assistant_end|>").unwrap());
        tok.push_user_turn(&mut ids, "Bye");
        let mut want = render(&conv(vec![user("Hi"), assistant("Yo"), user("Bye")])).ids;
        want.push(assistant_start);
        assert_eq!(ids, want);
    }

    #[test]
    fn assistant_stop_ids_are_assistant_end_and_bos() {
        assert_eq!(byte_tokenizer().assistant_stop_ids(), [260, 256]);
    }

    /// `decode` renders specials as literal strings, so the round trip shows
    /// the full wire format.
    #[test]
    fn decode_round_trip() {
        let tok = byte_tokenizer();
        let r = tok
            .render_conversation(&conv(vec![user("Hi"), assistant("Yo")]), 10_000)
            .unwrap();
        assert_eq!(
            tok.decode(&r.ids),
            "<|bos|><|user_start|>Hi<|user_end|><|assistant_start|>Yo<|assistant_end|>"
        );
    }

    #[test]
    fn empty_conversation_errors() {
        assert_render_err(&conv(vec![]), "no messages");
    }

    #[test]
    fn assistant_first_errors() {
        assert_render_err(
            &conv(vec![assistant("A")]),
            "message 0: expected user, got assistant",
        );
    }

    #[test]
    fn user_after_user_errors() {
        assert_render_err(
            &conv(vec![user("a"), user("b")]),
            "message 1: expected assistant, got user",
        );
    }

    /// Anywhere but first, a system message breaks alternation and is
    /// reported at the caller's index.
    #[test]
    fn mid_conversation_system_errors() {
        assert_render_err(
            &conv(vec![user("a"), system("S")]),
            "message 1: expected assistant, got system",
        );
    }

    #[test]
    fn system_not_followed_by_user_errors() {
        let needle = "system message must be followed by a user message";
        assert_render_err(&conv(vec![system("S")]), needle);
        assert_render_err(&conv(vec![system("S"), assistant("A")]), needle);
    }

    #[test]
    fn user_with_parts_errors() {
        let c = conv(vec![Message {
            role: Role::User,
            content: Content::Parts(vec![Part::Text("x".into())]),
        }]);
        assert_render_err(&c, "message 0: user message content must be text");
    }

    /// The merge path validates content too: both halves must be plain text.
    #[test]
    fn system_or_merged_user_with_parts_errors() {
        let parts = Content::Parts(vec![Part::Text("x".into())]);
        let c = conv(vec![
            Message {
                role: Role::System,
                content: parts.clone(),
            },
            user("u"),
        ]);
        assert_render_err(&c, "system message content must be text");

        let c = conv(vec![
            system("S"),
            Message {
                role: Role::User,
                content: parts,
            },
        ]);
        assert_render_err(&c, "message 1: user message content must be text");
    }
}
