//! The `chat` subcommand: talk to a finetuned checkpoint on stdin/stdout.
//! Port of nanochat's `scripts/chat_cli.py` plus the sampling half of
//! `nanochat/engine.py`. Spec: `writeups/sft-chat-plan.md`.
//!
//! **History is a token stream, not a message list.** `ids` starts as
//! `[<|bos|>]` and only ever grows: each turn appends the rendered user span
//! (via [`BpeTokenizer::push_user_turn`]) and then the model's own generated
//! ids, verbatim. Re-rendering a `Vec<Message>` each turn would have to
//! `decode` the reply and `encode` it back, and neither direction is
//! information-preserving — `decode` writes a special id as its literal text
//! and `encode` never produces special ids, so one emitted `<|python_start|>`
//! would come back as ~7 ordinary tokens and the model would see a history it
//! could not have produced.
//!
//! Single process, single device: inference needs no `DistCtx`, so there is no
//! `--gpus` here.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use candle_core::{Device, Result, bail};

use crate::checkpoint;
use crate::eval::sample::{SampleOptions, generate_ids};
use crate::model::Gpt;
use crate::tokenizer::{BpeTokenizer, TokenId};

/// Resolved `chat` inputs — the CLI's flags after mapping.
#[derive(Debug, Clone)]
pub struct ChatConfig {
    /// The finetuned checkpoint to talk to, e.g. `out-sft/<run>/best`.
    pub checkpoint: PathBuf,
    /// Tiktoken-format vocabulary — must be the one the checkpoint was
    /// trained with.
    pub vocab: PathBuf,
    /// Single-shot mode: answer this once and exit (`chat_cli.py:99-100`).
    pub prompt: Option<String>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub seed: u64,
}

impl ChatConfig {
    pub fn validate(&self) -> std::result::Result<(), String> {
        // At 0 the REPL looks like a model that has learned to answer every
        // question with nothing: `generate_ids` returns empty and the
        // stop-token fixup still closes the turn.
        if self.max_tokens == 0 {
            return Err("--max-tokens must be >= 1".into());
        }
        // `SampleOptions` treats `<= 0` as greedy, which is the cheap library
        // contract; without this rule `--temperature -1` would be silently
        // greedy here while `sft --sample-temperature -1` errors.
        // `is_finite` first, as `--init-lr-frac` does: `x < 0.0` alone is false
        // for NaN, and `f64: FromStr` yields `inf` on overflow rather than an
        // error. Both slip past the greedy branch into `sample`, where NaN
        // weights make `draw` fall through to the last vocab id for *every*
        // token and `inf` flattens the softmax to uniform — silently, either way.
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(format!(
                "--temperature must be a finite value >= 0, got {}",
                self.temperature
            ));
        }
        Ok(())
    }
}

/// Load the checkpoint and talk to it on stdin/stdout.
pub fn run(cfg: &ChatConfig, device: &Device) -> Result<()> {
    // Defend against our own callers, as `sft::run` does: this is a library
    // entry point, and `main`'s check only covers the CLI path.
    cfg.validate().map_err(candle_core::Error::msg)?;

    // Cheap checks before the safetensors read, as `sft::run` does.
    let meta = checkpoint::load_meta(&cfg.checkpoint)?;
    let tok = BpeTokenizer::from_file(&cfg.vocab)?;
    if tok.vocab_size() != meta.config.vocab_size {
        bail!(
            "vocab {} has {} tokens but the checkpoint was trained with {}; a mismatched \
             vocabulary loads fine and scores garbage",
            cfg.vocab.display(),
            tok.vocab_size(),
            meta.config.vocab_size
        );
    }
    let (model, _varmap, _) = checkpoint::load(&cfg.checkpoint, device)?;

    println!(
        "chat: {} | step {} | val_bpb {:.4} | context {} tokens",
        cfg.checkpoint.display(),
        meta.step,
        meta.val_bpb,
        meta.config.sequence_len
    );
    println!(
        "sampling: temperature {} | top_k {} | max_tokens {} | seed {}",
        cfg.temperature, cfg.top_k, cfg.max_tokens, cfg.seed
    );
    // The command list, as `chat_cli.py:37-41` prints it: the context guard's
    // advice below refers to `clear`, so the user has to have been told it.
    println!("commands: 'clear' starts a new conversation, 'quit' or Ctrl-D exits");

    repl(
        &model,
        &tok,
        cfg,
        device,
        &mut std::io::stdin().lock(),
        &mut std::io::stdout(),
    )
}

/// The turn loop, over any reader/writer so tests can drive it.
pub fn repl(
    model: &Gpt,
    tok: &BpeTokenizer,
    cfg: &ChatConfig,
    device: &Device,
    input: &mut impl BufRead,
    out: &mut impl Write,
) -> Result<()> {
    let mut ids = vec![tok.bos_id()];
    let mut line = String::new();
    loop {
        let text = match &cfg.prompt {
            Some(p) => p.trim(),
            None => {
                write!(out, "\nUser: ")?;
                out.flush()?;
                line.clear();
                if input.read_line(&mut line)? == 0 {
                    writeln!(out, "\nGoodbye!")?;
                    return Ok(());
                }
                line.trim()
            }
        };

        if text.eq_ignore_ascii_case("quit") || text.eq_ignore_ascii_case("exit") {
            writeln!(out, "Goodbye!")?;
            return Ok(());
        } else if text.eq_ignore_ascii_case("clear") {
            ids = vec![tok.bos_id()];
            writeln!(out, "conversation cleared")?;
        } else if !text.is_empty() {
            chat_turn(model, tok, cfg, device, &mut ids, text, out)?;
        }

        // Single-shot mode answers once and exits — including when the prompt
        // was empty or a command, which is also what stops this looping.
        if cfg.prompt.is_some() {
            return Ok(());
        }
    }
}

/// One turn: append the user's text, generate, stream, append the reply.
/// `ids` is left untouched if the turn cannot fit the context.
fn chat_turn(
    model: &Gpt,
    tok: &BpeTokenizer,
    cfg: &ChatConfig,
    device: &Device,
    ids: &mut Vec<TokenId>,
    text: &str,
    out: &mut impl Write,
) -> Result<()> {
    let seq_len = model.config().sequence_len;
    let base = ids.len();
    tok.push_user_turn(ids, text);

    // Without a KV cache and with `sequence_len` fixed by RoPE, a long chat
    // eventually outgrows the context. `generate_ids` would crop to the last
    // `seq_len` ids — dropping BOS and starting mid-turn, which reads as
    // confident nonsense — so the budget is checked here instead, and the crop
    // path is closed rather than merely avoided: with `room >= 1`, the longest
    // context any forward sees is `seq_len - 1`.
    let room = seq_len.saturating_sub(ids.len());
    if room == 0 {
        ids.truncate(base);
        writeln!(
            out,
            "\n[this turn does not fit the model's {seq_len}-token context; type 'clear' to start over]"
        )?;
        return Ok(());
    }
    let max_tokens = cfg.max_tokens.min(room);
    if max_tokens < cfg.max_tokens {
        writeln!(
            out,
            "\n[only {max_tokens} of the {seq_len}-token context are left; the reply will be cut short]"
        )?;
    }

    write!(out, "\nAssistant: ")?;
    out.flush()?;
    let opts = SampleOptions {
        max_tokens,
        temperature: cfg.temperature,
        top_k: cfg.top_k,
        seed: cfg.seed,
    };
    // The decoder holds back a partial UTF-8 sequence rather than printing
    // U+FFFD in the middle of every multi-byte character.
    let mut dec = tok.decode_stream();
    let stop = tok.assistant_stop_ids();
    let mut generated = generate_ids(model, ids, opts, &stop, device, |id| {
        out.write_all(dec.push(id).as_bytes())?;
        out.flush()?;
        Ok(())
    })?;
    out.write_all(dec.finish().as_bytes())?;
    writeln!(out)?;

    // A reply may terminate on either stop id. `<|bos|>` is the pretraining
    // *document* delimiter — `render_conversation` only ever emits it at
    // position 0 — so leaving it in the history would tell the model a new
    // document started, off-distribution conditioning for every later turn.
    // Drop it. (`chat_cli.py:92-95` keeps it, but `engine.py:286` states the
    // intended rule: "Terminal tokens (assistant_end, bos) are not included".)
    let [assistant_end, bos] = stop;
    if generated.last() == Some(&bos) {
        generated.pop();
    }
    // Budget exhaustion leaves the turn unterminated. Appending the stop token
    // keeps the next turn's history well-formed (`chat_cli.py:92-95`).
    if generated.last() != Some(&assistant_end) {
        generated.push(assistant_end);
    }
    ids.extend(generated);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{byte_tokenizer, tiny_gpt};

    // byte_tokenizer(): byte value b = token id b, bos=256, user_start=257,
    // user_end=258, assistant_start=259, assistant_end=260.

    fn chat_cfg(max_tokens: usize) -> ChatConfig {
        ChatConfig {
            checkpoint: PathBuf::from("unused"),
            vocab: PathBuf::from("unused"),
            prompt: None,
            max_tokens,
            temperature: 0.0,
            top_k: 0,
            seed: 42,
        }
    }

    fn text_of(out: &[u8]) -> String {
        String::from_utf8_lossy(out).into_owned()
    }

    /// The part of a turn the caller controls: the rendered user span plus the
    /// `<|assistant_start|>` that primes the reply. What the model generates
    /// after it is not assertable — `tiny_gpt` has an unseeded random init, so
    /// a reply may contain any id at all, specials included.
    fn assert_user_span(tok: &BpeTokenizer, ids: &[TokenId], base: usize, text: &str) {
        let mut want = vec![tok.special_id("<|user_start|>").unwrap()];
        want.extend(tok.encode(text));
        want.push(tok.special_id("<|user_end|>").unwrap());
        want.push(tok.special_id("<|assistant_start|>").unwrap());
        assert_eq!(&ids[base..base + want.len()], &want[..]);
    }

    #[test]
    fn chat_turn_appends_a_well_formed_turn() -> Result<()> {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 128);
        let cfg = chat_cfg(4);
        let assistant_end = tok.special_id("<|assistant_end|>").unwrap();

        let mut ids = vec![tok.bos_id()];
        let mut out: Vec<u8> = Vec::new();

        chat_turn(&model, &tok, &cfg, &dev, &mut ids, "hi", &mut out)?;
        assert_user_span(&tok, &ids, 1, "hi");
        // Whether the model stopped on its own or the fixup supplied it, the
        // turn closes on the stop token — that invariant is what keeps the
        // history well-formed, not any particular reply length.
        assert_eq!(ids.last(), Some(&assistant_end));

        let after_first = ids.clone();
        let base = ids.len();
        chat_turn(&model, &tok, &cfg, &dev, &mut ids, "again", &mut out)?;
        assert!(
            ids.starts_with(&after_first),
            "a turn only ever appends to the history"
        );
        assert_user_span(&tok, &ids, base, "again");
        assert_eq!(ids.last(), Some(&assistant_end));

        // The stop token lives in the history, never in the transcript.
        assert!(!text_of(&out).contains("<|assistant_end|>"));
        Ok(())
    }

    #[test]
    fn chat_turn_refuses_a_turn_that_cannot_fit() -> Result<()> {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 16);
        let cfg = chat_cfg(2);

        let mut ids = vec![tok.bos_id()];
        let mut out: Vec<u8> = Vec::new();
        // 1 (bos) + 1 + 20 (bytes) + 1 + 1 is well past the 16-token context.
        chat_turn(
            &model,
            &tok,
            &cfg,
            &dev,
            &mut ids,
            "a question far too long",
            &mut out,
        )?;
        assert_eq!(ids, [tok.bos_id()], "a refused turn leaves ids untouched");
        let printed = text_of(&out);
        assert!(printed.contains("16-token context"), "got {printed:?}");
        assert!(!printed.contains("Assistant:"));

        // The session survives it: a short turn still works afterwards.
        chat_turn(&model, &tok, &cfg, &dev, &mut ids, "hi", &mut out)?;
        assert_user_span(&tok, &ids, 1, "hi");
        Ok(())
    }

    /// `sequence_len: 12` is chosen so both outcomes are forced arithmetically
    /// rather than likely: `push_user_turn("hi")` appends 5 ids
    /// (`us h i ue as`) to the `[bos]` already there, so turn 1 always has
    /// `room == 6`; the shortest possible post-turn history is 7 (`max_tokens
    /// >= 1` and `room >= 1` force at least one generated token, and the
    /// cheapest is that token being `<|assistant_end|>` itself, or a `<|bos|>`
    /// the fixup pops and replaces with one), so a second
    /// push of 5 always leaves `room == 0`. The refusal case is the point —
    /// without it, "post-`clear` succeeds" proves nothing about `clear`.
    #[test]
    fn repl_handles_clear_quit_and_eof() -> Result<()> {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 12);
        let cfg = chat_cfg(4);

        let drive = |input: &str| -> Result<String> {
            let mut out: Vec<u8> = Vec::new();
            repl(
                &model,
                &tok,
                &cfg,
                &dev,
                &mut std::io::Cursor::new(input.as_bytes().to_vec()),
                &mut out,
            )?;
            Ok(text_of(&out))
        };

        let uncleared = drive("hi\nhi\nquit\n")?;
        assert_eq!(uncleared.matches("Assistant:").count(), 1);
        assert_eq!(uncleared.matches("type 'clear'").count(), 1);

        let cleared = drive("hi\nclear\nhi\nquit\n")?;
        assert_eq!(
            cleared.matches("Assistant:").count(),
            2,
            "clear must reset the history, so the second turn fits again"
        );
        assert!(!cleared.contains("type 'clear'"));
        assert!(cleared.contains("conversation cleared"));

        let eof = drive("hi\n")?;
        assert!(eof.contains("Goodbye!"), "got {eof:?}");
        Ok(())
    }

    /// The middle of the three budget outcomes — the one where the `min` could
    /// be inverted or the count misreported with nothing else noticing.
    /// `sequence_len: 12` forces it: `[bos]` plus the 5 ids `push_user_turn`
    /// appends leaves `room == 6`, below `max_tokens == 8`.
    #[test]
    fn chat_turn_warns_when_the_budget_truncates_the_reply() -> Result<()> {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 12);
        let cfg = chat_cfg(8);

        let mut ids = vec![tok.bos_id()];
        let mut out: Vec<u8> = Vec::new();
        chat_turn(&model, &tok, &cfg, &dev, &mut ids, "hi", &mut out)?;

        let printed = text_of(&out);
        assert!(
            printed.contains("only 6 of the 12-token context"),
            "got {printed:?}"
        );
        // Warned, not refused: the turn still runs.
        assert!(printed.contains("Assistant:"));
        assert_user_span(&tok, &ids, 1, "hi");
        Ok(())
    }

    #[test]
    fn repl_prompt_mode_runs_exactly_one_turn() -> Result<()> {
        let dev = Device::Cpu;
        let tok = byte_tokenizer();
        let (_vm, model) = tiny_gpt(tok.vocab_size(), 64);
        let mut cfg = chat_cfg(4);
        cfg.prompt = Some("hi".into());

        let mut out: Vec<u8> = Vec::new();
        repl(
            &model,
            &tok,
            &cfg,
            &dev,
            &mut std::io::Cursor::new(Vec::new()),
            &mut out,
        )?;
        let printed = text_of(&out);
        assert_eq!(printed.matches("Assistant:").count(), 1);
        assert!(!printed.contains("User:"), "no prompt echo in prompt mode");
        Ok(())
    }

    #[test]
    fn chat_config_validate_rejects_one_rule_at_a_time() {
        let rejects = |mutate: fn(&mut ChatConfig)| {
            let mut c = chat_cfg(256);
            mutate(&mut c);
            assert!(c.validate().is_err());
        };
        assert_eq!(chat_cfg(256).validate(), Ok(()));
        rejects(|c| c.max_tokens = 0);
        rejects(|c| c.temperature = -0.1);
        // Both slip past `< 0.0` and degenerate sampling silently: NaN pins
        // every draw to the last vocab id, inf flattens it to uniform.
        rejects(|c| c.temperature = f64::NAN);
        rejects(|c| c.temperature = f64::INFINITY);
    }
}
