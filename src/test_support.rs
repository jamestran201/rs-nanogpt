//! Shared fixtures for unit tests across modules (compiled only under `cfg(test)`).

use std::io::Write;
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::model::{Block, CausalSelfAttention, Gpt, GptConfig, Rope, build_causal_mask};
use crate::tokenizer::BpeTokenizer;

/// Write [`byte_tokenizer`]'s vocabulary to `path`, for tests that hand a
/// *path* to code that opens the file itself (`sft::run` takes `vocab:
/// PathBuf`) — `byte_tokenizer`'s own file is a `NamedTempFile` that drops when
/// it returns.
pub(crate) fn write_byte_vocab(path: &Path) {
    let mut file = std::fs::File::create(path).unwrap();
    for b in 0u32..256 {
        writeln!(file, "{} {}", STANDARD.encode([b as u8]), b).unwrap();
    }
    file.flush().unwrap();
}

/// A byte-level tokenizer: one token per byte value, no merges.
/// `from_file` reads eagerly, so the backing tempfile can be dropped on return.
pub(crate) fn byte_tokenizer() -> BpeTokenizer {
    let vocab_file = tempfile::NamedTempFile::new().unwrap();
    write_byte_vocab(vocab_file.path());
    BpeTokenizer::from_file(vocab_file.path()).unwrap()
}

/// Write a single-column (`text`) parquet shard; `None` entries are null rows.
pub(crate) fn write_shard(path: &Path, texts: Vec<Option<&str>>) {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use parquet::arrow::ArrowWriter;

    let arr: ArrayRef = Arc::new(StringArray::from(texts));
    let batch = RecordBatch::try_from_iter([("text", arr)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Write a parquet file shaped like smoltalk: one `messages`
/// list<struct<content, role>> column, one list entry per conversation.
pub(crate) fn write_smoltalk_shard(path: &Path, rows: &[&[(&str, &str)]]) {
    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, StringBuilder, StructBuilder};
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Fields};
    use parquet::arrow::ArrowWriter;

    let fields = Fields::from(vec![
        Field::new("content", DataType::Utf8, true),
        Field::new("role", DataType::Utf8, true),
    ]);
    let mut lists = ListBuilder::new(StructBuilder::from_fields(fields, 0));
    for conversation in rows {
        let structs = lists.values();
        for (role, content) in *conversation {
            structs
                .field_builder::<StringBuilder>(0)
                .unwrap()
                .append_value(content);
            structs
                .field_builder::<StringBuilder>(1)
                .unwrap()
                .append_value(role);
            structs.append(true);
        }
        lists.append(true);
    }
    let arr: ArrayRef = Arc::new(lists.finish());
    let batch = RecordBatch::try_from_iter([("messages", arr)]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Write a parquet file shaped like mmlu: `question` (string), `choices`
/// (list<string>), `answer` (int64), one row per tuple.
pub(crate) fn write_mmlu_shard(path: &Path, rows: &[(&str, &[&str], i64)]) {
    use std::sync::Arc;

    use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};
    use arrow_array::{ArrayRef, RecordBatch};
    use parquet::arrow::ArrowWriter;

    let mut questions = StringBuilder::new();
    let mut choices = ListBuilder::new(StringBuilder::new());
    let mut answers = Int64Builder::new();
    for (question, row_choices, answer) in rows {
        questions.append_value(question);
        for choice in *row_choices {
            choices.values().append_value(choice);
        }
        choices.append(true);
        answers.append_value(*answer);
    }
    let batch = RecordBatch::try_from_iter([
        ("question", Arc::new(questions.finish()) as ArrayRef),
        ("choices", Arc::new(choices.finish()) as ArrayRef),
        ("answer", Arc::new(answers.finish()) as ArrayRef),
    ])
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// Write a parquet file shaped like gsm8k: `question` and `answer` string
/// columns, one row per tuple.
pub(crate) fn write_gsm8k_shard(path: &Path, rows: &[(&str, &str)]) {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use parquet::arrow::ArrowWriter;

    let questions: Vec<&str> = rows.iter().map(|(q, _)| *q).collect();
    let answers: Vec<&str> = rows.iter().map(|(_, a)| *a).collect();
    let batch = RecordBatch::try_from_iter([
        (
            "question",
            Arc::new(StringArray::from(questions)) as ArrayRef,
        ),
        ("answer", Arc::new(StringArray::from(answers)) as ArrayRef),
    ])
    .unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// A two-shard corpus so a train/val split exists: shard 0 (sorted first) is
/// the Train split's filler, shard 1 the Val split's real text.
pub(crate) fn two_shard_corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        &dir.path().join("0.parquet"),
        vec![Some("filler text here")],
    );
    write_shard(
        &dir.path().join("1.parquet"),
        vec![Some("the quick brown fox jumps over the lazy dog")],
    );
    dir
}

/// A 1-layer, 1-head, 8-wide GPT on CPU for capstone tests.
pub(crate) fn tiny_gpt(vocab_size: usize, sequence_len: usize) -> (VarMap, Gpt) {
    let cfg = GptConfig {
        vocab_size,
        sequence_len,
        n_layer: 1,
        n_head: 1,
        n_embd: 8,
        rope_base: 100_000.0,
        norm_eps: 1e-6,
    };
    let vm = VarMap::new();
    let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
    (vm, model)
}

/// The RoPE tables + causal mask that `Gpt::new` builds once and shares into
/// every block — for tests that construct an attention module or block alone.
fn rope_and_mask(cfg: &GptConfig, dev: &Device) -> Result<(Rope, Tensor)> {
    Ok((
        Rope::from_config(cfg, dev)?,
        build_causal_mask(cfg.sequence_len, dev)?,
    ))
}

pub(crate) fn new_attn(cfg: &GptConfig, vm: &VarMap, dev: &Device) -> Result<CausalSelfAttention> {
    let (rope, mask) = rope_and_mask(cfg, dev)?;
    let vb = VarBuilder::from_varmap(vm, DType::F32, dev);
    CausalSelfAttention::new(cfg, vb, &rope, &mask)
}

pub(crate) fn new_block(cfg: &GptConfig, vm: &VarMap, dev: &Device) -> Result<Block> {
    let (rope, mask) = rope_and_mask(cfg, dev)?;
    let vb = VarBuilder::from_varmap(vm, DType::F32, dev);
    Block::new(cfg, vb, &rope, &mask)
}

/// Break a freshly built [`Gpt`] out of its degenerate init, in place, so that
/// parity tests against it are not vacuous. Two separate traps:
///
/// *Vacuity.* Both residual projections are zero-init, so every block is the
/// identity at init and the logits reduce to `lm_head(rmsnorm(embed(idx)))` —
/// which is *position-independent*. A KV cache that ignored its own contents
/// entirely would pass any parity test on such a model, so both `c_proj`
/// families have to be randomized before the model depends on position at all.
///
/// *Argmax ties.* `lm_head` is tiny-init (`Linear::normal(.., 0.001)`), so a
/// fresh model's logits sit within ~3e-3 across the whole vocab and typical
/// top-2 gaps are O(1e-5) — the same order as the cached-vs-uncached numerical
/// difference, which would make any greedy token-equality test a coin flip.
/// Overwriting `lm_head` at `std = 1.0` makes the gaps O(1). Do not raise that
/// `std`: `assert_close` compares *absolute* differences, so a larger scale
/// eats the tolerance margin silently.
///
/// That same amplification is why the `lm_head` line is needed by the
/// `assert_close` parity tests too, not only the greedy ones: at the stock
/// tiny-init the logits are ~1e-3-scaled, so a real cache bug perturbing the
/// hidden state by O(1e-3) would reach the logits at ~1e-6 and slip under the
/// fixed 1e-5 threshold.
///
/// `VarMap::set_one` writes in place into the `Var` storage the model already
/// holds a handle on, so this works after construction.
pub(crate) fn detie(vm: &mut VarMap, cfg: &GptConfig, dev: &Device) -> Result<()> {
    let c = cfg.n_embd;
    for i in 0..cfg.n_layer {
        let attn = Tensor::randn(0.0f32, 0.5, (c, c), dev)?;
        vm.set_one(format!("blocks.{i}.attn.c_proj.weight"), attn)?;
        let mlp = Tensor::randn(0.0f32, 0.5, (c, 4 * c), dev)?;
        vm.set_one(format!("blocks.{i}.mlp.c_proj.weight"), mlp)?;
    }
    let head = Tensor::randn(0.0f32, 1.0, (cfg.vocab_size, c), dev)?;
    vm.set_one("lm_head.weight", head)
}

/// Elementwise `|got − want| ≤ tol` on the flattened f32 views; shapes must match.
pub(crate) fn assert_close(got: &Tensor, want: &Tensor, tol: f64, what: &str) -> Result<()> {
    assert_eq!(got.dims(), want.dims(), "{what}: shape mismatch");
    let got = got.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let want = want.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() as f64 <= tol,
            "{what}[{i}]: {g} vs {w} (tol {tol})"
        );
    }
    Ok(())
}
