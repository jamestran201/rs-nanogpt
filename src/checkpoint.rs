//! Model checkpointing: persist a trained `Gpt` and reload it for inference (resume-training is out of scope).

use std::fs;
use std::path::Path;

use candle_core::{DType, Device, Error, Result, bail};
use candle_nn::{VarBuilder, VarMap};

use crate::model::{Gpt, GptConfig};

const MODEL_FILE: &str = "model.safetensors";
const META_FILE: &str = "meta.txt";

#[derive(Debug, Clone)]
pub struct CheckpointMeta {
    pub config: GptConfig,
    pub step: usize,
    pub val_bpb: f64,
}

pub fn save(dir: &Path, varmap: &VarMap, meta: &CheckpointMeta) -> Result<()> {
    fs::create_dir_all(dir)?;
    varmap.save(dir.join(MODEL_FILE))?;
    fs::write(dir.join(META_FILE), write_meta(meta))?;
    Ok(())
}

/// Load a model from a checkpoint.
///
/// Tokenizer-free by design: the caller is responsible for cross-checking
/// `meta.config.vocab_size` against its own tokenizer (a real footgun — a
/// checkpoint built with a different vocab would load but score garbage).
pub fn load(dir: &Path, device: &Device) -> Result<(Gpt, VarMap, CheckpointMeta)> {
    let contents = fs::read_to_string(dir.join(META_FILE))?;
    let meta = parse_meta(&contents)?;
    meta.config.validate()?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = Gpt::new(meta.config, vb)?;
    varmap.load(dir.join(MODEL_FILE))?;

    Ok((model, varmap, meta))
}

fn write_meta(meta: &CheckpointMeta) -> String {
    let c = &meta.config;
    format!(
        "vocab_size {}\n\
         sequence_len {}\n\
         n_layer {}\n\
         n_head {}\n\
         n_embd {}\n\
         rope_base {}\n\
         norm_eps {}\n\
         step {}\n\
         val_bpb {}\n",
        c.vocab_size,
        c.sequence_len,
        c.n_layer,
        c.n_head,
        c.n_embd,
        c.rope_base,
        c.norm_eps,
        meta.step,
        meta.val_bpb,
    )
}

fn parse_meta(contents: &str) -> Result<CheckpointMeta> {
    let mut vocab_size = None;
    let mut sequence_len = None;
    let mut n_layer = None;
    let mut n_head = None;
    let mut n_embd = None;
    let mut rope_base = None;
    let mut norm_eps = None;
    let mut step = None;
    let mut val_bpb = None;

    for (i, raw) in contents.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap();
        let Some(value) = parts.next() else {
            bail!(
                "checkpoint meta line {}: expected `key value`, got {raw:?}",
                i + 1
            );
        };
        if parts.next().is_some() {
            bail!(
                "checkpoint meta line {}: expected exactly `key value`, got {raw:?}",
                i + 1
            );
        }
        match key {
            "vocab_size" => vocab_size = Some(parse_field(key, value)?),
            "sequence_len" => sequence_len = Some(parse_field(key, value)?),
            "n_layer" => n_layer = Some(parse_field(key, value)?),
            "n_head" => n_head = Some(parse_field(key, value)?),
            "n_embd" => n_embd = Some(parse_field(key, value)?),
            "rope_base" => rope_base = Some(parse_field(key, value)?),
            "norm_eps" => norm_eps = Some(parse_field(key, value)?),
            "step" => step = Some(parse_field(key, value)?),
            "val_bpb" => val_bpb = Some(parse_field(key, value)?),
            other => bail!("checkpoint meta line {}: unknown key {other:?}", i + 1),
        }
    }

    let config = GptConfig {
        vocab_size: require(vocab_size, "vocab_size")?,
        sequence_len: require(sequence_len, "sequence_len")?,
        n_layer: require(n_layer, "n_layer")?,
        n_head: require(n_head, "n_head")?,
        n_embd: require(n_embd, "n_embd")?,
        rope_base: require(rope_base, "rope_base")?,
        norm_eps: require(norm_eps, "norm_eps")?,
    };
    Ok(CheckpointMeta {
        config,
        step: require(step, "step")?,
        val_bpb: require(val_bpb, "val_bpb")?,
    })
}

fn parse_field<T>(key: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value.parse::<T>().map_err(|e| {
        Error::msg(format!(
            "checkpoint meta: bad value for `{key}`: {value:?} ({e})"
        ))
    })
}

fn require<T>(opt: Option<T>, key: &str) -> Result<T> {
    opt.ok_or_else(|| Error::msg(format!("checkpoint meta: missing key `{key}`")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len: 16,
            n_layer: 2,
            n_head: 2,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    fn meta_eq(a: &CheckpointMeta, b: &CheckpointMeta) {
        let (x, y) = (&a.config, &b.config);
        assert_eq!(x.vocab_size, y.vocab_size);
        assert_eq!(x.sequence_len, y.sequence_len);
        assert_eq!(x.n_layer, y.n_layer);
        assert_eq!(x.n_head, y.n_head);
        assert_eq!(x.n_embd, y.n_embd);
        assert_eq!(
            x.rope_base, y.rope_base,
            "rope_base must round-trip exactly"
        );
        assert_eq!(x.norm_eps, y.norm_eps, "norm_eps must round-trip exactly");
        assert_eq!(a.step, b.step);
        assert_eq!(a.val_bpb, b.val_bpb);
    }

    #[test]
    fn meta_round_trips_and_ignores_blank_lines() {
        let meta = CheckpointMeta {
            config: tiny_cfg(),
            step: 4999,
            val_bpb: 1.234_567,
        };
        let mut text = write_meta(&meta);
        text.insert_str(0, "\n   \n"); // leading blank/whitespace lines must be skipped
        meta_eq(&meta, &parse_meta(&text).unwrap());
    }

    #[test]
    fn parse_rejects_invalid_meta() {
        // Key with no value, and key with two values.
        assert!(parse_meta("vocab_size\n").is_err());
        assert!(parse_meta("vocab_size 32 64\n").is_err());
        // Non-numeric value for a numeric field.
        assert!(parse_meta("vocab_size lots\n").is_err());
        // Unknown key.
        assert!(parse_meta("bogus 1\n").is_err());
        // A complete meta minus one required line must fail (not default-fill).
        let without_n_head: String = write_meta(&CheckpointMeta {
            config: tiny_cfg(),
            step: 1,
            val_bpb: 0.5,
        })
        .lines()
        .filter(|l| !l.starts_with("n_head "))
        .map(|l| format!("{l}\n"))
        .collect();
        assert!(parse_meta(&without_n_head).is_err());
    }

    #[test]
    fn checkpoint_round_trips_to_identical_logits() -> Result<()> {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let cfg = tiny_cfg();

        let vm_a = VarMap::new();
        let model_a = Gpt::new(cfg, VarBuilder::from_varmap(&vm_a, DType::F32, &dev))?;
        save(
            dir.path(),
            &vm_a,
            &CheckpointMeta {
                config: cfg,
                step: 4999,
                val_bpb: 1.234_567,
            },
        )?;

        // Meta field round-tripping is covered by `meta_round_trips_and_ignores_blank_lines`;
        // this test's job is to confirm the loaded weights reproduce the saved model.
        let (model_b, _vm_b, _meta) = load(dir.path(), &dev)?;

        let idx = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let la = model_a.forward(&idx)?.flatten_all()?.to_vec1::<f32>()?;
        let lb = model_b.forward(&idx)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            la, lb,
            "loaded model must reproduce the saved model's logits"
        );

        // Control: an independently initialized model has different weights, so
        // its logits differ — confirming the match above is the load, not that
        // every fresh model happens to agree.
        let vm_c = VarMap::new();
        let model_c = Gpt::new(cfg, VarBuilder::from_varmap(&vm_c, DType::F32, &dev))?;
        let lc = model_c.forward(&idx)?.flatten_all()?.to_vec1::<f32>()?;
        assert_ne!(la, lc, "a fresh model should not match by chance");
        Ok(())
    }
}
