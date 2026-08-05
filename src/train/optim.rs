use candle_core::backprop::GradStore;
use candle_core::{Result, Var, bail};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

#[derive(Debug, Clone, Copy)]
pub struct GroupLrs {
    pub embedding: f64,
    pub unembedding: f64,
    pub matrix: f64,
}

/// Per-group AdamW weight decay. A phase-level policy knob: pretraining keeps
/// the defaults, SFT zeroes the matrix group.
///
/// Why only the matrix group for SFT: nanochat's chat finetune passes
/// `weight_decay=0.0` (`chat_sft.py:134`), which — traced through its optimizer
/// construction (`gpt.py:393-408`) — reaches only the Muon (matrix) param
/// groups; the AdamW embedding/unembedding decays are hardcoded at 0.001/0.01
/// in both phases. So the faithful port zeroes the matrix group alone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupWeightDecay {
    pub embedding: f64,
    pub unembedding: f64,
    pub matrix: f64,
}

impl Default for GroupWeightDecay {
    /// The values `GroupedAdamW` hardcoded before this became configurable;
    /// pretraining keeps them.
    fn default() -> Self {
        Self {
            embedding: 0.001,
            unembedding: 0.01,
            matrix: 0.1,
        }
    }
}

enum Group {
    Embedding,
    Unembedding,
    Matrix,
}

fn classify(name: &str) -> Result<Group> {
    match name {
        "wte.weight" => Ok(Group::Embedding),
        "lm_head.weight" => Ok(Group::Unembedding),
        n if n.starts_with("blocks.") => Ok(Group::Matrix),
        other => bail!("unrecognized parameter name for optimizer grouping: {other}"),
    }
}

#[derive(Default)]
struct Groups {
    embedding: Vec<Var>,
    unembedding: Vec<Var>,
    matrix: Vec<Var>,
}

fn partition(varmap: &VarMap) -> Result<Groups> {
    let data = varmap
        .data()
        .lock()
        .expect("VarMap lock should not be poisoned (would require another thread to panic while holding it)");
    let mut g = Groups::default();
    for (name, var) in data.iter() {
        match classify(name)? {
            Group::Embedding => g.embedding.push(var.clone()),
            Group::Unembedding => g.unembedding.push(var.clone()),
            Group::Matrix => g.matrix.push(var.clone()),
        }
    }
    Ok(g)
}

/// muP-flavored width scaling: LRs ∝ 1/√(n_embd/768), tuned for a 768-dim
/// reference (nanochat `gpt.py:389`). Applied once per group at construction.
/// Divergence: nanochat scales only its AdamW groups (matrices use unscaled
/// Muon); the MVP runs matrices through AdamW and scales all three uniformly.
fn mup_lr_scale(n_embd: usize) -> f64 {
    (n_embd as f64 / 768.0).powf(-0.5)
}

pub struct GroupedAdamW {
    embedding: AdamW,
    unembedding: AdamW,
    matrix: AdamW,
    base_lrs: GroupLrs,
}

impl GroupedAdamW {
    pub fn new(
        varmap: &VarMap,
        lrs: GroupLrs,
        n_embd: usize,
        wd: GroupWeightDecay,
    ) -> Result<Self> {
        let g = partition(varmap)?;
        let scale = mup_lr_scale(n_embd);
        let base_lrs = GroupLrs {
            embedding: lrs.embedding * scale,
            unembedding: lrs.unembedding * scale,
            matrix: lrs.matrix * scale,
        };

        let embedding = AdamW::new(
            g.embedding,
            ParamsAdamW {
                lr: base_lrs.embedding,
                beta1: 0.8,
                beta2: 0.995,
                eps: 1e-10,
                weight_decay: wd.embedding,
            },
        )?;
        let unembedding = AdamW::new(
            g.unembedding,
            ParamsAdamW {
                lr: base_lrs.unembedding,
                beta1: 0.8,
                beta2: 0.96,
                eps: 1e-10,
                weight_decay: wd.unembedding,
            },
        )?;
        let matrix = AdamW::new(
            g.matrix,
            ParamsAdamW {
                lr: base_lrs.matrix,
                beta1: 0.9,
                beta2: 0.95,
                eps: 1e-8,
                weight_decay: wd.matrix,
            },
        )?;

        Ok(Self {
            embedding,
            unembedding,
            matrix,
            base_lrs,
        })
    }

    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.embedding.step(grads)?;
        self.unembedding.step(grads)?;
        self.matrix.step(grads)
    }

    pub fn set_lr_mult(&mut self, m: f64) {
        self.embedding
            .set_learning_rate(self.base_lrs.embedding * m);
        self.unembedding
            .set_learning_rate(self.base_lrs.unembedding * m);
        self.matrix.set_learning_rate(self.base_lrs.matrix * m);
    }

    pub fn current_lrs(&self) -> GroupLrs {
        GroupLrs {
            embedding: self.embedding.learning_rate(),
            unembedding: self.unembedding.learning_rate(),
            matrix: self.matrix.learning_rate(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Group, GroupLrs, GroupWeightDecay, GroupedAdamW, classify, partition};
    use crate::model::{Gpt, GptConfig, Reduction, cross_entropy};
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

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

    /// Step 0: backward + AdamW step run end-to-end over the whole parameter
    /// set and move the loss down, finite throughout (no NaN through the causal
    /// -inf mask or the rms_norm sqrt). Isolates the mechanical core before
    /// group-partition / WSD / grad-accum stack on top.
    #[test]
    fn backward_step_decreases_loss() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), VarBuilder::from_varmap(&vm, DType::F32, &dev))?;

        // One fixed batch (B=2, T=4), reused every step.
        let inputs = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?;
        let loss_of = |m: &Gpt| -> Result<f32> {
            cross_entropy(&m.forward(&inputs)?, &targets, -1, Reduction::Mean)?.to_scalar::<f32>()
        };

        // lr 0.02 (vs the 0.001 default) so 20 steps move the loss
        // unambiguously; still well within stable range for this tiny model.
        let mut opt = AdamW::new(
            vm.all_vars(),
            ParamsAdamW {
                lr: 0.02,
                ..Default::default()
            },
        )?;

        let l0 = loss_of(&model)?;
        assert!(l0.is_finite(), "initial loss not finite: {l0}");

        for step in 0..20 {
            // Recompute the loss each step so it reflects the updated weights:
            // the `Var`s are shared by Arc, so stepping the optimizer mutates
            // the model in place.
            let loss = cross_entropy(&model.forward(&inputs)?, &targets, -1, Reduction::Mean)?;
            assert!(
                loss.to_scalar::<f32>()?.is_finite(),
                "loss not finite at step {step}"
            );
            opt.backward_step(&loss)?;
        }

        let l1 = loss_of(&model)?;
        assert!(l1 < l0, "loss did not decrease: {l0} -> {l1}");
        Ok(())
    }

    #[test]
    fn classify_routes_by_name() -> Result<()> {
        assert!(matches!(classify("wte.weight")?, Group::Embedding));
        assert!(matches!(classify("lm_head.weight")?, Group::Unembedding));
        assert!(matches!(
            classify("blocks.0.attn.c_q.weight")?,
            Group::Matrix
        ));
        assert!(matches!(
            classify("blocks.5.mlp.c_proj.weight")?,
            Group::Matrix
        ));
        assert!(classify("mystery.weight").is_err());
        Ok(())
    }

    #[test]
    fn partition_routes_every_param() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_layer = cfg.n_layer;
        let _model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;

        let g = partition(&vm)?;
        assert_eq!(g.embedding.len(), 1); // wte
        assert_eq!(g.unembedding.len(), 1); // lm_head
        assert_eq!(g.matrix.len(), 6 * n_layer); // 6 weights per block
        // Completeness: every var routed exactly once, none dropped/overlapping.
        let total = g.embedding.len() + g.unembedding.len() + g.matrix.len();
        assert_eq!(total, vm.all_vars().len());
        Ok(())
    }

    #[test]
    fn grouped_step_updates_all_groups() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_embd = cfg.n_embd;
        let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;

        let inputs = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?;

        let snapshot = |name: &str| -> Result<Vec<f32>> {
            let data = vm.data().lock().unwrap();
            data[name].as_tensor().flatten_all()?.to_vec1::<f32>()
        };

        let names = [
            "wte.weight",
            "lm_head.weight",
            "blocks.0.attn.c_proj.weight",
        ];
        let before: Vec<Vec<f32>> = names.iter().map(|&n| snapshot(n)).collect::<Result<_>>()?;

        let lrs = GroupLrs {
            embedding: 0.2,
            unembedding: 0.004,
            matrix: 0.02,
        };
        let mut opt = GroupedAdamW::new(&vm, lrs, n_embd, GroupWeightDecay::default())?;
        let loss = cross_entropy(&model.forward(&inputs)?, &targets, -1, Reduction::Mean)?;
        let grads = loss.backward()?;
        opt.step(&grads)?;

        for (&n, b) in names.iter().zip(&before) {
            let a = snapshot(n)?;
            assert!(
                a.iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-12),
                "group param {n} did not change after step"
            );
        }
        Ok(())
    }

    /// The whole content of the weight-decay seam is **per-group routing**, so
    /// that is what this pins: three distinct decays, and a crossed wire cannot
    /// hide behind a shared value.
    ///
    /// Mechanism: with a **zero** gradient candle's AdamW leaves only the
    /// decoupled decay term `θ·(1 − lr·wd)` — `m̂ = 0`, `v̂ = 0`, and
    /// `0/(0 + eps) = 0` kill the update (`candle-nn/src/optim.rs`). So each
    /// group's params shrink by exactly its own `1 − lr_g·wd_g`, where `lr_g` is
    /// the muP-scaled base LR.
    #[test]
    fn weight_decay_routes_to_the_right_group() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_embd = cfg.n_embd;
        let _model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;

        let lrs = GroupLrs {
            embedding: 0.2,
            unembedding: 0.004,
            matrix: 0.02,
        };
        let wd = GroupWeightDecay {
            embedding: 0.1,
            unembedding: 0.2,
            matrix: 0.3,
        };
        // The lr·wd *products* must stay distinct — equal ones would collapse
        // two factors and let a crossing pass.
        let scale = (n_embd as f64 / 768.0).powf(-0.5); // √96 ≈ 9.798
        let factor = |lr: f64, d: f64| (1.0 - lr * scale * d) as f32;
        let names = [
            ("wte.weight", factor(lrs.embedding, wd.embedding)),
            ("lm_head.weight", factor(lrs.unembedding, wd.unembedding)),
            ("blocks.0.attn.c_proj.weight", factor(lrs.matrix, wd.matrix)),
        ];
        for (i, (n, f)) in names.iter().enumerate() {
            for (m, g) in names.iter().skip(i + 1) {
                assert!(
                    (f - g).abs() > 1e-3,
                    "{n} and {m} share a decay factor ({f} vs {g}); the test cannot \
                     detect a crossed wire"
                );
            }
        }

        let snapshot = |name: &str| -> Result<Vec<f32>> {
            let data = vm.data().lock().unwrap();
            data[name].as_tensor().flatten_all()?.to_vec1::<f32>()
        };
        let before: Vec<Vec<f32>> = names
            .iter()
            .map(|(n, _)| snapshot(n))
            .collect::<Result<_>>()?;

        // A zero gradient for one var per group.
        let mut grads = candle_core::backprop::GradStore::default();
        {
            let data = vm.data().lock().unwrap();
            for (n, _) in &names {
                let t = data[*n].as_tensor();
                grads.insert(t, t.zeros_like()?);
            }
        }
        let mut opt = GroupedAdamW::new(&vm, lrs, n_embd, wd)?;
        opt.set_lr_mult(1.0);
        opt.step(&grads)?;

        for ((n, f), b) in names.iter().zip(&before) {
            for (after, &orig) in snapshot(n)?.iter().zip(b) {
                // Bit-exact: candle's `affine` casts the f64 factor to f32 and
                // multiplies, so this is the same arithmetic the step did —
                // asserting a computed ratio instead would need a tolerance.
                assert_eq!(
                    *after,
                    orig * f,
                    "group param {n} decayed by the wrong factor"
                );
            }
        }
        Ok(())
    }

    /// The SFT policy literal spells out to "pretraining's AdamW decays, matrix
    /// group off" — the whole of what nanochat's `weight_decay=0.0` does.
    #[test]
    fn sft_decay_literal_zeroes_only_the_matrix_group() {
        assert_eq!(
            GroupWeightDecay {
                matrix: 0.0,
                ..Default::default()
            },
            GroupWeightDecay {
                embedding: 0.001,
                unembedding: 0.01,
                matrix: 0.0,
            }
        );
    }

    #[test]
    fn set_lr_mult_scales_each_base_lr() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_embd = cfg.n_embd;
        let _model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;

        let lrs = GroupLrs {
            embedding: 0.2,
            unembedding: 0.004,
            matrix: 0.02,
        };
        let mut opt = GroupedAdamW::new(&vm, lrs, n_embd, GroupWeightDecay::default())?;

        // Expected uses the muP scale to the *first* power; a double-applied
        // width factor would fail this, pinning the double-count risk.
        let scale = (n_embd as f64 / 768.0).powf(-0.5);
        let m = 0.5;
        opt.set_lr_mult(m);

        assert!((opt.embedding.learning_rate() - 0.2 * scale * m).abs() < 1e-12);
        assert!((opt.unembedding.learning_rate() - 0.004 * scale * m).abs() < 1e-12);
        assert!((opt.matrix.learning_rate() - 0.02 * scale * m).abs() < 1e-12);

        // current_lrs() must report those same in-effect values.
        let cur = opt.current_lrs();
        assert!((cur.embedding - 0.2 * scale * m).abs() < 1e-12);
        assert!((cur.unembedding - 0.004 * scale * m).abs() < 1e-12);
        assert!((cur.matrix - 0.02 * scale * m).abs() < 1e-12);
        Ok(())
    }
}
