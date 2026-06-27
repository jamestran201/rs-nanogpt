use candle_core::backprop::GradStore;
use candle_core::{Result, Var, bail};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

#[derive(Debug, Clone, Copy)]
pub struct GroupLrs {
    pub embedding: f64,
    pub unembedding: f64,
    pub matrix: f64,
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
    pub fn new(varmap: &VarMap, lrs: GroupLrs, n_embd: usize) -> Result<Self> {
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
                weight_decay: 0.001,
            },
        )?;
        let unembedding = AdamW::new(
            g.unembedding,
            ParamsAdamW {
                lr: base_lrs.unembedding,
                beta1: 0.8,
                beta2: 0.96,
                eps: 1e-10,
                weight_decay: 0.01,
            },
        )?;
        let matrix = AdamW::new(
            g.matrix,
            ParamsAdamW {
                lr: base_lrs.matrix,
                beta1: 0.9,
                beta2: 0.95,
                eps: 1e-8,
                weight_decay: 0.1,
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
}

#[cfg(test)]
mod tests {
    use super::{Group, GroupLrs, GroupedAdamW, classify, partition};
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
        let mut opt = GroupedAdamW::new(&vm, lrs, n_embd)?;
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
        let mut opt = GroupedAdamW::new(&vm, lrs, n_embd)?;

        // Expected uses the muP scale to the *first* power; a double-applied
        // width factor would fail this, pinning the double-count risk.
        let scale = (n_embd as f64 / 768.0).powf(-0.5);
        let m = 0.5;
        opt.set_lr_mult(m);

        assert!((opt.embedding.learning_rate() - 0.2 * scale * m).abs() < 1e-12);
        assert!((opt.unembedding.learning_rate() - 0.004 * scale * m).abs() < 1e-12);
        assert!((opt.matrix.learning_rate() - 0.02 * scale * m).abs() < 1e-12);
        Ok(())
    }
}
