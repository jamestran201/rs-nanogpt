use candle_core::backprop::GradStore;
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::VarMap;

use super::{GroupLrs, GroupedAdamW, lr_mult};
use crate::data::DataLoader;
use crate::model::{Gpt, Reduction, cross_entropy};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub num_iters: usize,
    pub grad_accum: usize,
    pub lrs: GroupLrs,
    pub warmup_steps: usize,
    pub warmdown_ratio: f64,
    pub final_lr_frac: f64,
    pub log_every: usize,
}

fn micro_backward(
    model: &Gpt,
    inputs: &Tensor,
    targets: &Tensor,
    grad_accum: usize,
) -> Result<(GradStore, Tensor)> {
    let logits = model.forward(inputs)?;
    let ce = cross_entropy(&logits, targets, -1, Reduction::Mean)?;
    let loss = (&ce * (1.0 / grad_accum as f64))?;
    let grads = loss.backward()?;
    Ok((grads, ce))
}

fn accumulate(acc: &mut GradStore, src: &GradStore, vars: &[Var]) -> Result<()> {
    for v in vars {
        let t = v.as_tensor();
        if let Some(g) = src.get(t) {
            let summed = match acc.get(t) {
                Some(a) => a.add(g)?,
                None => g.clone(),
            };
            acc.insert(t, summed);
        }
    }
    Ok(())
}

pub fn train(
    model: &Gpt,
    varmap: &VarMap,
    loader: &mut DataLoader,
    cfg: &TrainConfig,
    device: &Device,
) -> Result<()> {
    let mut opt = GroupedAdamW::new(varmap, cfg.lrs, model.config().n_embd)?;
    let vars = varmap.all_vars();
    for step in 0..cfg.num_iters {
        let logging = step % cfg.log_every == 0 || step + 1 == cfg.num_iters;
        let mut accum: Option<GradStore> = None;
        let mut ce_sum = 0.0f32;
        for _ in 0..cfg.grad_accum {
            let batch = loader.next_batch(device)?;
            let (grads, ce) = micro_backward(model, &batch.inputs, &batch.targets, cfg.grad_accum)?;
            if logging {
                ce_sum += ce.to_scalar::<f32>()?;
            }
            match &mut accum {
                None => accum = Some(grads),
                Some(acc) => accumulate(acc, &grads, &vars)?,
            }
        }
        let grads = accum.expect("grad_accum >= 1 guarantees one micro-batch ran");
        let m = lr_mult(
            step,
            cfg.num_iters,
            cfg.warmup_steps,
            cfg.warmdown_ratio,
            cfg.final_lr_frac,
        );
        opt.set_lr_mult(m);
        opt.step(&grads)?;
        if logging {
            // Mean micro-batch CE (the per-token loss), independent of grad_accum.
            println!(
                "step {step:>6}/{}  loss {:.4}  lr_mult {m:.4}",
                cfg.num_iters,
                ce_sum / cfg.grad_accum as f32
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GptConfig;
    use candle_core::DType;
    use candle_nn::VarBuilder;

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

    /// The accumulation contract: one backward over a 4-row batch equals two
    /// 2-row micro-batches summed (grad_accum = 2), per parameter. With all
    /// targets valid (no -1) each micro-batch has the same valid-token count, so
    /// the per-micro-batch means compose exactly into the full-batch mean:
    ///   0.5·∇mean_a + 0.5·∇mean_b = ∇((mean_a + mean_b)/2) = ∇mean_all.
    /// Compared on the same params before any step, so no init cloning needed.
    #[test]
    fn accumulation_matches_single_batch() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_embd = cfg.n_embd;
        let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let vars = vm.all_vars();

        let inputs = Tensor::new(
            &[
                [1u32, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ],
            &dev,
        )?;
        let targets = Tensor::new(
            &[
                [2i64, 3, 4, 5],
                [6, 7, 8, 9],
                [10, 11, 12, 13],
                [14, 15, 16, 17],
            ],
            &dev,
        )?;

        // One step off the zero-init residual projections: at strict init the
        // params behind the zeroed c_proj (c_q/c_k/c_v/c_fc) get exactly zero
        // gradient and candle stores no grad for them. After a step c_proj is
        // nonzero, the residual path is live, and every param has a real grad —
        // so the contract is tested on all params, not just a subset.
        {
            let lrs = GroupLrs {
                embedding: 0.01,
                unembedding: 0.01,
                matrix: 0.01,
            };
            let mut warm = GroupedAdamW::new(&vm, lrs, n_embd)?;
            let (g, _) = micro_backward(&model, &inputs, &targets, 1)?;
            warm.set_lr_mult(1.0);
            warm.step(&g)?;
        }

        // Single batch, grad_accum = 1.
        let (single, _) = micro_backward(&model, &inputs, &targets, 1)?;

        // Two micro-batches of 2 rows each, grad_accum = 2, summed.
        let (mut accum, _) = micro_backward(
            &model,
            &inputs.narrow(0, 0, 2)?,
            &targets.narrow(0, 0, 2)?,
            2,
        )?;
        let (g_b, _) = micro_backward(
            &model,
            &inputs.narrow(0, 2, 2)?,
            &targets.narrow(0, 2, 2)?,
            2,
        )?;
        accumulate(&mut accum, &g_b, &vars)?;

        // c_q/c_k reach the loss only through the softmax (scores) path, whose
        // gradient is ~zero right after a symmetric init, so candle stores no
        // grad for them; the rest (incl. c_v via the value path) get real grads.
        // The contract: present-and-equal, or absent-in-both — never a one-sided
        // presence (which would mean accumulation changed the gradient set).
        let mut compared = 0;
        for v in &vars {
            let t = v.as_tensor();
            match (single.get(t), accum.get(t)) {
                (Some(s), Some(a)) => {
                    let s = s.flatten_all()?.to_vec1::<f32>()?;
                    let a = a.flatten_all()?.to_vec1::<f32>()?;
                    for (x, y) in s.iter().zip(&a) {
                        // Relative tol: f32 sum reassociation, not a real mismatch.
                        assert!(
                            (x - y).abs() <= 1e-4 * (1.0 + x.abs()),
                            "grad mismatch: single {x} vs accum {y}"
                        );
                    }
                    compared += 1;
                }
                (None, None) => {}
                (s, a) => panic!(
                    "presence mismatch: single={} accum={}",
                    s.is_some(),
                    a.is_some()
                ),
            }
        }
        assert!(
            compared >= 8,
            "expected several params compared, got {compared}"
        );
        Ok(())
    }

    /// The stepping loop (micro_backward → schedule → grouped step) drives the
    /// loss down over many steps, finite throughout (grad_accum = 1).
    #[test]
    fn loop_drives_loss_down() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_embd = cfg.n_embd;
        let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;

        let inputs = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?;

        let tcfg = TrainConfig {
            num_iters: 50,
            grad_accum: 1,
            lrs: GroupLrs {
                embedding: 0.02,
                unembedding: 0.004,
                matrix: 0.01,
            },
            warmup_steps: 10,
            warmdown_ratio: 0.65,
            final_lr_frac: 0.05,
            log_every: 1000,
        };
        let mut opt = GroupedAdamW::new(&vm, tcfg.lrs, n_embd)?;

        let mut last = f32::NAN;
        let mut first = f32::NAN;
        for step in 0..=tcfg.num_iters {
            let (grads, ce) = micro_backward(&model, &inputs, &targets, tcfg.grad_accum)?;
            let l = ce.to_scalar::<f32>()?;
            assert!(l.is_finite(), "loss not finite at step {step}: {l}");
            if step == 0 {
                first = l;
            }
            last = l;
            let m = lr_mult(
                step,
                tcfg.num_iters,
                tcfg.warmup_steps,
                tcfg.warmdown_ratio,
                tcfg.final_lr_frac,
            );
            opt.set_lr_mult(m);
            opt.step(&grads)?;
        }
        assert!(last < first, "loss did not decrease: {first} -> {last}");
        Ok(())
    }
}
