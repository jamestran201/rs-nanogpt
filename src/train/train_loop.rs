use std::path::Path;
use std::time::Instant;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::VarMap;

use super::{GroupLrs, GroupedAdamW, lr_mult};
use crate::checkpoint::{self, CheckpointMeta};
use crate::data::{Batch, DataLoader};
use crate::eval::{evaluate, generate};
use crate::metrics::{MetricRecord, MetricsLogger, Throughput};
use crate::model::{Gpt, cross_entropy_sum_count};
use crate::tokenizer::BpeTokenizer;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub num_iters: usize,
    pub grad_accum: usize,
    /// Tokens per optimizer step: `grad_accum · device_batch · sequence_len`
    /// (the CLI's `--total-batch`). Asserted against real batch dims in the loop.
    pub tokens_per_step: usize,
    /// Enables the actual-valid-token grad rescale. Leave false when the data
    /// pipeline never emits `ignore_index` (all of pretraining): the fixed
    /// 1/expected pre-scale is then already exact, and skipping the rescale
    /// saves a full per-parameter multiply each step.
    pub targets_may_be_ignored: bool,
    pub lrs: GroupLrs,
    pub warmup_steps: usize,
    pub warmdown_ratio: f64,
    pub final_lr_frac: f64,
    pub log_every: usize,
}

pub struct EvalContext<'a> {
    pub val_batches: &'a [Batch],
    pub tokenizer: &'a BpeTokenizer,
    pub token_bytes: &'a [u32],
    pub ckpt_root: &'a Path,        // best model saved at ckpt_root/best/
    pub metrics: &'a MetricsLogger, // append-only JSONL telemetry sink
    // cadences + sampling params; 0 disables the cadence
    pub eval_every: usize,
    pub sample_every: usize,
    pub sample_tokens: usize,
    pub sample_temperature: f64,
}

const SAMPLE_PROMPTS: &[&str] = &[
    "",
    "The capital of France is",
    "The opposite of hot is",
    "My favorite color is",
];

/// Whether a hook on cadence `every` fires at `step` of a `0..=num_iters` loop.
fn cadence_fires(step: usize, num_iters: usize, every: usize) -> bool {
    if every == 0 {
        return false;
    }
    step == num_iters || (step > 0 && step.is_multiple_of(every))
}

/// Seconds as `HH:MM:SS`
fn fmt_hms(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
}

/// Abort the run if the logged loss is non-finite (NaN/Inf ⇒ divergence).
fn check_finite(loss: f32, step: usize) -> Result<()> {
    if !loss.is_finite() {
        candle_core::bail!("loss became non-finite ({loss}) at step {step}; aborting run");
    }
    Ok(())
}

/// Global L2 norm ‖g‖₂ of the accumulated per-step gradient. Sums each grad's
/// squared elements on-device and reads one scalar (one sync), so gate it behind
/// `logging` like the loss. The pre-Adam gradient magnitude is the standard
/// instability early-warning — it spikes before the loss goes NaN.
fn grad_global_norm(grads: &GradStore, vars: &[Var]) -> Result<f64> {
    let mut sumsq: Option<Tensor> = None;
    for v in vars {
        let t = v.as_tensor();
        if let Some(g) = grads.get(t) {
            let s = g.sqr()?.sum_all()?; // scalar tensor
            sumsq = Some(match sumsq {
                Some(a) => a.add(&s)?,
                None => s,
            });
        }
    }
    match sumsq {
        Some(t) => Ok((t.to_scalar::<f32>()? as f64).sqrt()),
        // Strict-init step: zeroed residual projections store no grad.
        None => Ok(0.0),
    }
}

/// One micro-batch forward/backward on `sum(nll) × inv_expected_tokens`,
/// where `inv_expected_tokens = 1/(grad_accum·B·T)`. Backwarding a *sum*
/// (not the micro-batch mean) is what lets unequal valid-token counts
/// compose correctly across micro-batches: the step's single division by the
/// actual total count happens later, in `rescale_grads`. The fixed pre-scale
/// by the *expected* count keeps intermediate gradient magnitudes where a
/// mean-based loss would have them, and makes the rescale a no-op (scale
/// exactly 1.0) whenever no target is ignored — all of pretraining.
///
/// Returns `(grads, Σ nll, Σ valid)`; the two scalars are f32 tensors, the
/// sum detached so accumulating it across micro-batches cannot pin each
/// micro-batch's whole graph.
fn micro_backward(
    model: &Gpt,
    inputs: &Tensor,
    targets: &Tensor,
    inv_expected_tokens: f64,
) -> Result<(GradStore, Tensor, Tensor)> {
    let logits = model.forward(inputs)?;
    let (nll_sum, valid_count) = cross_entropy_sum_count(&logits, targets, -1)?;
    let loss = (&nll_sum * inv_expected_tokens)?;
    let grads = loss.backward()?;
    Ok((grads, nll_sum.detach(), valid_count))
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

/// Multiply every accumulated grad by the scalar tensor `scale` (on-device,
/// no host sync). This corrects `micro_backward`'s fixed 1/expected pre-scale
/// to the step's actual valid-token count — the step gradient becomes the
/// true per-token mean over every scored token, however the tokens were
/// distributed across micro-batches.
fn rescale_grads(grads: &mut GradStore, vars: &[Var], scale: &Tensor) -> Result<()> {
    for v in vars {
        let t = v.as_tensor();
        if let Some(g) = grads.get(t) {
            let scaled = g.broadcast_mul(scale)?;
            grads.insert(t, scaled);
        }
    }
    Ok(())
}

pub fn train(
    model: &Gpt,
    varmap: &VarMap,
    loader: &mut DataLoader,
    cfg: &TrainConfig,
    eval: &EvalContext,
    device: &Device,
) -> Result<()> {
    let mut opt = GroupedAdamW::new(varmap, cfg.lrs, model.config().n_embd)?;
    let vars = varmap.all_vars();
    let mut min_val_bpb = f64::INFINITY;

    let t_start = Instant::now();
    let mut window_start_time = Instant::now(); // start of the current log window
    let mut window_start_step = 0usize; // step at window_start_time

    // Loss-scaling invariants, fixed for the whole run (see micro_backward
    // and rescale_grads for how they compose).
    let inv_expected_tokens = 1.0 / cfg.tokens_per_step as f64;
    let expected_tokens = Tensor::full(cfg.tokens_per_step as f32, (), device)?;

    for step in 0..=cfg.num_iters {
        let last = step == cfg.num_iters;

        if cadence_fires(step, cfg.num_iters, eval.eval_every) {
            let m = evaluate(model, eval.val_batches, eval.token_bytes)?;
            println!(
                "step {step:>6}  val_loss {:.4}  bpb {:.4}",
                m.val_loss, m.bpb
            );
            eval.metrics.log(&MetricRecord::eval(
                step,
                t_start.elapsed().as_secs_f64(),
                m.val_loss,
                m.bpb,
            ));
            if m.bpb < min_val_bpb {
                min_val_bpb = m.bpb;
                checkpoint::save(
                    &eval.ckpt_root.join("best"),
                    varmap,
                    &CheckpointMeta {
                        config: *model.config(),
                        step,
                        val_bpb: m.bpb,
                    },
                )?;
            }
        }

        if cadence_fires(step, cfg.num_iters, eval.sample_every) {
            for p in SAMPLE_PROMPTS {
                let s = generate(
                    model,
                    eval.tokenizer,
                    p,
                    eval.sample_tokens,
                    eval.sample_temperature,
                    step as u64,
                    device,
                )?;
                println!("sample: {s:?}");
            }
        }

        if last {
            break;
        }

        let logging = step.is_multiple_of(cfg.log_every);
        let mut accum: Option<GradStore> = None;
        let mut step_sum = Tensor::zeros((), DType::F32, device)?; // Σ nll over the step
        let mut step_count = Tensor::zeros((), DType::F32, device)?; // Σ valid tokens
        for _ in 0..cfg.grad_accum {
            let batch = loader.next_batch(device)?;
            let (b, t) = batch.inputs.dims2()?;
            assert_eq!(
                cfg.grad_accum * b * t,
                cfg.tokens_per_step,
                "batch shape ({b}, {t}) × grad_accum disagrees with cfg.tokens_per_step"
            );
            let (grads, nll_sum, valid_count) =
                micro_backward(model, &batch.inputs, &batch.targets, inv_expected_tokens)?;
            match &mut accum {
                None => accum = Some(grads),
                Some(acc) => accumulate(acc, &grads, &vars)?,
            }
            step_sum = (step_sum + &nll_sum)?;
            step_count = (step_count + &valid_count)?;
        }
        let mut grads = accum.expect("grad_accum >= 1 guarantees one micro-batch ran");

        if cfg.targets_may_be_ignored {
            // scale = expected/actual valid tokens, correcting the fixed
            // 1/expected pre-scale to the step's true count when ignored
            // targets make micro-batch counts unequal. Exactly 1.0 when they
            // don't (f32 counts are exact to 2^24, and IEEE guarantees
            // N/N == 1); with the flag off that holds for every step, so the
            // whole rescale is skipped.
            let scale = (&expected_tokens / &step_count)?;
            rescale_grads(&mut grads, &vars, &scale)?;
        }

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
            // True per-token CE over the step's valid tokens (a single global
            // sum/count, not a mean of per-micro-batch means); one scalar read
            // instead of the per-micro-batch reads a host-side sum would need.
            let mean_ce = step_sum.broadcast_div(&step_count)?.to_scalar::<f32>()?;
            check_finite(mean_ce, step)?;

            device.synchronize()?; // drain queued GPU work so the timing is real
            let now = Instant::now();
            let lrs = opt.current_lrs();
            let gnorm = grad_global_norm(&grads, &vars)?;
            let steps_in_window = step - window_start_step; // 0 only at the step-0 log
            let elapsed_s = now.duration_since(t_start).as_secs_f64();
            let elapsed = fmt_hms(elapsed_s);

            let rate = if steps_in_window > 0 {
                let window_secs = now.duration_since(window_start_time).as_secs_f64();
                let ms_per_step = 1000.0 * window_secs / steps_in_window as f64;
                let tok_s = (steps_in_window * cfg.tokens_per_step) as f64 / window_secs;
                let eta = fmt_hms(ms_per_step / 1000.0 * (cfg.num_iters - step) as f64);
                println!(
                    "step {step:>6}/{} | loss {mean_ce:.4} | gnorm {gnorm:.3} \
                     | lr m={:.2e} e={:.2e} u={:.2e} \
                     | {tok_s:.0} tok/s | {ms_per_step:.0} ms/step | t+{elapsed} | eta {eta}",
                    cfg.num_iters, lrs.matrix, lrs.embedding, lrs.unembedding,
                );
                Some(Throughput {
                    tok_per_s: tok_s,
                    ms_per_step,
                })
            } else {
                // step-0 window has no elapsed steps yet: baseline loss only.
                println!(
                    "step {step:>6}/{} | loss {mean_ce:.4} | gnorm {gnorm:.3} | t+{elapsed}",
                    cfg.num_iters
                );
                None
            };

            eval.metrics.log(&MetricRecord::train(
                step, elapsed_s, mean_ce, gnorm, lrs, rate,
            ));

            window_start_time = now;
            window_start_step = step;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GptConfig, Reduction, cross_entropy};
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

    /// One optimizer step off the zero-init residual projections: at strict
    /// init the params behind the zeroed c_proj (c_q/c_k/c_v/c_fc) get exactly
    /// zero gradient and candle stores no grad for them; after a step c_proj
    /// is nonzero and the residual path is live.
    fn warm_one_step(
        vm: &VarMap,
        model: &Gpt,
        inputs: &Tensor,
        targets: &Tensor,
        inv: f64,
    ) -> Result<()> {
        let lrs = GroupLrs {
            embedding: 0.01,
            unembedding: 0.01,
            matrix: 0.01,
        };
        let mut warm = GroupedAdamW::new(vm, lrs, model.config().n_embd)?;
        let (g, _, _) = micro_backward(model, inputs, targets, inv)?;
        warm.set_lr_mult(1.0);
        warm.step(&g)?;
        Ok(())
    }

    /// The grad-composition contract, per parameter: present-and-equal
    /// (relative tol — f32 sum reassociation, not a real mismatch), or absent
    /// in both stores; a one-sided presence means the two paths disagree on
    /// which params received gradient. Returns how many params were compared.
    fn compare_grads(want: &GradStore, got: &GradStore, vars: &[Var], label: &str) -> Result<usize> {
        let mut compared = 0;
        for v in vars {
            let t = v.as_tensor();
            match (want.get(t), got.get(t)) {
                (Some(w), Some(g)) => {
                    let w = w.flatten_all()?.to_vec1::<f32>()?;
                    let g = g.flatten_all()?.to_vec1::<f32>()?;
                    for (x, y) in w.iter().zip(&g) {
                        assert!(
                            (x - y).abs() <= 1e-4 * (1.0 + x.abs()),
                            "{label}: grad mismatch: {x} vs {y}"
                        );
                    }
                    compared += 1;
                }
                (None, None) => {}
                (w, g) => panic!(
                    "{label}: presence mismatch: want={} got={}",
                    w.is_some(),
                    g.is_some()
                ),
            }
        }
        Ok(compared)
    }

    /// The accumulation contract: one backward over a 4-row batch equals two
    /// 2-row micro-batches summed (grad_accum = 2), per parameter. Both paths
    /// pre-scale by the same fixed 1/16 (the step's expected token count), so
    /// the sums compose exactly:
    ///   ∇(sum_a/16) + ∇(sum_b/16) = ∇(sum_all/16) = ∇mean_all.
    /// All targets valid ⇒ the step's rescale would be a no-op (scale 1.0);
    /// the unequal-count case is covered by the sibling test below.
    /// Compared on the same params before any step, so no init cloning needed.
    #[test]
    fn accumulation_matches_single_batch() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
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

        warm_one_step(&vm, &model, &inputs, &targets, 1.0 / 16.0)?;

        // 16 expected tokens either way: 1·B(4)·T(4) single, 2·B(2)·T(4) split.
        let inv = 1.0 / 16.0;

        // Single batch, grad_accum = 1.
        let (single, _, _) = micro_backward(&model, &inputs, &targets, inv)?;

        // Two micro-batches of 2 rows each, grad_accum = 2, summed.
        let (mut accum, _, _) = micro_backward(
            &model,
            &inputs.narrow(0, 0, 2)?,
            &targets.narrow(0, 0, 2)?,
            inv,
        )?;
        let (g_b, _, _) = micro_backward(
            &model,
            &inputs.narrow(0, 2, 2)?,
            &targets.narrow(0, 2, 2)?,
            inv,
        )?;
        accumulate(&mut accum, &g_b, &vars)?;

        // c_q/c_k reach the loss only through the softmax (scores) path, whose
        // gradient is ~zero right after a symmetric init, so candle stores no
        // grad for them; the rest (incl. c_v via the value path) get real grads.
        let compared = compare_grads(&single, &accum, &vars, "single vs accum")?;
        assert!(
            compared >= 8,
            "expected several params compared, got {compared}"
        );
        Ok(())
    }

    /// The mean-of-means regression: micro-batches with *unequal* valid-token
    /// counts (via ignore_index) must still compose into the gradient of the
    /// global per-token mean. Per-micro-batch means would up-weight tokens in
    /// the sparse micro-batch ~2.7× here (5 vs 8 scored tokens); the
    /// sum + count + rescale pipeline keeps every scored token equal-weight.
    #[test]
    fn accumulation_with_unequal_valid_counts_matches_global_mean() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
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
        // Micro-batch A (rows 0-1) scores 5 tokens; B (rows 2-3) scores 8.
        let targets = Tensor::new(
            &[
                [2i64, 3, 4, 5],
                [6, -1, -1, -1],
                [10, 11, 12, 13],
                [14, 15, 16, 17],
            ],
            &dev,
        )?;

        warm_one_step(&vm, &model, &inputs, &targets, 1.0 / 16.0)?;

        // Ground truth: autograd through the global valid-token mean (13 tokens).
        let logits = model.forward(&inputs)?;
        let want = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.backward()?;

        // Accumulation path exactly as `train` runs it: fixed 1/expected
        // pre-scale per micro-batch, then one rescale to the actual count.
        let inv = 1.0 / 16.0; // grad_accum(2) · B(2) · T(4) expected tokens
        let (mut accum, _, count_a) = micro_backward(
            &model,
            &inputs.narrow(0, 0, 2)?,
            &targets.narrow(0, 0, 2)?,
            inv,
        )?;
        let (g_b, _, count_b) = micro_backward(
            &model,
            &inputs.narrow(0, 2, 2)?,
            &targets.narrow(0, 2, 2)?,
            inv,
        )?;
        accumulate(&mut accum, &g_b, &vars)?;
        let count = (count_a + count_b)?;
        assert_eq!(count.to_scalar::<f32>()?, 13.0);
        let scale = (Tensor::full(16.0f32, (), &dev)? / &count)?;
        rescale_grads(&mut accum, &vars, &scale)?;

        let compared = compare_grads(&want, &accum, &vars, "global-mean vs accum")?;
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
            tokens_per_step: 8, // grad_accum(1) · B(2) · T(4)
            targets_may_be_ignored: false,
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
            // grad_accum(1) · B(2) · T(4) = 8 expected tokens per step.
            let (grads, nll_sum, valid_count) =
                micro_backward(&model, &inputs, &targets, 1.0 / 8.0)?;
            let l = nll_sum.broadcast_div(&valid_count)?.to_scalar::<f32>()?;
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

    #[test]
    fn cadence_fires_predicate() {
        let n = 10;
        // Step 0 is always suppressed (untrained model); multiples of `every`
        // fire, and the final step always fires.
        assert!(!cadence_fires(0, n, 2));
        assert!(cadence_fires(2, n, 2));
        assert!(!cadence_fires(3, n, 2));
        assert!(cadence_fires(n, n, 2)); // last is a multiple here
        assert!(cadence_fires(n, n, 3)); // last fires even when it isn't a multiple

        // every == 0 disables the cadence entirely, including the final step.
        assert!(!cadence_fires(0, n, 0));
        assert!(!cadence_fires(n, n, 0));
    }

    #[test]
    fn fmt_hms_formats_and_saturates() {
        assert_eq!(fmt_hms(0.0), "00:00:00");
        assert_eq!(fmt_hms(3723.0), "01:02:03"); // 1h 2m 3s
        assert_eq!(fmt_hms(59.9), "00:00:59"); // truncates, not rounds
        assert_eq!(fmt_hms(-5.0), "00:00:00"); // saturates at zero
    }

    #[test]
    fn check_finite_rejects_nan_and_inf() {
        assert!(check_finite(1.5, 0).is_ok());
        assert!(check_finite(f32::NAN, 3).is_err());
        assert!(check_finite(f32::INFINITY, 3).is_err());
        assert!(check_finite(f32::NEG_INFINITY, 3).is_err());
    }

    #[test]
    fn grad_global_norm_matches_reference() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let vars = vm.all_vars();

        let inputs = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?;

        warm_one_step(&vm, &model, &inputs, &targets, 1.0 / 8.0)?;

        let (grads, _, _) = micro_backward(&model, &inputs, &targets, 1.0 / 8.0)?;

        // Independent f64 reference: sum of squares over every stored grad.
        let mut ref_sumsq = 0.0f64;
        for v in &vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                for x in g.flatten_all()?.to_vec1::<f32>()? {
                    ref_sumsq += (x as f64) * (x as f64);
                }
            }
        }
        let reference = ref_sumsq.sqrt();
        let got = grad_global_norm(&grads, &vars)?;
        assert!(reference > 0.0, "expected a nonzero gradient norm");
        assert!(
            (got - reference).abs() <= 1e-5 * (1.0 + reference),
            "grad_global_norm {got} vs reference {reference}"
        );
        Ok(())
    }

    /// No vars to match against ⇒ nothing summed ⇒ norm is 0.0 (the empty path a
    /// strict-init step exercises when no residual grads are stored).
    #[test]
    fn grad_global_norm_no_matching_vars_is_zero() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let inputs = Tensor::new(&[[1u32, 2, 3, 4]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5]], &dev)?;
        let (grads, _, _) = micro_backward(&model, &inputs, &targets, 1.0 / 4.0)?;
        assert_eq!(grad_global_norm(&grads, &[])?, 0.0);
        Ok(())
    }

    /// Capstone: a full tiny `train()` run must leave a loadable best checkpoint
    /// whose stored bpb the reloaded model reproduces exactly on the same val set.
    #[test]
    fn train_saves_loadable_best_checkpoint_matching_reported_bpb() -> Result<()> {
        use crate::data::Split;
        use crate::test_support::{byte_tokenizer, tiny_gpt, two_shard_corpus};

        let dev = Device::Cpu;
        let corpus = two_shard_corpus();
        let tok = byte_tokenizer();
        let token_bytes = tok.token_byte_lengths();

        let (b, t) = (2usize, 8usize);
        let (vm, model) = tiny_gpt(tok.vocab_size(), t);

        // Snapshot a fixed val set once; a separate train loader drives the steps.
        let mut val_loader =
            DataLoader::open_with_buffer_size(corpus.path(), Split::Val, &tok, b, t, 4).unwrap();
        let val_batches = val_loader.take_batches(2, &dev)?;
        let mut train_loader =
            DataLoader::open_with_buffer_size(corpus.path(), Split::Train, &tok, b, t, 4).unwrap();

        let out = tempfile::tempdir().unwrap();
        let tcfg = TrainConfig {
            num_iters: 4,
            grad_accum: 1,
            tokens_per_step: 16, // grad_accum(1) · B(2) · T(8)
            // All targets are valid, so scale == 1.0 exactly; true here keeps
            // the rescale branch of train() exercised.
            targets_may_be_ignored: true,
            lrs: GroupLrs {
                embedding: 0.02,
                unembedding: 0.004,
                matrix: 0.01,
            },
            warmup_steps: 1,
            warmdown_ratio: 0.5,
            final_lr_frac: 0.05,
            log_every: 1_000_000, // effectively silence the per-step loss log
        };
        let metrics = MetricsLogger::create(&out.path().join("metrics.jsonl")).unwrap();
        let eval = EvalContext {
            val_batches: &val_batches,
            tokenizer: &tok,
            token_bytes: &token_bytes,
            ckpt_root: out.path(),
            metrics: &metrics,
            eval_every: 2,
            sample_every: 0, // sampling off: output is noise on a 4-step run
            sample_tokens: 8,
            sample_temperature: 0.0,
        };

        train(&model, &vm, &mut train_loader, &tcfg, &eval, &dev)?;

        // Telemetry landed: at least the step-0 train record plus eval records.
        let metrics_len = std::fs::metadata(out.path().join("metrics.jsonl"))
            .unwrap()
            .len();
        assert!(
            metrics_len > 0,
            "metrics.jsonl should be non-empty after train()"
        );

        // The best checkpoint reloads, and re-scoring the *same* val batches on the
        // loaded weights reproduces the stored bpb bit-for-bit (identical f32
        // forward + f64 accumulation over an exact safetensors round-trip).
        let (loaded, _vm, meta) = checkpoint::load(&out.path().join("best"), &dev)?;
        let rescored = evaluate(&loaded, &val_batches, &token_bytes)?;
        assert_eq!(
            rescored.bpb, meta.val_bpb,
            "reloaded model must reproduce the saved bpb exactly"
        );
        Ok(())
    }
}
