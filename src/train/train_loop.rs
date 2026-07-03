use std::path::Path;
use std::time::Instant;

use candle_core::backprop::GradStore;
use candle_core::{Device, Result, Tensor, Var};
use candle_nn::VarMap;

use super::{GroupLrs, GroupedAdamW, lr_mult};
use crate::checkpoint::{self, CheckpointMeta};
use crate::data::{Batch, DataLoader};
use crate::eval::{evaluate, generate};
use crate::model::{Gpt, Reduction, cross_entropy};
use crate::tokenizer::BpeTokenizer;

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

pub struct EvalContext<'a> {
    pub val_batches: &'a [Batch],
    pub tokenizer: &'a BpeTokenizer,
    pub token_bytes: &'a [u32],
    pub ckpt_root: &'a Path, // best model saved at ckpt_root/best/
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
/// `every == 0` disables. `skip_first` suppresses step 0 (sampling; untrained
/// output is noise). The final step (`step == num_iters`) always fires.
fn cadence_fires(step: usize, num_iters: usize, every: usize, skip_first: bool) -> bool {
    if every == 0 {
        return false;
    }
    step == num_iters || ((!skip_first || step > 0) && step.is_multiple_of(every))
}

/// Seconds as `HH:MM:SS`, saturating (no days field — training runs are hours).
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
    eval: &EvalContext,
    device: &Device,
) -> Result<()> {
    let mut opt = GroupedAdamW::new(varmap, cfg.lrs, model.config().n_embd)?;
    let vars = varmap.all_vars();
    let mut min_val_bpb = f64::INFINITY;

    let t_start = Instant::now();
    let mut win_start = Instant::now(); // start of the current log window
    let mut win_step = 0usize; // step at win_start
    let mut tokens_per_step = 0usize; // set from the first micro-batch

    for step in 0..=cfg.num_iters {
        let last = step == cfg.num_iters;

        if cadence_fires(step, cfg.num_iters, eval.eval_every, false) {
            let m = evaluate(model, eval.val_batches, eval.token_bytes)?;
            println!(
                "step {step:>6}  val_loss {:.4}  bpb {:.4}",
                m.val_loss, m.bpb
            );
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

        if cadence_fires(step, cfg.num_iters, eval.sample_every, true) {
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
        let mut ce_sum = 0.0f32;
        for _ in 0..cfg.grad_accum {
            let batch = loader.next_batch(device)?;
            if tokens_per_step == 0 {
                // B*T*grad_accum; fixed across the run, so derive it once.
                tokens_per_step = cfg.grad_accum * batch.inputs.dim(0)? * batch.inputs.dim(1)?;
            }
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
            let mean_ce = ce_sum / cfg.grad_accum as f32;
            check_finite(mean_ce, step)?;

            device.synchronize()?; // drain queued GPU work so the timing is real
            let now = Instant::now();
            let lrs = opt.current_lrs();
            let dsteps = step - win_step; // 0 only at the step-0 log
            let elapsed = fmt_hms(now.duration_since(t_start).as_secs_f64());

            if dsteps > 0 {
                let win_s = now.duration_since(win_start).as_secs_f64();
                let ms_per_step = 1000.0 * win_s / dsteps as f64;
                let tok_s = (dsteps * tokens_per_step) as f64 / win_s;
                let eta = fmt_hms(ms_per_step / 1000.0 * (cfg.num_iters - step) as f64);
                println!(
                    "step {step:>6}/{} | loss {mean_ce:.4} \
                     | lr m={:.2e} e={:.2e} u={:.2e} \
                     | {tok_s:.0} tok/s | {ms_per_step:.0} ms/step | t+{elapsed} | eta {eta}",
                    cfg.num_iters, lrs.matrix, lrs.embedding, lrs.unembedding,
                );
            } else {
                // step-0 window has no elapsed steps yet: baseline loss only.
                println!("step {step:>6}/{} | loss {mean_ce:.4} | t+{elapsed}", cfg.num_iters);
            }
            win_start = now;
            win_step = step;
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
        // gradient and candle stores no grad for them; after a step c_proj is
        // nonzero and the residual path is live.
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

    #[test]
    fn cadence_fires_predicate() {
        let n = 10;
        // eval-style (skip_first = false): fires at step 0, multiples of `every`,
        // and the final step.
        assert!(cadence_fires(0, n, 2, false));
        assert!(cadence_fires(2, n, 2, false));
        assert!(!cadence_fires(3, n, 2, false));
        assert!(cadence_fires(n, n, 2, false)); // last is a multiple here
        assert!(cadence_fires(n, n, 3, false)); // last fires even when it isn't a multiple

        // sampling-style (skip_first = true): step 0 suppressed, everything else
        // identical.
        assert!(!cadence_fires(0, n, 2, true));
        assert!(cadence_fires(2, n, 2, true));
        assert!(cadence_fires(n, n, 2, true));

        // every == 0 disables the cadence entirely, including the final step.
        assert!(!cadence_fires(0, n, 0, false));
        assert!(!cadence_fires(n, n, 0, false));
        assert!(!cadence_fires(n, n, 0, true));
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
        let eval = EvalContext {
            val_batches: &val_batches,
            tokenizer: &tok,
            token_bytes: &token_bytes,
            ckpt_root: out.path(),
            eval_every: 2,
            sample_every: 0, // sampling off: output is noise on a 4-step run
            sample_tokens: 8,
            sample_temperature: 0.0,
        };

        train(&model, &vm, &mut train_loader, &tcfg, &eval, &dev)?;

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
