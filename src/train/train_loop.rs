use std::path::Path;
use std::time::Instant;

use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::VarMap;

use super::dist::{DistCtx, canonical_vars};
use super::mem::MemProbe;
use super::{GroupLrs, GroupWeightDecay, GroupedAdamW, lr_mult};
use crate::checkpoint::{self, CheckpointMeta};
use crate::data::{Batch, BatchSource};
use crate::eval::sample::{SampleOptions, generate_ids};
use crate::eval::{BpbAccumulator, evaluate_shard_sums};
use crate::metrics::{MetricRecord, MetricsLogger, Throughput};
use crate::model::{Gpt, cross_entropy_sum_count};
use crate::tokenizer::{BpeTokenizer, TokenId};

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub num_iters: usize,
    pub grad_accum: usize,
    /// *Per-rank* tokens per optimizer step: `grad_accum · device_batch ·
    /// sequence_len` (the CLI's `--total-batch` divided by `--gpus`). The
    /// cross-rank gradient `Avg` supplies the remaining 1/world factor, so the
    /// step gradient is the global per-token mean. Asserted against real batch
    /// dims in the loop.
    pub tokens_per_step: usize,
    /// Enables the actual-valid-token grad rescale. Leave false when the data
    /// pipeline never emits `ignore_index` (all of pretraining): the fixed
    /// 1/expected pre-scale is then already exact, and skipping the rescale
    /// saves a full per-parameter multiply each step.
    pub targets_may_be_ignored: bool,
    pub lrs: GroupLrs,
    /// Per-group AdamW weight decay. `Default::default()` is pretraining's
    /// (and the historical) setting; SFT zeroes the matrix group.
    pub weight_decays: GroupWeightDecay,
    pub warmup_steps: usize,
    pub warmdown_ratio: f64,
    pub final_lr_frac: f64,
    pub log_every: usize,
}

/// Gradient accumulation from the batch geometry: the global `total_batch` is
/// reached by whole micro-batches on every rank, so
/// `grad_accum = total_batch / (gpus · device_batch · sequence_len)`.
///
/// Shared by both training phases so their arithmetic and error text cannot
/// drift. `sequence_len` is a CLI flag for pretraining and comes from the
/// checkpoint meta for SFT, which is why it is a parameter rather than read
/// off a config here.
///
/// Rejects a zero `gpus` / `device_batch` / `sequence_len` **itself** rather
/// than trusting the caller to have checked: `usize::is_multiple_of(0)` is
/// `true` for `0`, so the divisibility test would pass and the division below
/// would panic. Pretraining happens to check those flags earlier, but SFT calls
/// this straight after reading the checkpoint meta.
pub fn validate_batch_geometry(
    gpus: usize,
    device_batch: usize,
    sequence_len: usize,
    total_batch: usize,
) -> std::result::Result<usize, String> {
    if gpus == 0 {
        return Err("gpus must be >= 1".into());
    }
    if device_batch == 0 {
        return Err("device_batch must be >= 1".into());
    }
    if sequence_len == 0 {
        return Err("sequence_len must be >= 1".into());
    }

    let per_step = gpus * device_batch * sequence_len;
    if !total_batch.is_multiple_of(per_step) {
        return Err(format!(
            "--total-batch ({total_batch}) must be a multiple of \
             gpus*device_batch*seq_len ({per_step})"
        ));
    }
    let grad_accum = total_batch / per_step;
    if grad_accum == 0 {
        return Err(format!(
            "--total-batch ({total_batch}) must be at least \
             gpus*device_batch*seq_len ({per_step})"
        ));
    }
    Ok(grad_accum)
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

/// The prefix an in-loop sample conditions on: a bare completion stem in base
/// mode, a primed chat turn (`<|user_start|>…<|assistant_start|>`) in chat
/// mode, which is what a finetuned model was actually trained to continue.
///
/// A named function because the sampling hook inside [`train`] cannot run in a
/// unit test — the same reason [`ignore_rescale`] is one.
fn sample_prefix(tok: &BpeTokenizer, prompt: &str, chat: bool) -> Vec<TokenId> {
    let mut ids = vec![tok.bos_id()];
    if chat {
        tok.push_user_turn(&mut ids, prompt);
    } else {
        ids.extend(tok.encode(prompt));
    }
    ids
}

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
/// micro-batch's whole graph. The returned store holds *only* parameter
/// gradients, all detached — see `retain_param_grads` for why that has to be
/// enforced rather than assumed.
fn micro_backward(
    model: &Gpt,
    inputs: &Tensor,
    targets: &Tensor,
    inv_expected_tokens: f64,
    vars: &[Var],
) -> Result<(GradStore, Tensor, Tensor)> {
    let logits = model.forward(inputs)?;
    let (nll_sum, valid_count) = cross_entropy_sum_count(&logits, targets, -1)?;
    let loss = (&nll_sum * inv_expected_tokens)?;
    let grads = retain_param_grads(loss.backward()?, vars);
    Ok((grads, nll_sum.detach(), valid_count))
}

/// Drop every `GradStore` entry that is not a parameter gradient.
///
/// candle's backward writes a gradient for *both* operands of every binary op,
/// constants included — rope's cos/sin (`model/rope.rs:58-59`) and the loss
/// mask (`model/loss.rs:38`). Those keys never enter `sorted_nodes` (`walk`
/// only pushes nodes with a `Var` upstream), and the pop out of `sorted_nodes`
/// is the *only* place candle detaches and removes. So each such entry keeps
/// `zeros ⊕ (g ⊙ activation)` with live op history and pins the forward graph.
/// Micro-batch 0's store becomes the step accumulator, so without this prune
/// one entire forward graph stays alive for the whole optimizer step — ~24.5 GB
/// at d24, measured (`writeups/pretrain-oom-investigation.md`).
///
/// Numerically inert: `accumulate`, `rescale_grads`, `DistCtx::all_reduce_grads`,
/// `opt.step` and `grad_global_norm` all look gradients up *by var*, so the
/// dropped entries are written once by candle and never read by anything.
///
/// `vars` is used as a set — unlike the collectives, the order does not matter.
fn retain_param_grads(mut grads: GradStore, vars: &[Var]) -> GradStore {
    let mut kept = GradStore::default();
    for v in vars {
        let t = v.as_tensor();
        // `remove` moves the tensor across instead of cloning. A var with no
        // stored gradient simply stays absent: that is the strict-init step,
        // where the zeroed residual projections get exactly zero and candle
        // stores nothing for them.
        if let Some(g) = grads.remove(t) {
            kept.insert(t, g);
        }
    }
    kept // `grads` drops here, releasing the graph-pinning entries
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

/// `expected · world / global_count` — the correction from `micro_backward`'s
/// fixed 1/expected pre-scale to the step's true **global** per-token mean.
///
/// With `E` = per-rank expected tokens (`cfg.tokens_per_step`), `W` = world and
/// `C = Σ_r C_r` the global valid count, each rank enters the gradient `Avg`
/// holding `g_r = ∇(Σ_r nll)/E`, and the `Avg` supplies a 1/W. For the step
/// gradient to come out as `∇(Σ_all nll)/C`, each rank must pre-multiply by
/// `E·W/C` — and `E·W` is just the global `--total-batch`.
///
/// At `W = 1` this collapses to today's `E/C_0` bit for bit: candle's
/// `Tensor * f64` is `affine(1.0, 0.0)`, which is exact.
///
/// A named function rather than an expression inlined in the loop: the `W > 1`
/// branch of [`train`] cannot run on a dev machine, so the test has to be able
/// to call the real thing (`two_rank_rescale_matches_the_global_per_token_mean`).
fn ignore_rescale(expected_tokens: &Tensor, world: usize, global_count: &Tensor) -> Result<Tensor> {
    &(expected_tokens * world as f64)? / global_count
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

/// `&mut dyn`, not a generic: `next_batch` is called once per micro-batch,
/// next to a full forward/backward, so the vtable dispatch is free — and
/// `train()` stays a single instantiation instead of one per source type.
pub fn train(
    model: &Gpt,
    varmap: &VarMap,
    batches: &mut dyn BatchSource,
    cfg: &TrainConfig,
    eval: &EvalContext,
    dist: &DistCtx,
    device: &Device,
) -> Result<()> {
    let mut opt = GroupedAdamW::new(varmap, cfg.lrs, model.config().n_embd, cfg.weight_decays)?;
    // Canonical (name-sorted) order: NCCL matches collectives across ranks by
    // call order alone, so every per-var walk below uses this one list. The
    // local uses (accumulate/rescale/norm) are order-insensitive anyway.
    let vars: Vec<Var> = canonical_vars(varmap).into_iter().map(|(_, v)| v).collect();
    // One-time init sync: every rank takes rank 0's random init, so replicas
    // start identical without relying on cross-GPU RNG agreement. From then
    // on the per-step gradient Avg plus deterministic AdamW keeps them
    // identical. No-op in a world of 1.
    dist.broadcast_vars(&vars)?;
    let mut min_val_bpb = f64::INFINITY;

    let t_start = Instant::now();
    let mut window_start_time = Instant::now(); // start of the current log window
    let mut window_start_step = 0usize; // step at window_start_time

    // Loss-scaling invariants, fixed for the whole run (see micro_backward
    // and rescale_grads for how they compose).
    let inv_expected_tokens = 1.0 / cfg.tokens_per_step as f64;
    let expected_tokens = Tensor::full(cfg.tokens_per_step as f32, (), device)?;

    // CUDA async-mempool telemetry for the OOM investigation; `None` unless
    // RS_NANOGPT_MEM_TRACE asks for it (see `super::mem`). Master-only: every
    // rank's pool behaves alike and one stream of lines is what gets read.
    let probe = if dist.is_master() {
        MemProbe::from_env(device)?
    } else {
        None
    };
    if let Some(p) = &probe {
        // The one sync the probe adds, and it is outside the steady state:
        // there is no cross-step allocation history to perturb yet, and it
        // makes the baseline exact (masters + moments ≈ 9.4 GB at d24).
        device.synchronize()?;
        let s = p.stats()?;
        println!("mem init         | {}", s.summary());
        eval.metrics
            .log(&MetricRecord::mem(0, t_start.elapsed().as_secs_f64(), s));
        p.reset_highs()?;
    }

    for step in 0..=cfg.num_iters {
        let last = step == cfg.num_iters;

        if cadence_fires(step, cfg.num_iters, eval.eval_every) {
            // Every rank scores its 1/world slice of the shared val snapshot;
            // the reduced sums give all ranks the identical metrics (which
            // keeps the best-bpb control flow below in lockstep for free).
            let local = evaluate_shard_sums(
                model,
                eval.val_batches,
                eval.token_bytes,
                dist.rank(),
                dist.world_size(),
            )?;
            let sums: [f64; 4] = dist
                .all_reduce_sums(&local)?
                .try_into()
                .expect("all_reduce_sums preserves length");
            let m = BpbAccumulator::metrics_from_sums(sums);
            if dist.is_master() {
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
            }
            // Every rank tracks the best bpb (identical value everywhere, so
            // control flow stays in lockstep); only master writes the files.
            if m.bpb < min_val_bpb {
                min_val_bpb = m.bpb;
                if dist.is_master() {
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
        }

        if dist.is_master() && cadence_fires(step, cfg.num_iters, eval.sample_every) {
            let opts = SampleOptions {
                max_tokens: eval.sample_tokens,
                temperature: eval.sample_temperature,
                // Greedy here, so the top-k restriction would do nothing.
                top_k: 0,
                seed: step as u64,
            };
            for p in SAMPLE_PROMPTS {
                let prefix = sample_prefix(eval.tokenizer, p, false);
                let ids = generate_ids(model, &prefix, opts, &[], device, |_| Ok(()))?;
                // decode_lossy, not decode: a token budget cuts a byte-level
                // vocab mid-character routinely, and decode's panic would kill
                // a multi-hour run from inside the training loop.
                let out = eval.tokenizer.decode_lossy(&ids);
                // The prompt too — `generate_ids` returns the continuation
                // alone, and one of the prompts is "".
                println!("sample {p:?} -> {out:?}");
            }
        }

        if last {
            break;
        }

        let logging = step.is_multiple_of(cfg.log_every);
        let mut accum: Option<GradStore> = None;
        let mut step_sum = Tensor::zeros((), DType::F32, device)?; // Σ nll over the step
        let mut step_count = Tensor::zeros((), DType::F32, device)?; // Σ valid tokens
        for micro in 0..cfg.grad_accum {
            let batch = batches.next_batch(device)?;
            let (b, t) = batch.inputs.dims2()?;
            assert_eq!(
                cfg.grad_accum * b * t,
                cfg.tokens_per_step,
                "batch shape ({b}, {t}) × grad_accum disagrees with cfg.tokens_per_step \
                 (per-rank tokens: total_batch / world_size)"
            );
            let (grads, nll_sum, valid_count) = match micro_backward(
                model,
                &batch.inputs,
                &batch.targets,
                inv_expected_tokens,
                &vars,
            ) {
                Ok(v) => v,
                Err(e) => {
                    // Essentially all allocation happens inside the
                    // forward/backward, so this is where an OOM surfaces.
                    // Capture the pool at the failure instant — `dev_free`
                    // in particular bounds the request that could not be
                    // placed — before the error unwinds out of `train`.
                    if let Some(p) = &probe {
                        p.dump(&format!("OOM step {step} micro {micro}"));
                    }
                    return Err(e);
                }
            };
            match &mut accum {
                None => accum = Some(grads),
                Some(acc) => accumulate(acc, &grads, &vars)?,
            }
            step_sum = (step_sum + &nll_sum)?;
            step_count = (step_count + &valid_count)?;
            if let Some(p) = &probe
                && p.per_micro()
            {
                // No sync: `cuMemAllocAsync` grows `reserved` when the request
                // is *made*, so the enqueue-order view is the one that shows
                // where inside a step the pool actually grows.
                println!(
                    "mem step {step:>6} micro {micro:>3} | {}",
                    p.stats()?.summary()
                );
            }
        }
        let mut grads = accum.expect("grad_accum >= 1 guarantees one micro-batch ran");

        if cfg.targets_may_be_ignored {
            // Correct the fixed 1/expected pre-scale to the step's true count
            // when ignored targets make micro-batch counts unequal. At world 1
            // with nothing ignored the scale is exactly 1.0 (f32 counts are
            // exact to 2^24, and IEEE guarantees N/N == 1); with the flag off
            // that holds for every step, so the whole rescale is skipped.
            //
            // The count is `Sum`-reduced across ranks *first*, so the scale is
            // built from the **global** valid count: the step gradient is then
            // the exact global per-token mean rather than a mean of per-rank
            // means (see `ignore_rescale`). It is also what keeps a rank whose
            // micro-batches happen to score nothing from producing
            // `scale = inf` and spreading `0 × inf = NaN` to every replica
            // through the gradient Avg — with a global count it simply
            // contributes zero.
            //
            // Exactness: the counts are integers bounded by --total-batch and
            // f32 is exact to 2^24 = 16.7 M, so both the per-rank counts and
            // their cross-rank sum are exact for any plausible batch size.
            // The reduce runs on-device, so it adds no host sync (the loop
            // syncs only on logging steps, below).
            let global_count = dist.all_reduce_sum_tensor(&step_count)?;
            let scale = ignore_rescale(&expected_tokens, dist.world_size(), &global_count)?;
            rescale_grads(&mut grads, &vars, &scale)?;
        }

        // Cross-rank gradient averaging — the whole of DDP. Runs *every*
        // step; the NaN-abort argument in the logging block below leans on
        // that (a NaN on any rank spreads to all replicas through this Avg),
        // so revisit decision 7 of the plan if it is ever made conditional.
        dist.all_reduce_grads(&mut grads, &vars)?;

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
            // sum/count, not a mean of per-micro-batch means). The two scalars
            // are Sum-reduced across ranks so the logged loss is the *global*
            // mean and — since every rank now computes the identical value —
            // the NaN abort below fires on all ranks together instead of
            // leaving peers hung in the next collective. (A NaN born on one
            // rank at a non-log step reaches every replica within one step
            // via the gradient Avg above, so the abort lags by at most
            // log_every steps.) Exactness: the per-rank f32 count is exact to
            // 2^24 tokens and the reduce sums in f64, so the post-reduce
            // count — *global* tokens per step — is exact for any plausible
            // --total-batch. The final division stays in f32 to match the
            // single-GPU path bit for bit at world 1.
            let local = [
                step_sum.to_scalar::<f32>()? as f64,
                step_count.to_scalar::<f32>()? as f64,
            ];
            let global = dist.all_reduce_sums(&local)?;
            let mean_ce = global[0] as f32 / global[1] as f32;
            check_finite(mean_ce, step)?;

            device.synchronize()?; // drain queued GPU work so the timing is real
            // Ride the sync above instead of adding one. `grads` is still live
            // here, so this reads the pool at the same instant the on-box
            // trace's per-step floor was measured; sampling after `grads`
            // drops would amount to applying the candidate fix (writeup step
            // B) under the guise of measuring the bug. Sampled before
            // `grad_global_norm` below so its transients are not counted.
            if let Some(p) = &probe {
                let s = p.stats()?;
                println!("mem step {step:>6}   | {}", s.summary());
                eval.metrics
                    .log(&MetricRecord::mem(step, t_start.elapsed().as_secs_f64(), s));
                p.reset_highs()?;
            }
            let now = Instant::now();
            if dist.is_master() {
                // Post-reduce the gradients are identical on every rank, so
                // the norm is too: compute and log it on master only.
                let lrs = opt.current_lrs();
                let gnorm = grad_global_norm(&grads, &vars)?;
                let steps_in_window = step - window_start_step; // 0 only at the step-0 log
                let elapsed_s = now.duration_since(t_start).as_secs_f64();
                let elapsed = fmt_hms(elapsed_s);

                let rate = if steps_in_window > 0 {
                    let window_secs = now.duration_since(window_start_time).as_secs_f64();
                    let ms_per_step = 1000.0 * window_secs / steps_in_window as f64;
                    // tokens_per_step is per-rank; report global throughput.
                    let tok_s = (steps_in_window * cfg.tokens_per_step * dist.world_size()) as f64
                        / window_secs;
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
            }

            window_start_time = now;
            window_start_step = step;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
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
        let mut warm =
            GroupedAdamW::new(vm, lrs, model.config().n_embd, GroupWeightDecay::default())?;
        let (g, _, _) = micro_backward(model, inputs, targets, inv, &vm.all_vars())?;
        warm.set_lr_mult(1.0);
        warm.step(&g)?;
        Ok(())
    }

    /// The grad-composition contract, per parameter: present-and-equal
    /// (relative tol — f32 sum reassociation, not a real mismatch), or absent
    /// in both stores; a one-sided presence means the two paths disagree on
    /// which params received gradient. Returns how many params were compared.
    fn compare_grads(
        want: &GradStore,
        got: &GradStore,
        vars: &[Var],
        label: &str,
    ) -> Result<usize> {
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

    /// The invariant `micro_backward` guarantees, and the regression test for
    /// the OOM root cause: the returned store holds *only* parameter
    /// gradients, every one of them detached. A single entry with live op
    /// history pins the whole forward graph until the store drops — and since
    /// micro-batch 0's store becomes the step accumulator, that means one
    /// forward graph per step (~24.5 GB at d24, measured).
    #[test]
    fn micro_backward_returns_only_detached_param_grads() -> Result<()> {
        use std::collections::HashSet;

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let model = Gpt::new(tiny_cfg(), VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let vars = vm.all_vars();
        let inputs = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?;

        let (grads, _, _) = micro_backward(&model, &inputs, &targets, 1.0 / 8.0, &vars)?;

        let var_ids: HashSet<_> = vars.iter().map(|v| v.as_tensor().id()).collect();
        let mut kept = 0usize;
        for id in grads.get_ids() {
            assert!(
                var_ids.contains(id),
                "store holds an entry that is not a parameter gradient"
            );
            let g = grads.get_id(*id).expect("id came from the store");
            assert_eq!(
                g.sorted_nodes().len(),
                0,
                "a stored gradient still reaches graph nodes, so it pins activations"
            );
            kept += 1;
        }
        // The prune kept real work rather than emptying the store. Fewer than
        // `vars.len()` is expected at strict init: the zeroed residual
        // projections get exactly zero gradient and candle stores nothing.
        assert!(kept > 0 && kept <= vars.len(), "kept {kept}/{}", vars.len());
        Ok(())
    }

    /// Why `retain_param_grads` has to exist, pinned to candle's actual
    /// behaviour so a candle bump cannot silently make it dead code.
    ///
    /// candle writes a gradient for *both* operands of every binary op,
    /// constants included, but only detaches and removes entries it pops out
    /// of `sorted_nodes` (`backprop.rs:174-182`) — and `walk` populates that
    /// list with grad-*tracking* nodes only. So a raw store comes back holding
    /// activation-shaped entries keyed by rope's cos/sin broadcasts (4 per
    /// `Rope::apply` × q and k = 8 per block, `model/rope.rs:58-59`) plus one
    /// for the loss mask (`model/loss.rs:38`), each carrying live op history.
    ///
    /// If this fails after a candle upgrade, candle's backward changed: work
    /// out whether the prune is still needed before deleting it.
    #[test]
    fn retain_param_grads_drops_the_graph_pinning_entries() -> Result<()> {
        use std::collections::HashSet;

        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let model = Gpt::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev))?;
        let vars = vm.all_vars();
        let inputs = Tensor::new(&[[1u32, 2, 3, 4], [5, 6, 7, 8]], &dev)?;
        let targets = Tensor::new(&[[2i64, 3, 4, 5], [6, 7, 8, 9]], &dev)?;

        // The unpruned store, exactly as candle hands it back.
        let logits = model.forward(&inputs)?;
        let (nll_sum, _) = cross_entropy_sum_count(&logits, &targets, -1)?;
        let raw = nll_sum.backward()?;

        let var_ids: HashSet<_> = vars.iter().map(|v| v.as_tensor().id()).collect();
        let mut non_param = 0usize;
        let mut pinned = 0usize; // graph nodes reachable from a non-param entry
        for id in raw.get_ids() {
            if !var_ids.contains(id) {
                non_param += 1;
                let g = raw.get_id(*id).expect("id came from the store");
                pinned = pinned.max(g.sorted_nodes().len());
            }
        }
        assert_eq!(
            non_param,
            8 * cfg.n_layer + 1,
            "unexpected non-parameter entry count in candle's raw store"
        );
        assert!(
            pinned > 0,
            "non-parameter entries pin nothing — candle's backward may have changed"
        );

        // And the prune removes exactly those.
        let pruned = retain_param_grads(raw, &vars);
        assert!(
            pruned.get_ids().all(|id| var_ids.contains(id)),
            "prune left a non-parameter entry behind"
        );
        Ok(())
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
        let (single, _, _) = micro_backward(&model, &inputs, &targets, inv, &vars)?;

        // Two micro-batches of 2 rows each, grad_accum = 2, summed.
        let (mut accum, _, _) = micro_backward(
            &model,
            &inputs.narrow(0, 0, 2)?,
            &targets.narrow(0, 0, 2)?,
            inv,
            &vars,
        )?;
        let (g_b, _, _) = micro_backward(
            &model,
            &inputs.narrow(0, 2, 2)?,
            &targets.narrow(0, 2, 2)?,
            inv,
            &vars,
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
            &vars,
        )?;
        let (g_b, _, count_b) = micro_backward(
            &model,
            &inputs.narrow(0, 2, 2)?,
            &targets.narrow(0, 2, 2)?,
            inv,
            &vars,
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

    /// The DDP form of the rescale, simulated without NCCL: two ranks with
    /// *different* valid-token counts must compose into the gradient of the
    /// **global** per-token mean, not a mean of per-rank means.
    ///
    /// `train()` performs exactly this composition — `micro_backward` per rank,
    /// one `ignore_rescale`, then the cross-rank `Avg` — so running it by hand
    /// covers the `world > 1` branch that neither dev machine can execute.
    /// Calling `ignore_rescale` (rather than writing `16/13` here) is what makes
    /// a dropped `world` factor in the implementation fail this test.
    #[test]
    fn two_rank_rescale_matches_the_global_per_token_mean() -> Result<()> {
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
        // Rank A (rows 0-1) scores 5 tokens; rank B (rows 2-3) scores 8.
        let targets = Tensor::new(
            &[
                [2i64, 3, 4, 5],
                [6, -1, -1, -1],
                [10, 11, 12, 13],
                [14, 15, 16, 17],
            ],
            &dev,
        )?;

        // Must come first: at strict init the zeroed c_proj leaves
        // c_q/c_k/c_v/c_fc with exactly zero gradient and candle stores
        // nothing, so a scale error would be invisible on those params.
        warm_one_step(&vm, &model, &inputs, &targets, 1.0 / 16.0)?;

        // Ground truth: autograd through the global valid-token mean (13).
        let logits = model.forward(&inputs)?;
        let want = cross_entropy(&logits, &targets, -1, Reduction::Mean)?.backward()?;

        // E = grad_accum(1) · B(2) · T(4) = 8 expected tokens *per rank*.
        let inv = 1.0 / 8.0;
        let (mut g_a, _, count_a) = micro_backward(
            &model,
            &inputs.narrow(0, 0, 2)?,
            &targets.narrow(0, 0, 2)?,
            inv,
            &vars,
        )?;
        let (mut g_b, _, count_b) = micro_backward(
            &model,
            &inputs.narrow(0, 2, 2)?,
            &targets.narrow(0, 2, 2)?,
            inv,
            &vars,
        )?;

        // The global count an `all_reduce_sum_tensor` would produce.
        let global_count = (count_a + count_b)?;
        assert_eq!(global_count.to_scalar::<f32>()?, 13.0);
        let scale = ignore_rescale(&Tensor::full(8f32, (), &dev)?, 2, &global_count)?;
        rescale_grads(&mut g_a, &vars, &scale)?;
        rescale_grads(&mut g_b, &vars, &scale)?;

        // The cross-rank `Avg` (1/world), by hand.
        let mut got = GradStore::default();
        for v in &vars {
            let t = v.as_tensor();
            if let (Some(a), Some(b)) = (g_a.get(t), g_b.get(t)) {
                got.insert(t, ((a + b)? * 0.5)?);
            }
        }

        // ∇Σ_A · 16/(8·13) + ∇Σ_B · 16/(8·13), halved, is ∇Σ_all/13 — exactly
        // what Reduction::Mean divides by. A per-rank scale would give
        // 0.5(∇Σ_A/5 + ∇Σ_B/8); a missing world factor, ∇Σ_all/26.
        let compared = compare_grads(&want, &got, &vars, "global-mean vs two-rank")?;
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
            weight_decays: GroupWeightDecay::default(),
            warmup_steps: 10,
            warmdown_ratio: 0.65,
            final_lr_frac: 0.05,
            log_every: 1000,
        };
        let mut opt = GroupedAdamW::new(&vm, tcfg.lrs, n_embd, tcfg.weight_decays)?;

        let mut last = f32::NAN;
        let mut first = f32::NAN;
        for step in 0..=tcfg.num_iters {
            // grad_accum(1) · B(2) · T(4) = 8 expected tokens per step.
            let (grads, nll_sum, valid_count) =
                micro_backward(&model, &inputs, &targets, 1.0 / 8.0, &vm.all_vars())?;
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

    /// The geometry both phases share: accepts the pretrain defaults, and
    /// `grad_accum` is the whole-micro-batch quotient, so it scales with every
    /// factor of the denominator.
    #[test]
    fn validate_batch_geometry_computes_grad_accum() {
        // 32 · 512 = 16384 per rank-step ⇒ grad_accum 1 at one GPU.
        assert_eq!(validate_batch_geometry(1, 32, 512, 16384), Ok(1));
        assert_eq!(validate_batch_geometry(1, 32, 512, 4 * 16384), Ok(4));
        // Adding ranks divides the per-rank work, not the global batch.
        assert_eq!(validate_batch_geometry(8, 32, 512, 8 * 16384), Ok(1));
        assert_eq!(validate_batch_geometry(8, 32, 512, 32 * 16384), Ok(4));
    }

    /// The zeros this rejects itself. `is_multiple_of(0)` is `true` for `0`, so
    /// without these the divisibility test passes and the division panics —
    /// reachable from SFT, which calls this straight after `load_meta`.
    #[test]
    fn validate_batch_geometry_rejects_zero_factors_before_dividing() {
        assert!(validate_batch_geometry(0, 32, 512, 16384).is_err());
        assert!(validate_batch_geometry(1, 0, 512, 16384).is_err());
        assert!(validate_batch_geometry(1, 32, 0, 16384).is_err());
    }

    #[test]
    fn validate_batch_geometry_rejects_a_total_batch_that_does_not_fit() {
        // Not a whole number of micro-batches.
        assert!(validate_batch_geometry(1, 32, 512, 16384 + 1).is_err());
        // A multiple of the per-step size, but less than one of them.
        assert!(validate_batch_geometry(1, 32, 512, 0).is_err());
        // Enough for one rank's micro-batch, but not for two ranks'.
        assert!(validate_batch_geometry(2, 32, 512, 16384).is_err());
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

    /// byte_tokenizer: byte value b = token id b, bos=256, user_start=257,
    /// user_end=258, assistant_start=259.
    #[test]
    fn sample_prefix_switches_format() {
        let tok = crate::test_support::byte_tokenizer();
        assert_eq!(sample_prefix(&tok, "hi", false), [256, 104, 105]);
        assert_eq!(
            sample_prefix(&tok, "hi", true),
            [256, 257, 104, 105, 258, 259]
        );
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

        let (grads, _, _) = micro_backward(&model, &inputs, &targets, 1.0 / 8.0, &vars)?;

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
        // Real vars for the prune (so `grads` is populated); the *empty* slice
        // goes to `grad_global_norm`, which is what this test is about.
        let (grads, _, _) = micro_backward(&model, &inputs, &targets, 1.0 / 4.0, &vm.all_vars())?;
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
            weight_decays: GroupWeightDecay::default(),
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

        train(
            &model,
            &vm,
            &mut train_loader,
            &tcfg,
            &eval,
            &DistCtx::single(),
            &dev,
        )?;

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
        let rescored = crate::eval::evaluate(&loaded, &val_batches, &token_bytes)?;
        assert_eq!(
            rescored.bpb, meta.val_bpb,
            "reloaded model must reproduce the saved bpb exactly"
        );
        Ok(())
    }
}
