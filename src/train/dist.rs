//! Distributed-training context (DDP over NCCL, one process per GPU).
//!
//! `DistCtx` is the seam between the training loop and multi-GPU gradient
//! synchronization: in the default single-process world every method is a
//! no-op (or the identity), so CPU/Metal builds and the whole test suite run
//! the unchanged single-GPU path. The NCCL-backed implementation lives behind
//! the `nccl` cargo feature.
//!
//! NCCL matches collectives across ranks purely **by call order** — there is
//! no name or tag in the protocol — so every rank must issue the same sequence
//! of operations over the same tensors. `VarMap::all_vars()` iterates a
//! `HashMap` whose order is randomized per process, which across spawned ranks
//! would silently average rank 0's layer-3 gradient with rank 1's layer-7
//! gradient (identical shapes ⇒ the mismatched reduce *succeeds*). Every
//! collective call site must therefore use the one canonical, name-sorted
//! parameter list from [`canonical_vars`].

use candle_core::backprop::GradStore;
use candle_core::{Result, Var};
use candle_nn::VarMap;

/// Rank/world handle threaded through the training loop. Constructed once at
/// startup; all collective methods are no-ops when `world_size == 1`.
pub struct DistCtx {
    pub rank: usize,
    pub world_size: usize,
}

impl DistCtx {
    /// The single-process context: rank 0 of a world of 1, no communicator.
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }

    /// Rank 0 owns every side effect: console output, run dir, metrics,
    /// checkpoints, sampling.
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Average every accumulated gradient across ranks (in place), making the
    /// step gradient the global per-token mean. `vars` must be the canonical
    /// list — same order on every rank. Identity when `world_size == 1`.
    pub fn all_reduce_grads(&self, _grads: &mut GradStore, _vars: &[Var]) -> Result<()> {
        // world_size == 1: the local gradient already is the global one.
        Ok(())
    }

    /// Element-wise `Sum` over ranks of a small vector of host scalars (loss
    /// and eval sums). Identity when `world_size == 1`.
    pub fn all_reduce_sums(&self, xs: &[f64]) -> Result<Vec<f64>> {
        Ok(xs.to_vec())
    }

    /// Overwrite every rank's params with rank 0's (one-time, at init), so
    /// replicas start identical without relying on cross-GPU RNG agreement.
    /// `vars` must be the canonical list. No-op when `world_size == 1`.
    pub fn broadcast_vars(&self, _vars: &[Var]) -> Result<()> {
        Ok(())
    }
}

/// The one canonical parameter list every collective call site must use:
/// `(name, Var)` pairs sorted by name, so all ranks enumerate the model's
/// parameters in the same order regardless of per-process `HashMap` layout.
pub fn canonical_vars(varmap: &VarMap) -> Vec<(String, Var)> {
    let data = varmap
        .data()
        .lock()
        .expect("VarMap lock should not be poisoned (would require another thread to panic while holding it)");
    let mut named: Vec<(String, Var)> = data
        .iter()
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::tiny_gpt;
    use candle_core::{Device, Tensor};

    /// The property NCCL's order-matched collectives rely on: two identically
    /// built models (fresh processes in real runs) enumerate their parameters
    /// as the same name sequence, and that sequence is sorted.
    #[test]
    fn canonical_vars_is_sorted_and_identical_across_builds() {
        let (vm_a, _) = tiny_gpt(32, 16);
        let (vm_b, _) = tiny_gpt(32, 16);

        let names_a: Vec<String> = canonical_vars(&vm_a).into_iter().map(|(n, _)| n).collect();
        let names_b: Vec<String> = canonical_vars(&vm_b).into_iter().map(|(n, _)| n).collect();

        assert_eq!(names_a, names_b, "var order must not depend on the process");
        assert!(names_a.is_sorted(), "canonical order must be name-sorted");
        assert_eq!(
            names_a.len(),
            vm_a.all_vars().len(),
            "canonical list must cover every var"
        );
    }

    /// Every `DistCtx::single()` method is a no-op / identity, so the
    /// single-process training path is untouched by the DDP seam.
    #[test]
    fn single_ctx_methods_are_identity() -> Result<()> {
        let dist = DistCtx::single();
        assert!(dist.is_master());
        assert_eq!(dist.world_size, 1);

        assert_eq!(dist.all_reduce_sums(&[1.5, -2.0])?, vec![1.5, -2.0]);

        // A real grad store: values must be bit-identical after the "reduce".
        let dev = Device::Cpu;
        let v = Var::new(&[1.0f32, 2.0, 3.0], &dev)?;
        let loss = v.as_tensor().sqr()?.sum_all()?;
        let mut grads = loss.backward()?;
        let before = grads
            .get(v.as_tensor())
            .expect("grad present")
            .to_vec1::<f32>()?;
        let vars = [v.clone()];
        dist.all_reduce_grads(&mut grads, &vars)?;
        let after = grads
            .get(v.as_tensor())
            .expect("grad still present")
            .to_vec1::<f32>()?;
        assert_eq!(before, after);

        dist.broadcast_vars(&vars)?;
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &dev)?;
        assert_eq!(
            v.as_tensor().to_vec1::<f32>()?,
            t.to_vec1::<f32>()?,
            "broadcast must not touch params in a world of 1"
        );
        Ok(())
    }
}
