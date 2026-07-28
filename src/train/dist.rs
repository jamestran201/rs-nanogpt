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
//!
//! Rendezvous: the parent process generates one NCCL unique id and passes it
//! (hex-encoded) to each spawned rank through the `RS_NANOGPT_*` env vars,
//! along with the rank and world size.

use candle_core::backprop::GradStore;
use candle_core::{Result, Var};
use candle_nn::VarMap;

/// Env var carrying the spawned process's rank (its presence marks a child).
pub const ENV_RANK: &str = "RS_NANOGPT_RANK";
/// Env var carrying the total number of ranks.
pub const ENV_WORLD_SIZE: &str = "RS_NANOGPT_WORLD_SIZE";
/// Env var carrying the hex-encoded 128-byte NCCL unique id.
pub const ENV_NCCL_ID: &str = "RS_NANOGPT_NCCL_ID";

/// Byte length of an NCCL unique id (`ncclUniqueId`).
pub const NCCL_ID_LEN: usize = 128;

/// Rank/world handle threaded through the training loop. Constructed once at
/// startup; all collective methods are no-ops when `world_size == 1`.
pub struct DistCtx {
    rank: usize,
    world_size: usize,
    #[cfg(feature = "nccl")]
    nccl: Option<nccl::NcclComm>,
}

impl DistCtx {
    /// The single-process context: rank 0 of a world of 1, no communicator.
    pub fn single() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            #[cfg(feature = "nccl")]
            nccl: None,
        }
    }

    /// This process's rank in `0..world_size`.
    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Rank 0 owns every side effect: console output, run dir, metrics,
    /// checkpoints, sampling.
    pub fn is_master(&self) -> bool {
        self.rank == 0
    }

    /// Average every accumulated gradient across ranks (in place), making the
    /// step gradient the global per-token mean. `vars` must be the canonical
    /// list — same order on every rank. Identity when `world_size == 1`.
    pub fn all_reduce_grads(&self, grads: &mut GradStore, vars: &[Var]) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }
        #[cfg(feature = "nccl")]
        return self.comm()?.all_reduce_grads(grads, vars);
        #[cfg(not(feature = "nccl"))]
        {
            let _ = (grads, vars);
            unreachable!("world_size > 1 cannot be constructed without the nccl feature")
        }
    }

    /// Element-wise `Sum` over ranks of a small vector of host scalars (loss
    /// and eval sums). Identity when `world_size == 1`.
    pub fn all_reduce_sums(&self, xs: &[f64]) -> Result<Vec<f64>> {
        if self.world_size == 1 {
            return Ok(xs.to_vec());
        }
        #[cfg(feature = "nccl")]
        return self.comm()?.all_reduce_sums(xs);
        #[cfg(not(feature = "nccl"))]
        unreachable!("world_size > 1 cannot be constructed without the nccl feature")
    }

    /// Overwrite every rank's params with rank 0's (one-time, at init), so
    /// replicas start identical without relying on cross-GPU RNG agreement.
    /// `vars` must be the canonical list. No-op when `world_size == 1`.
    pub fn broadcast_vars(&self, vars: &[Var]) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }
        #[cfg(feature = "nccl")]
        return self.comm()?.broadcast_vars(vars);
        #[cfg(not(feature = "nccl"))]
        {
            let _ = vars;
            unreachable!("world_size > 1 cannot be constructed without the nccl feature")
        }
    }

    #[cfg(feature = "nccl")]
    fn comm(&self) -> Result<&nccl::NcclComm> {
        self.nccl.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("DistCtx has world_size > 1 but no NCCL communicator".into())
        })
    }
}

#[cfg(feature = "nccl")]
impl DistCtx {
    /// The multi-process context for one spawned rank: binds this process to
    /// the NCCL world described by its launcher-provided env. `device` must be
    /// the rank's own CUDA device — the communicator is created on candle's
    /// compute stream, so collectives are ordered after the backward that
    /// produced the gradients and before the optimizer step that reads them.
    pub fn new_nccl(device: &candle_core::Device, env: &ChildEnv) -> Result<Self> {
        let comm = nccl::NcclComm::new(device, env.rank, env.world_size, &env.id_hex)?;
        Ok(Self {
            rank: env.rank,
            world_size: env.world_size,
            nccl: Some(comm),
        })
    }
}

/// Generate a fresh NCCL unique id, hex-encoded for [`ENV_NCCL_ID`]. Called
/// once by the parent launcher before spawning the ranks.
#[cfg(feature = "nccl")]
pub fn generate_nccl_id_hex() -> Result<String> {
    nccl::generate_id_hex()
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

/// The launcher-provided identity of a spawned rank, parsed from the
/// `RS_NANOGPT_*` env vars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildEnv {
    pub rank: usize,
    pub world_size: usize,
    pub id_hex: String,
}

/// Read the child identity from the process environment: `Ok(None)` for a
/// plain (parent / single-process) invocation, `Ok(Some(_))` for a spawned
/// rank, `Err` for a half-set or invalid environment.
pub fn child_env() -> std::result::Result<Option<ChildEnv>, String> {
    parse_child_env(
        std::env::var(ENV_RANK).ok().as_deref(),
        std::env::var(ENV_WORLD_SIZE).ok().as_deref(),
        std::env::var(ENV_NCCL_ID).ok().as_deref(),
    )
}

fn parse_child_env(
    rank: Option<&str>,
    world_size: Option<&str>,
    id_hex: Option<&str>,
) -> std::result::Result<Option<ChildEnv>, String> {
    let (rank, world_size, id_hex) = match (rank, world_size, id_hex) {
        (None, None, None) => return Ok(None),
        (Some(r), Some(w), Some(id)) => (r, w, id),
        _ => {
            return Err(format!(
                "{ENV_RANK}, {ENV_WORLD_SIZE} and {ENV_NCCL_ID} must be set together \
                 (they are launcher-internal; unset them for a normal run)"
            ));
        }
    };
    let rank: usize = rank
        .parse()
        .map_err(|_| format!("{ENV_RANK} must be a non-negative integer, got {rank:?}"))?;
    let world_size: usize = world_size
        .parse()
        .map_err(|_| format!("{ENV_WORLD_SIZE} must be a positive integer, got {world_size:?}"))?;
    if world_size == 0 || rank >= world_size {
        return Err(format!(
            "rank {rank} out of range for world size {world_size}"
        ));
    }
    decode_id_hex(id_hex)?; // validate now, before any CUDA/NCCL work
    Ok(Some(ChildEnv {
        rank,
        world_size,
        id_hex: id_hex.to_string(),
    }))
}

/// Hex-encode an NCCL unique id's bytes for env-var transport.
pub fn encode_id_hex(bytes: &[u8; NCCL_ID_LEN]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Decode [`encode_id_hex`]'s output back into the id bytes.
pub fn decode_id_hex(hex: &str) -> std::result::Result<[u8; NCCL_ID_LEN], String> {
    if hex.len() != 2 * NCCL_ID_LEN {
        return Err(format!(
            "NCCL id hex must be {} chars, got {}",
            2 * NCCL_ID_LEN,
            hex.len()
        ));
    }
    let mut bytes = [0u8; NCCL_ID_LEN];
    for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
        let pair =
            std::str::from_utf8(chunk).map_err(|_| "NCCL id hex is not ASCII".to_string())?;
        bytes[i] = u8::from_str_radix(pair, 16)
            .map_err(|_| format!("NCCL id hex has a non-hex pair {pair:?} at byte {i}"))?;
    }
    Ok(bytes)
}

/// NCCL collectives, implemented as `CustomOp1`s in the style of candle's
/// `llama_multiprocess` example: unwrap the input `CudaStorage`, run the
/// collective into a freshly allocated buffer on the same stream, wrap it
/// back. Applied with `apply_op1_no_bwd` strictly *outside* any autograd
/// graph (gradient sync runs after `backward()`), so the known
/// "`_no_bwd` silently cuts gradients" footgun does not apply.
///
/// Unlike the example, the ops carry no `unsafe impl Send/Sync`: they are
/// constructed at the call site on the one thread that owns the `Rc<Comm>`,
/// and `apply_op1_no_bwd` requires neither bound.
#[cfg(feature = "nccl")]
mod nccl {
    use std::rc::Rc;

    use candle_core::backprop::GradStore;
    use candle_core::cuda_backend::cudarc::nccl::safe::{Comm, Id, ReduceOp};
    use candle_core::{
        CpuStorage, CudaStorage, CustomOp1, DType, Device, Layout, Result, Shape, Tensor, Var,
    };

    pub fn generate_id_hex() -> Result<String> {
        let id = Id::new().map_err(candle_core::Error::debug)?;
        let mut bytes = [0u8; super::NCCL_ID_LEN];
        for (dst, src) in bytes.iter_mut().zip(id.internal()) {
            *dst = *src as u8;
        }
        Ok(super::encode_id_hex(&bytes))
    }

    pub(super) struct NcclComm {
        comm: Rc<Comm>,
        device: Device,
    }

    impl NcclComm {
        pub(super) fn new(
            device: &Device,
            rank: usize,
            world_size: usize,
            id_hex: &str,
        ) -> Result<Self> {
            let Device::Cuda(cuda) = device else {
                candle_core::bail!("NCCL requires a CUDA device, got {device:?}")
            };
            let bytes = super::decode_id_hex(id_hex).map_err(candle_core::Error::Msg)?;
            // c_char is i8 on x86-64 Linux but u8 on aarch64 (GH200): cast
            // element-wise instead of assuming either representation.
            let mut internal = [0 as core::ffi::c_char; super::NCCL_ID_LEN];
            for (dst, src) in internal.iter_mut().zip(bytes) {
                *dst = src as core::ffi::c_char;
            }
            // `cuda_stream()` is the stream candle launches kernels on, so
            // collectives enqueue in program order w.r.t. compute.
            let comm = Comm::from_rank(cuda.cuda_stream(), rank, world_size, Id::uninit(internal))
                .map_err(candle_core::Error::debug)?;
            Ok(Self {
                comm: Rc::new(comm),
                device: device.clone(),
            })
        }

        pub(super) fn all_reduce_grads(&self, grads: &mut GradStore, vars: &[Var]) -> Result<()> {
            let op = AllReduce {
                comm: self.comm.clone(),
                op: ReduceOp::Avg,
            };
            for v in vars {
                let t = v.as_tensor();
                // A var with no local grad (the zero-init residual projections
                // at the strict-init step) still joins the collective — as
                // zeros, which is also its correct contribution to the
                // average — so every rank issues the same reduce sequence
                // even if grad presence were ever to differ across ranks.
                let g = match grads.get(t) {
                    Some(g) => g.contiguous()?,
                    None => t.zeros_like()?,
                };
                let reduced = g.apply_op1_no_bwd(&op)?;
                grads.insert(t, reduced);
            }
            Ok(())
        }

        pub(super) fn all_reduce_sums(&self, xs: &[f64]) -> Result<Vec<f64>> {
            let t = Tensor::from_vec(xs.to_vec(), xs.len(), &self.device)?;
            let op = AllReduce {
                comm: self.comm.clone(),
                op: ReduceOp::Sum,
            };
            t.apply_op1_no_bwd(&op)?.to_vec1::<f64>()
        }

        pub(super) fn broadcast_vars(&self, vars: &[Var]) -> Result<()> {
            let op = Broadcast {
                comm: self.comm.clone(),
            };
            for v in vars {
                let received = v.as_tensor().contiguous()?.apply_op1_no_bwd(&op)?;
                v.set(&received)?;
            }
            Ok(())
        }
    }

    /// All-reduce with `self.op` across the world: f32 gradients, f64 scalar
    /// sums.
    struct AllReduce {
        comm: Rc<Comm>,
        op: ReduceOp,
    }

    impl CustomOp1 for AllReduce {
        fn name(&self) -> &'static str {
            "nccl-all-reduce"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("nccl all-reduce is CUDA-only")
        }

        fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = s.device().clone();
            let elem_count = l.shape().elem_count();
            let Some((start, end)) = l.contiguous_offsets() else {
                candle_core::bail!("nccl all-reduce input must be contiguous");
            };
            let dst = match s.dtype() {
                DType::F32 => {
                    let src = s.as_cuda_slice::<f32>()?.slice(start..end);
                    let mut dst = unsafe { dev.alloc::<f32>(elem_count) }?;
                    self.comm
                        .all_reduce(&src, &mut dst, &self.op)
                        .map_err(candle_core::Error::debug)?;
                    CudaStorage::wrap_cuda_slice(dst, dev)
                }
                DType::F64 => {
                    let src = s.as_cuda_slice::<f64>()?.slice(start..end);
                    let mut dst = unsafe { dev.alloc::<f64>(elem_count) }?;
                    self.comm
                        .all_reduce(&src, &mut dst, &self.op)
                        .map_err(candle_core::Error::debug)?;
                    CudaStorage::wrap_cuda_slice(dst, dev)
                }
                dtype => candle_core::bail!("nccl all-reduce: unsupported dtype {dtype:?}"),
            };
            Ok((dst, l.shape().clone()))
        }
    }

    /// Broadcast from rank 0 (params-at-init); non-root send buffers are
    /// ignored by NCCL, so every rank passes its own storage.
    struct Broadcast {
        comm: Rc<Comm>,
    }

    impl CustomOp1 for Broadcast {
        fn name(&self) -> &'static str {
            "nccl-broadcast"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("nccl broadcast is CUDA-only")
        }

        fn cuda_fwd(&self, s: &CudaStorage, l: &Layout) -> Result<(CudaStorage, Shape)> {
            let dev = s.device().clone();
            let elem_count = l.shape().elem_count();
            let Some((start, end)) = l.contiguous_offsets() else {
                candle_core::bail!("nccl broadcast input must be contiguous");
            };
            let dst = match s.dtype() {
                DType::F32 => {
                    let src = s.as_cuda_slice::<f32>()?.slice(start..end);
                    let mut dst = unsafe { dev.alloc::<f32>(elem_count) }?;
                    self.comm
                        .broadcast(Some(&src), &mut dst, 0)
                        .map_err(candle_core::Error::debug)?;
                    CudaStorage::wrap_cuda_slice(dst, dev)
                }
                dtype => candle_core::bail!("nccl broadcast: unsupported dtype {dtype:?}"),
            };
            Ok((dst, l.shape().clone()))
        }
    }
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
        assert_eq!(dist.world_size(), 1);

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

    #[test]
    fn id_hex_round_trips() {
        let mut bytes = [0u8; NCCL_ID_LEN];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(37).wrapping_add(11);
        }
        let hex = encode_id_hex(&bytes);
        assert_eq!(hex.len(), 2 * NCCL_ID_LEN);
        assert_eq!(decode_id_hex(&hex).unwrap(), bytes);
    }

    #[test]
    fn id_hex_decode_rejects_malformed_input() {
        assert!(decode_id_hex("").is_err());
        assert!(decode_id_hex(&"ab".repeat(NCCL_ID_LEN - 1)).is_err()); // too short
        assert!(decode_id_hex(&"zz".repeat(NCCL_ID_LEN)).is_err()); // non-hex
    }

    #[test]
    fn child_env_absent_means_single_process() {
        assert_eq!(parse_child_env(None, None, None), Ok(None));
    }

    #[test]
    fn child_env_parses_a_complete_valid_set() {
        let id = encode_id_hex(&[7u8; NCCL_ID_LEN]);
        let env = parse_child_env(Some("3"), Some("8"), Some(&id))
            .unwrap()
            .unwrap();
        assert_eq!(
            env,
            ChildEnv {
                rank: 3,
                world_size: 8,
                id_hex: id,
            }
        );
    }

    #[test]
    fn child_env_rejects_partial_or_invalid_sets() {
        let id = encode_id_hex(&[7u8; NCCL_ID_LEN]);
        // Half-set environment.
        assert!(parse_child_env(Some("0"), None, None).is_err());
        assert!(parse_child_env(None, Some("8"), Some(&id)).is_err());
        // Non-numeric / out-of-range values.
        assert!(parse_child_env(Some("x"), Some("8"), Some(&id)).is_err());
        assert!(parse_child_env(Some("0"), Some("0"), Some(&id)).is_err());
        assert!(parse_child_env(Some("8"), Some("8"), Some(&id)).is_err());
        // Malformed id.
        assert!(parse_child_env(Some("0"), Some("8"), Some("nothex")).is_err());
    }
}
