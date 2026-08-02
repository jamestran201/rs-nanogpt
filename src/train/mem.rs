//! CUDA async-mempool telemetry for the OOM investigation (off by default).
//!
//! Every device tensor candle allocates goes through cudarc's
//! `CudaStream::alloc` → `cuMemAllocAsync`, so all tensor memory lives in the
//! device's stream-ordered **default** memory pool. That pool takes memory
//! from the driver in large granules and suballocates tensors inside them; a
//! granule can only go back to the driver when *every* suballocation in it is
//! free. `nvidia-smi` reports the granule reservation, which is why the
//! failing d24 run looked like it needed ≥76 GiB to do ~40 GB of work
//! (`writeups/pretrain-oom-investigation.md`).
//!
//! This probe reads the two numbers `nvidia-smi` cannot separate: `reserved`
//! (held from the driver) and `used` (handed out to live tensors). Their
//! difference *is* the fragmentation. A flat `used` next to a climbing
//! `reserved` confirms pool retention and rules out a candle-backend leak,
//! which would grow `used` too.
//!
//! **The probe never adds a `synchronize()` inside a step.** `cuMemFreeAsync`
//! frees are stream-ordered and the pool's release threshold is 0, so any
//! added sync trims the pool — and since the training loop's `grads` drops at
//! the end of each iteration, a sync at the top of a step would silently apply
//! the candidate fix (step B of the writeup) to the very run that is supposed
//! to measure the bug. Every in-loop sample therefore rides on a sync the loop
//! already performs.
//!
//! The high-water attributes are the other half: reset them at each step
//! boundary and the next sample reports the *true* intra-step peak, which no
//! external sampler can catch (the original 2 Hz `nvidia-smi` trace ran
//! against 0.73 s micro-batches and undersampled every peak).

use candle_core::{Device, Result};

/// Env var that enables the probe. Unset or `0` = off (the default), `1` = one
/// sample per logged step, `2` = adds a per-micro-batch sample.
pub const ENV_MEM_TRACE: &str = "RS_NANOGPT_MEM_TRACE";

/// Bytes as decimal GB — the unit the investigation writeup reports in, so
/// numbers here can be compared to it directly (note `nvidia-smi` prints MiB).
fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

/// One snapshot of the device's default async memory pool, plus the
/// device-wide free/total for context on how much room the driver has left.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// `CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT` — bytes the pool holds from the
    /// driver. This is (modulo the CUDA context) what `nvidia-smi` shows.
    pub reserved: u64,
    /// High-water `reserved` since the last [`MemProbe::reset_highs`].
    pub reserved_high: u64,
    /// `CU_MEMPOOL_ATTR_USED_MEM_CURRENT` — bytes currently handed out to live
    /// allocations. The real live footprint of the process.
    pub used: u64,
    /// High-water `used` since the last [`MemProbe::reset_highs`].
    pub used_high: u64,
    /// `cuMemGetInfo` free bytes on the device.
    pub dev_free: u64,
    /// `cuMemGetInfo` total bytes on the device.
    pub dev_total: u64,
}

impl PoolStats {
    /// `reserved − used`: pool memory held from the driver but not handed to
    /// any tensor. Unavailable to other processes and, when it is scattered
    /// across partly-occupied granules, unavailable to large requests in this
    /// one — the quantity the whole investigation turns on.
    pub fn fragmentation(&self) -> u64 {
        self.reserved.saturating_sub(self.used)
    }

    /// One greppable line, decimal GB.
    pub fn summary(&self) -> String {
        format!(
            "used {:.2} GB (peak {:.2}) | reserved {:.2} GB (peak {:.2}) | \
             frag {:.2} GB | dev_free {:.1}/{:.1} GB",
            gb(self.used),
            gb(self.used_high),
            gb(self.reserved),
            gb(self.reserved_high),
            gb(self.fragmentation()),
            gb(self.dev_free),
            gb(self.dev_total),
        )
    }
}

/// Reads pool counters for the training device. Construct with
/// [`MemProbe::from_env`]; `None` means "not tracing", so every call site is a
/// cheap `if let Some(..)` and non-CUDA builds pay nothing.
pub struct MemProbe {
    verbosity: u8,
    #[cfg(feature = "cuda")]
    pool: cuda::Pool,
}

impl MemProbe {
    /// `Some` only when [`ENV_MEM_TRACE`] asks for tracing *and* `device` is a
    /// CUDA device in a CUDA build. Anything else warns once and returns
    /// `None` rather than failing the run — this is a diagnostic knob, not a
    /// correctness feature.
    pub fn from_env(device: &Device) -> Result<Option<Self>> {
        let raw = std::env::var(ENV_MEM_TRACE).ok();
        let Some(verbosity) = parse_verbosity(raw.as_deref()) else {
            return Ok(None);
        };
        #[cfg(feature = "cuda")]
        {
            let Device::Cuda(cuda) = device else {
                eprintln!(
                    "warning: {ENV_MEM_TRACE} is set but the training device is {device:?}; \
                     memory tracing needs CUDA and stays off"
                );
                return Ok(None);
            };
            Ok(Some(Self {
                verbosity,
                pool: cuda::Pool::for_device(cuda)?,
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (device, verbosity);
            eprintln!(
                "warning: {ENV_MEM_TRACE} is set but this binary was built without \
                 --features cuda; memory tracing stays off"
            );
            Ok(None)
        }
    }

    /// Whether per-micro-batch sampling was requested (`RS_NANOGPT_MEM_TRACE=2`).
    /// Those samples show *where inside a step* the pool grows; the per-step
    /// samples alone only show that it did.
    pub fn per_micro(&self) -> bool {
        self.verbosity >= 2
    }

    /// Read the pool counters. Cheap host-side driver calls (no sync, no
    /// kernel launch), so calling this per micro-batch is free next to an
    /// 11.75 s step.
    pub fn stats(&self) -> Result<PoolStats> {
        #[cfg(feature = "cuda")]
        {
            self.pool.stats()
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!("MemProbe is only ever constructed on a CUDA device")
    }

    /// Zero both high-water marks so the next sample's `*_high` covers exactly
    /// the window since this call. Call it right after each step's sample.
    pub fn reset_highs(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.pool.reset_highs()
        }
        #[cfg(not(feature = "cuda"))]
        unreachable!("MemProbe is only ever constructed on a CUDA device")
    }

    /// Release every fully-free granule back to the driver, keeping at most
    /// `min_bytes_to_keep`.
    ///
    /// Unused by the telemetry itself — this is the lever step B of the
    /// investigation needs, and it lives here because this type owns the only
    /// pool handle in the process.
    pub fn trim(&self, min_bytes_to_keep: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            self.pool.trim(min_bytes_to_keep)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = min_bytes_to_keep;
            unreachable!("MemProbe is only ever constructed on a CUDA device")
        }
    }

    /// Sample and print to stderr, swallowing any driver error. For the
    /// failure path, where the caller's original error is what must survive.
    pub fn dump(&self, label: &str) {
        match self.stats() {
            Ok(s) => eprintln!("mem {label} | {}", s.summary()),
            Err(e) => eprintln!("mem {label} | probe failed: {e}"),
        }
    }
}

/// Split out from the env read so the parsing is testable without mutating a
/// shared process environment.
fn parse_verbosity(raw: Option<&str>) -> Option<u8> {
    match raw {
        None | Some("") | Some("0") => None,
        Some("1") => Some(1),
        Some("2") => Some(2),
        Some(other) => {
            eprintln!(
                "warning: {ENV_MEM_TRACE}={other:?} is not 0, 1 or 2; \
                 memory tracing stays off"
            );
            None
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::ffi::c_void;
    use std::sync::Arc;

    use candle_core::CudaDevice;
    use candle_core::cuda_backend::cudarc::driver::{CudaContext, result, sys};
    use candle_core::{Error, Result};

    use super::PoolStats;

    /// Handle to the device's *default* async memory pool — candle never
    /// creates a custom one, so this is the pool every tensor comes from.
    pub struct Pool {
        ctx: Arc<CudaContext>,
        pool: sys::CUmemoryPool,
    }

    impl Pool {
        pub fn for_device(device: &CudaDevice) -> Result<Self> {
            // `cuda_stream()` is the stream candle launches on; its context is
            // the one every allocation is charged against.
            let ctx = device.cuda_stream().context().clone();
            ctx.bind_to_thread().map_err(Error::debug)?;
            // SAFETY: `cu_device()` returns the device this context was
            // created for, which is exactly what the call expects.
            let pool = unsafe { result::device::get_default_mem_pool(ctx.cu_device()) }
                .map_err(Error::debug)?;
            Ok(Self { ctx, pool })
        }

        pub fn stats(&self) -> Result<PoolStats> {
            // `cuMemGetInfo` resolves against the *thread's* current context,
            // so bind before reading. One `cuCtxSetCurrent`, nanoseconds.
            self.ctx.bind_to_thread().map_err(Error::debug)?;
            let (dev_free, dev_total) = result::mem_get_info().map_err(Error::debug)?;
            Ok(PoolStats {
                reserved: self
                    .attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
                reserved_high: self
                    .attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH)?,
                used: self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
                used_high: self.attr(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH)?,
                dev_free: dev_free as u64,
                dev_total: dev_total as u64,
            })
        }

        pub fn reset_highs(&self) -> Result<()> {
            self.ctx.bind_to_thread().map_err(Error::debug)?;
            for attr in [
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            ] {
                // The driver accepts only 0 for a high-water attribute; that
                // write *is* the documented reset.
                let mut zero: u64 = 0;
                let value = std::ptr::from_mut(&mut zero).cast::<c_void>();
                // SAFETY: `self.pool` is the device's default pool, which the
                // driver owns and never destroys, and both attributes here are
                // `cuuint64_t`-typed — matching the `u64` behind `value`.
                unsafe { result::mem_pool::set_attribute(self.pool, attr, value) }
                    .map_err(Error::debug)?;
            }
            Ok(())
        }

        pub fn trim(&self, min_bytes_to_keep: usize) -> Result<()> {
            self.ctx.bind_to_thread().map_err(Error::debug)?;
            // SAFETY: as above — `self.pool` is the live default pool.
            unsafe { result::mem_pool::trim_to(self.pool, min_bytes_to_keep) }.map_err(Error::debug)
        }

        /// Read one `cuuint64_t` pool attribute.
        ///
        /// Only the four byte-count attributes go through here. The `REUSE_*`
        /// attributes are `int`-typed and would silently read garbage into the
        /// high half of the `u64` below.
        fn attr(&self, attr: sys::CUmemPool_attribute) -> Result<u64> {
            let mut out: u64 = 0;
            let value = std::ptr::from_mut(&mut out).cast::<c_void>();
            // SAFETY: `self.pool` is the device's default pool, and `value`
            // points at a writable `u64`, the documented type of every
            // attribute this function is called with (see the doc comment).
            unsafe { result::mem_pool::get_attribute(self.pool, attr, value) }
                .map_err(Error::debug)?;
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_probe_without_the_env_var() -> Result<()> {
        // The default path, and the one that matters: absent the env var the
        // probe is never constructed, so every call site in the training loop
        // is a null check. Deliberately does not set the variable — tests
        // share a process and mutating the environment would race.
        assert!(MemProbe::from_env(&Device::Cpu)?.is_none());
        Ok(())
    }

    #[test]
    fn verbosity_parses_off_step_and_micro() {
        assert_eq!(parse_verbosity(None), None);
        assert_eq!(parse_verbosity(Some("")), None);
        assert_eq!(parse_verbosity(Some("0")), None);
        assert_eq!(parse_verbosity(Some("1")), Some(1));
        assert_eq!(parse_verbosity(Some("2")), Some(2));
        // Unrecognized values warn and stay off rather than aborting a run
        // that is otherwise fine.
        assert_eq!(parse_verbosity(Some("yes")), None);
        assert_eq!(parse_verbosity(Some("3")), None);
    }

    #[test]
    fn fragmentation_is_reserved_minus_used() {
        let s = PoolStats {
            reserved: 61_440_000_000,
            used: 13_280_000_000,
            ..PoolStats::default()
        };
        assert_eq!(s.fragmentation(), 48_160_000_000);
    }

    #[test]
    fn fragmentation_saturates_instead_of_wrapping() {
        // `used > reserved` should not happen, but a torn read across two
        // driver calls must not turn into a 16-exabyte log line.
        let s = PoolStats {
            reserved: 1,
            used: 2,
            ..PoolStats::default()
        };
        assert_eq!(s.fragmentation(), 0);
    }

    #[test]
    fn summary_reports_both_halves_and_the_gap() {
        let s = PoolStats {
            reserved: 61_440_000_000,
            reserved_high: 61_500_000_000,
            used: 13_280_000_000,
            used_high: 41_900_000_000,
            dev_free: 21_100_000_000,
            dev_total: 85_520_000_000,
        };
        let line = s.summary();
        assert!(line.contains("used 13.28 GB (peak 41.90)"), "{line}");
        assert!(line.contains("reserved 61.44 GB (peak 61.50)"), "{line}");
        assert!(line.contains("frag 48.16 GB"), "{line}");
        assert!(line.contains("dev_free 21.1/85.5 GB"), "{line}");
    }
}
