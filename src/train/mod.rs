pub mod dist;
pub mod mem;
mod optim;
mod schedule;
mod train_loop;

pub use dist::{DistCtx, canonical_vars};
pub use mem::{ENV_MEM_TRACE, MemProbe, PoolStats};
pub use optim::{GroupLrs, GroupedAdamW};
pub use schedule::{DEFAULT_FINAL_LR_FRAC, DEFAULT_WARMDOWN_RATIO, DEFAULT_WARMUP_STEPS, lr_mult};
pub use train_loop::{EvalContext, TrainConfig, train};
