mod attention;
mod block;
mod config;
mod device;
mod embedding;
mod flash_attention;
mod gpt;
mod linear;
mod loss;
mod mlp;
mod rms_norm;
mod rope;

pub use attention::CausalSelfAttention;
#[cfg(test)]
pub(crate) use attention::build_causal_mask;
pub use block::Block;
pub use config::{
    DEFAULT_N_EMBD, DEFAULT_N_HEAD, DEFAULT_N_LAYER, DEFAULT_NORM_EPS, DEFAULT_ROPE_BASE,
    DEFAULT_SEQUENCE_LEN, DEFAULT_VOCAB_SIZE, GptConfig,
};
pub use device::{compute_dtype, default_device};
pub use embedding::TokenEmbedding;
pub use gpt::Gpt;
pub use linear::Linear;
pub use loss::{Reduction, cross_entropy, cross_entropy_sum_count};
pub use mlp::Mlp;
pub use rms_norm::rms_norm;
pub use rope::Rope;
