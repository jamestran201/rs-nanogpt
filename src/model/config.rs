use candle_core::{Result, bail};

#[derive(Debug, Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    /// Maximum context length (tokens per training example).
    pub sequence_len: usize,
    /// Number of transformer blocks.
    pub n_layer: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Residual-stream width. Must be divisible by `n_head`.
    pub n_embd: usize,
    /// RoPE frequency base (larger base = longer effective context).
    pub rope_base: f32,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
}

impl GptConfig {
    /// The Mac smoke-test configuration:
    /// depth 6, head_dim 64 (→ n_embd 384), seq 512, vocab 32768.
    pub fn mac_smoke() -> Self {
        Self {
            vocab_size: 32768,
            sequence_len: 512,
            n_layer: 6,
            n_head: 6,
            n_embd: 384,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_head == 0 {
            bail!("n_head must be non-zero");
        }
        if !self.n_embd.is_multiple_of(self.n_head) {
            bail!(
                "n_embd ({}) must be divisible by n_head ({}) so head_dim is an integer",
                self.n_embd,
                self.n_head
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mac_smoke_is_valid_and_head_dim_64() {
        let cfg = GptConfig::mac_smoke();
        assert_eq!(cfg.head_dim(), 64);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn rejects_n_embd_not_divisible_by_n_head() {
        let mut cfg = GptConfig::mac_smoke();
        cfg.n_embd = 100; // not divisible by n_head = 6
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_heads() {
        let mut cfg = GptConfig::mac_smoke();
        cfg.n_head = 0;
        assert!(cfg.validate().is_err());
    }
}
