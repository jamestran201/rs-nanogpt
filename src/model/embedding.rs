use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder};

use super::config::GptConfig;

/// Word token embedding (`wte`): the learned lookup table that maps token ids
/// to vectors entering the residual stream — `x = wte[idx]` in the forward
/// pass. Shape `(vocab_size, n_embd)`, one row per token.
pub struct TokenEmbedding {
    inner: Embedding,
}

impl TokenEmbedding {
    /// Register the embedding weight under `vb` and initialise it with
    /// `Normal(0, 0.8)`. The weight is created as a trainable variable in
    /// the builder's `VarMap`, ready for the optimizer and checkpointing.
    pub fn new(cfg: &GptConfig, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(
            (cfg.vocab_size, cfg.n_embd),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 0.8,
            },
        )?;
        Ok(Self {
            inner: Embedding::new(weight, cfg.n_embd),
        })
    }

    /// Look up `idx` (token ids, shape `(B, T)`, dtype U32) and return the
    /// corresponding embeddings of shape `(B, T, n_embd)`.
    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        self.inner.forward(idx)
    }

    pub fn weight(&self) -> &Tensor {
        self.inner.embeddings()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 10,
            sequence_len: 8,
            n_layer: 1,
            n_head: 2,
            n_embd: 4,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    fn builder(vm: &VarMap, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_varmap(vm, DType::F32, dev)
    }

    #[test]
    fn forward_returns_b_t_c() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let emb = TokenEmbedding::new(&cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?; // (B=2, T=3)
        let out = emb.forward(&idx)?;
        assert_eq!(out.dims(), &[2, 3, cfg.n_embd]);
        Ok(())
    }

    #[test]
    fn weight_shape_and_init_std() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        // Large enough for a meaningful empirical std estimate.
        let cfg = GptConfig {
            vocab_size: 4096,
            n_embd: 64,
            ..tiny_cfg()
        };
        let emb = TokenEmbedding::new(&cfg, builder(&vm, &dev))?;
        assert_eq!(emb.weight().dims(), &[cfg.vocab_size, cfg.n_embd]);

        let w = emb.weight();
        let mean = w.mean_all()?.to_scalar::<f32>()?;
        let std = (w.sqr()?.mean_all()?.to_scalar::<f32>()? - mean * mean).sqrt();
        assert!(
            (std - 0.8).abs() < 0.1,
            "empirical std was {std}, expected ~0.8"
        );
        Ok(())
    }

    #[test]
    fn lookup_matches_known_weight() -> Result<()> {
        let dev = Device::Cpu;
        let mut vm = VarMap::new();
        let cfg = tiny_cfg(); // vocab 10, n_embd 4
        let emb = TokenEmbedding::new(&cfg, builder(&vm, &dev))?;

        // Overwrite the random init with a known matrix: row i = [i, i, i, i].
        let rows: Vec<f32> = (0..cfg.vocab_size)
            .flat_map(|i| std::iter::repeat_n(i as f32, cfg.n_embd))
            .collect();
        let known = Tensor::from_vec(rows, (cfg.vocab_size, cfg.n_embd), &dev)?;
        vm.set_one("weight", &known)?;

        let idx = Tensor::new(&[[3u32, 7], [3, 0]], &dev)?;
        let out = emb.forward(&idx)?.to_vec3::<f32>()?; // (2, 2, 4)

        assert_eq!(out[0][0], vec![3.0, 3.0, 3.0, 3.0]);
        assert_eq!(out[0][1], vec![7.0, 7.0, 7.0, 7.0]);
        assert_eq!(out[1][1], vec![0.0, 0.0, 0.0, 0.0]);
        // Same id -> identical vector.
        assert_eq!(out[0][0], out[1][0]);
        Ok(())
    }
}
