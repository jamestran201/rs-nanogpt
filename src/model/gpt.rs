use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::GptConfig;
use super::embedding::TokenEmbedding;

/// The GPT model.
pub struct Gpt {
    wte: TokenEmbedding,
    config: GptConfig,
}

impl Gpt {
    pub fn new(cfg: GptConfig, vb: VarBuilder) -> Result<Self> {
        let wte = TokenEmbedding::new(&cfg, vb.pp("wte"))?;
        Ok(Self { wte, config: cfg })
    }

    pub fn forward(&self, idx: &Tensor) -> Result<Tensor> {
        self.wte.forward(idx)
    }

    pub fn config(&self) -> &GptConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    fn tiny_cfg() -> GptConfig {
        GptConfig {
            vocab_size: 32,
            sequence_len: 16,
            n_layer: 1,
            n_head: 2,
            n_embd: 8,
            rope_base: 100_000.0,
            norm_eps: 1e-6,
        }
    }

    fn builder(vm: &VarMap, dev: &Device) -> VarBuilder<'static> {
        VarBuilder::from_varmap(vm, DType::F32, dev)
    }

    #[test]
    fn forward_returns_b_t_n_embd() -> Result<()> {
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let n_embd = cfg.n_embd;
        let model = Gpt::new(cfg, builder(&vm, &dev))?;

        let idx = Tensor::new(&[[1u32, 2, 3], [4, 5, 6]], &dev)?; // (B=2, T=3)
        assert_eq!(model.forward(&idx)?.dims(), &[2, 3, n_embd]);
        Ok(())
    }

    #[test]
    fn registers_only_wte_weight() -> Result<()> {
        // Locks in the parameter naming: the embedding lives under `wte.weight`,
        // leaving room for `blocks.*`, `lm_head`, etc. as the model grows.
        let dev = Device::Cpu;
        let vm = VarMap::new();
        let cfg = tiny_cfg();
        let (vocab, n_embd) = (cfg.vocab_size, cfg.n_embd);
        let _model = Gpt::new(cfg, builder(&vm, &dev))?;

        let data = vm.data().lock().unwrap();
        let keys: Vec<&String> = data.keys().collect();
        assert_eq!(keys, vec!["wte.weight"]);
        assert_eq!(data["wte.weight"].dims(), &[vocab, n_embd]);
        Ok(())
    }
}
