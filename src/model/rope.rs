use candle_core::{D, Device, Result, Tensor};

use super::config::GptConfig;

#[derive(Debug, Clone)]
pub struct Rope {
    /// `cos(freqs)`, shape `(seq_len, head_dim/2)`.
    cos: Tensor,
    /// `sin(freqs)`, shape `(seq_len, head_dim/2)`.
    sin: Tensor,
}

impl Rope {
    pub fn new(seq_len: usize, head_dim: usize, base: f32, device: &Device) -> Result<Self> {
        assert_eq!(head_dim % 2, 0, "head_dim must be even, got {head_dim}");

        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| base.powf(-(2.0 * i as f32) / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t: Vec<f32> = (0..seq_len).map(|p| p as f32).collect();
        let t = Tensor::from_vec(t, (seq_len, 1), device)?;

        // outer product: freqs[p, i] = t[p] * inv_freq[i], shape (seq_len, half).
        let freqs = t.broadcast_mul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub fn from_config(cfg: &GptConfig, device: &Device) -> Result<Self> {
        Self::new(cfg.sequence_len, cfg.head_dim(), cfg.rope_base, device)
    }

    /// Apply RoPE to `x` of shape `(batch_size, n_head, T, head_dim)`, using positions
    /// `0..T`. Returns the rotated tensor of the same shape.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _h, t, head_dim) = x.dims4()?;
        let half = head_dim / 2;

        // Slice the tables to the actual sequence length of (T, head_dim/2)
        // and reshape so they broadcast over the batch and head axes
        let cos = self
            .cos
            .narrow(0, 0, t)?
            .reshape((1, 1, t, half))?
            .to_dtype(x.dtype())?;
        let sin = self
            .sin
            .narrow(0, 0, t)?
            .reshape((1, 1, t, half))?
            .to_dtype(x.dtype())?;

        let x1 = x.narrow(D::Minus1, 0, half)?; // (batch_size, n_head, T, head_dim/2)
        let x2 = x.narrow(D::Minus1, half, half)?; // (batch_size, n_head, T, head_dim/2)
        let y1 = (x1.broadcast_mul(&cos)? + x2.broadcast_mul(&sin)?)?;
        let y2 = (x2.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        Tensor::cat(&[y1, y2], D::Minus1) // (batch_size, n_head, T, head_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: f32 = 100_000.0;

    /// Rotate a single vector placed at every position: returns rows where
    /// row `p` is `rope(v, p)`. Shape in: `(head_dim,)`, out: `(seq, head_dim)`.
    fn rope_at_all_positions(rope: &Rope, v: &[f32], seq: usize, dev: &Device) -> Vec<Vec<f32>> {
        let head_dim = v.len();
        // Broadcast v across all T positions: (1, 1, seq, head_dim).
        let row = Tensor::from_vec(v.to_vec(), (1, 1, 1, head_dim), dev).unwrap();
        let x = row.broadcast_as((1, 1, seq, head_dim)).unwrap();
        let y = rope.apply(&x).unwrap();
        y.reshape((seq, head_dim))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn shape_round_trip() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, t, hd) = (2usize, 3, 5, 8);
        let rope = Rope::new(16, hd, BASE, &dev)?;
        let x = Tensor::randn(0.0f32, 1.0, (b, h, t, hd), &dev)?;
        assert_eq!(rope.apply(&x)?.dims(), &[b, h, t, hd]);
        Ok(())
    }

    #[test]
    fn position_zero_is_identity() -> Result<()> {
        // Position 0 → angle 0 → cos=1, sin=0 → rope(x, 0) == x.
        let dev = Device::Cpu;
        let hd = 8;
        let rope = Rope::new(4, hd, BASE, &dev)?;
        let v: Vec<f32> = (0..hd).map(|i| i as f32 - 3.5).collect();
        let rotated = rope_at_all_positions(&rope, &v, 4, &dev);
        for (a, b) in v.iter().zip(&rotated[0]) {
            assert!((a - b).abs() < 1e-5, "pos 0 changed: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn rotation_preserves_norm() -> Result<()> {
        // A rotation is orthogonal, so each rotated vector keeps its length.
        let dev = Device::Cpu;
        let hd = 8;
        let rope = Rope::new(6, hd, BASE, &dev)?;
        let v: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.7).sin()).collect();
        let norm0 = dot(&v, &v).sqrt();
        for row in rope_at_all_positions(&rope, &v, 6, &dev) {
            assert!((dot(&row, &row).sqrt() - norm0).abs() < 1e-4);
        }
        Ok(())
    }

    #[test]
    fn dot_product_depends_only_on_relative_offset() -> Result<()> {
        // The defining RoPE property: rope(q, m)·rope(k, n) is unchanged when m
        // and n are shifted together — it depends only on (m − n).
        let dev = Device::Cpu;
        let hd = 8;
        let seq = 12;
        let rope = Rope::new(seq, hd, BASE, &dev)?;
        let q: Vec<f32> = (0..hd).map(|i| (i as f32 * 1.3).cos()).collect();
        let k: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.6 + 1.0).sin()).collect();
        let rq = rope_at_all_positions(&rope, &q, seq, &dev);
        let rk = rope_at_all_positions(&rope, &k, seq, &dev);

        // Offset = +3, sampled at three different absolute base positions.
        let offset = 3;
        let reference = dot(&rq[offset], &rk[0]);
        for base in [1usize, 4, 7] {
            let shifted = dot(&rq[base + offset], &rk[base]);
            assert!(
                (shifted - reference).abs() < 1e-3,
                "offset-{offset} dot changed with absolute position: {reference} vs {shifted}"
            );
        }
        Ok(())
    }
}
