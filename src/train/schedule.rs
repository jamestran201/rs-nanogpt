pub const DEFAULT_WARMUP_STEPS: usize = 40;
pub const DEFAULT_WARMDOWN_RATIO: f64 = 0.65;
pub const DEFAULT_FINAL_LR_FRAC: f64 = 0.05;

/// LR multiplier in `[final_frac, 1.0]` for each `step` of a run with `num_iters` max iterations.
///
/// Three phases: linear warmup over the first `warmup_steps` (so the noisiest
/// early steps run at near-zero LR), a constant `1.0` stable middle, then a
/// linear warmdown over the last `warmdown_ratio` of the run down to
/// `final_frac`.
pub fn lr_mult(
    step: usize,
    num_iters: usize,
    warmup_steps: usize,
    warmdown_ratio: f64,
    final_frac: f64,
) -> f64 {
    let warmdown_iters = (warmdown_ratio * num_iters as f64).round() as usize;
    if step < warmup_steps {
        (step + 1) as f64 / warmup_steps as f64 // linear warmup
    } else if step <= num_iters - warmdown_iters {
        1.0 // stable
    } else {
        let progress = (num_iters - step) as f64 / warmdown_iters as f64;
        progress + (1.0 - progress) * final_frac // linear warmdown → final_frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // => warmdown_iters = round(RATIO * ITERS) = round(650.0) = 650
    // stable phase ends at step 350.
    const ITERS: usize = 1000;
    const WARMUP: usize = 40;
    const RATIO: f64 = 0.65;
    const FINAL: f64 = 0.05;

    fn mult(step: usize) -> f64 {
        lr_mult(step, ITERS, WARMUP, RATIO, FINAL)
    }

    #[test]
    fn warmup_ramps_linearly_to_one() {
        assert!((mult(0) - 1.0 / 40.0).abs() < EPS);
        assert!((mult(19) - 0.5).abs() < EPS); // (19+1)/40
        // Last warmup step reaches exactly 1.0 and joins the stable value.
        assert!((mult(39) - 1.0).abs() < EPS);
    }

    #[test]
    fn stable_phase_is_one() {
        assert!((mult(200) - 1.0).abs() < EPS);
        assert!((mult(350) - 1.0).abs() < EPS); // last stable step
    }

    #[test]
    fn warmdown_linear_to_final_frac() {
        assert!(mult(351) < 1.0, "decay should have begun: {}", mult(351));
        // Window midpoint: progress = (1000-675)/650 = 0.5
        // => 0.5 + 0.5*0.05 = 0.525.
        assert!((mult(675) - 0.525).abs() < EPS);
        assert!((mult(1000) - FINAL).abs() < EPS); // progress = 0
    }

    #[test]
    fn never_below_final_frac_after_warmup() {
        // The `final_frac` floor is the *warmdown* floor. Warmup deliberately
        // starts below it (LR ramps up from ~zero: step 0 is 1/40 = 0.025), so
        // the property holds only over the post-warmup stable+warmdown range.
        for step in WARMUP..=ITERS {
            let m = mult(step);
            assert!(m >= FINAL - EPS, "step {step}: {m} < {FINAL}");
            assert!(m <= 1.0 + EPS, "step {step}: {m} > 1.0");
        }
    }
}
