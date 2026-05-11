use crate::tokenizer::BpeTokenizer;

const FIXTURES: &[(&str, &str)] = &[
    ("en", include_str!("../../tests/fixtures/eval/en.txt")),
    ("korean", include_str!("../../tests/fixtures/eval/kr.txt")),
    ("code", include_str!("../../tests/fixtures/eval/code.txt")),
];

pub struct FixtureResult {
    pub name: &'static str,
    pub bytes: usize,
    pub tokens: usize,
    pub round_trip_ok: bool,
}

pub fn eval_fixtures(tok: &BpeTokenizer) -> Vec<FixtureResult> {
    FIXTURES
        .iter()
        .map(|&(name, text)| {
            let ids = tok.encode(text);
            let decoded = tok.decode(&ids);
            FixtureResult {
                name,
                bytes: text.len(),
                tokens: ids.len(),
                round_trip_ok: decoded == text,
            }
        })
        .collect()
}

pub fn print_table(results: &[FixtureResult]) {
    println!(
        "{:<10} {:>10} {:>10} {:>14} {:>11}",
        "fixture", "bytes", "tokens", "bytes/token", "round_trip"
    );
    for r in results {
        let ratio = if r.tokens == 0 {
            0.0
        } else {
            r.bytes as f64 / r.tokens as f64
        };
        println!(
            "{:<10} {:>10} {:>10} {:>14.3} {:>11}",
            r.name,
            r.bytes,
            r.tokens,
            ratio,
            if r.round_trip_ok { "ok" } else { "FAIL" }
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::BpeTokenizerTrainer;

    #[test]
    fn eval_fixtures_returns_three_results_in_order() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        BpeTokenizerTrainer::new("data", 10_000, usize::MAX)
            .train(temp.path(), 256)
            .unwrap();
        let tok = BpeTokenizer::from_file(temp.path()).unwrap();
        let results = eval_fixtures(&tok);
        let names: Vec<&str> = results.iter().map(|r| r.name).collect();
        assert_eq!(names, vec!["en", "korean", "code"]);
    }
}
