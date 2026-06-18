use fancy_regex::Regex;

pub type TokenId = u32;

pub(crate) const SINGLE_BYTE_TABLE: [[u8; 1]; 256] = {
    let mut table = [[0u8; 1]; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = [i as u8];
        i += 1;
    }
    table
};

pub(crate) const REGEX_PATTERNS: &[&str] = &[
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    r"\s*[\r\n]+",
    r"\s+(?!\S)",
    r"\s+",
];

pub(crate) fn build_pattern() -> Regex {
    Regex::new(&REGEX_PATTERNS.join("|")).expect("Built-in regex pattern should be valid")
}

pub(crate) fn pre_tokenize<'a>(pattern: &Regex, text: &'a str) -> Vec<&'a str> {
    let mut pieces = Vec::new();
    let mut start = 0;
    while let Some(m) = pattern
        .find_from_pos(text, start)
        .expect("Unexpected regex error in pre_tokenize")
    {
        pieces.push(&text[m.start()..m.end()]);
        start = m.end();
    }
    pieces
}

pub(crate) struct Vocab {
    pub(crate) merged: Vec<Vec<u8>>,
}

impl Vocab {
    /// Token id space is 0-based, matching tiktoken's wire format:
    /// 0..=255 → single bytes, 256.. → learned merges.
    pub(crate) fn bytes_of(&self, id: TokenId) -> &[u8] {
        if (id as usize) < 256 {
            &SINGLE_BYTE_TABLE[id as usize]
        } else {
            &self.merged[(id as usize) - 256]
        }
    }

    pub(crate) fn push_merge(&mut self, bytes: Vec<u8>) -> TokenId {
        let id = 256 + self.merged.len() as TokenId;
        self.merged.push(bytes);
        id
    }
}
