pub const DEFAULT_BOOK: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pp1ppppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq c3 0 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
];

pub fn default_book() -> Vec<String> {
    DEFAULT_BOOK.iter().map(|s| s.to_string()).collect()
}

pub fn load_book_or_default(path: &str) -> Vec<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let fens: Vec<String> = content
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| l.split(" c0 ").next().unwrap_or("").trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            if fens.is_empty() {
                default_book()
            } else {
                fens
            }
        }
        Err(_) => default_book(),
    }
}

pub fn is_book_position(fen: &str) -> bool {
    DEFAULT_BOOK.contains(&fen)
}

#[derive(Debug, Clone, Default)]
pub struct BookEntry {
    pub key: String,
    pub moves: Vec<String>,
}

pub fn normalize_key(fen: &str) -> String {
    fen.split_whitespace().take(4).collect::<Vec<_>>().join(" ")
}

pub fn parse_book_line(line: &str) -> Option<BookEntry> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    if let Some(bm_pos) = line.find(" bm ") {
        let (fen_part, moves_part) = line.split_at(bm_pos);
        let moves_part = moves_part[4..].trim().trim_end_matches(';').trim();
        let moves: Vec<String> = moves_part
            .split([',', ' '])
            .map(|s| s.trim().trim_end_matches(';').to_string())
            .filter(|s| s.len() >= 4)
            .collect();
        let key = normalize_key(fen_part);
        if key.is_empty() {
            return None;
        }
        return Some(BookEntry { key, moves });
    }
    for sep in ['|', '\t'] {
        if let Some((fen_part, moves_part)) = line.split_once(sep) {
            let moves: Vec<String> = moves_part
                .split([',', ' '])
                .map(|s| s.trim().to_string())
                .filter(|s| s.len() >= 4)
                .collect();
            let key = normalize_key(fen_part);
            if key.is_empty() {
                return None;
            }
            return Some(BookEntry { key, moves });
        }
    }
    let key = normalize_key(line.split(" c0 ").next().unwrap_or(line));
    if key.is_empty() {
        return None;
    }
    Some(BookEntry {
        key,
        moves: Vec::new(),
    })
}

pub fn load_book_entries(path: &str) -> Vec<BookEntry> {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            let entries: Vec<BookEntry> = content.lines().filter_map(parse_book_line).collect();
            if entries.is_empty() {
                default_book()
                    .into_iter()
                    .map(|fen| BookEntry {
                        key: normalize_key(&fen),
                        moves: Vec::new(),
                    })
                    .collect()
            } else {
                entries
            }
        }
        Err(_) => default_book()
            .into_iter()
            .map(|fen| BookEntry {
                key: normalize_key(&fen),
                moves: Vec::new(),
            })
            .collect(),
    }
}

pub fn probe_book(entries: &[BookEntry], board_key: &str, seed: u64) -> Option<String> {
    let mut pooled: Vec<&str> = Vec::new();
    for e in entries {
        if e.key == board_key {
            pooled.extend(e.moves.iter().map(|s| s.as_str()));
        }
    }
    if pooled.is_empty() {
        return None;
    }
    Some(pooled[(seed % pooled.len() as u64) as usize].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;

    #[test]
    fn default_book_lines_parse() {
        assert!(DEFAULT_BOOK.len() >= 8);
        for fen in DEFAULT_BOOK {
            let board = BoardState::parse_fen(fen);
            let placement = board.to_fen().split(' ').next().unwrap().to_string();
            let want = fen.split(' ').next().unwrap();
            assert_eq!(placement, want, "placement mismatch for {fen}");
        }
    }

    #[test]
    fn missing_file_falls_back_to_default() {
        let book = load_book_or_default("definitely/missing/book.epd");
        assert_eq!(book.len(), DEFAULT_BOOK.len());
    }

    #[test]
    fn book_membership_check() {
        assert!(is_book_position(DEFAULT_BOOK[0]));
        assert!(!is_book_position("8/8/8/8/8/5K2/5Q2/6k1 w - - 0 1"));
    }

    #[test]
    fn normalize_ignores_clocks() {
        assert_eq!(
            normalize_key("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3"
        );
    }

    #[test]
    fn parse_epd_bm_line() {
        let e =
            parse_book_line("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 bm e2e4;")
                .unwrap();
        assert_eq!(e.moves, vec!["e2e4"]);
    }

    #[test]
    fn parse_pipe_line_with_several_moves() {
        let e = parse_book_line(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1 | e7e5,c7c5 e7e6",
        )
        .unwrap();
        assert_eq!(e.moves.len(), 3);
    }

    #[test]
    fn probe_picks_deterministically_and_ignores_bare_fens() {
        let entries = vec![
            parse_book_line("8/8/8/8/8/5K2/5Q2/6k1 w - - 0 1").unwrap(),
            parse_book_line("8/8/8/8/8/5K2/5Q2/6k1 w - - 0 1 bm f3f7;").unwrap(),
        ];
        let key = normalize_key("8/8/8/8/8/5K2/5Q2/6k1 w - - 99 99");
        assert_eq!(probe_book(&entries, &key, 0), Some("f3f7".to_string()));
        assert_eq!(probe_book(&entries, "nope", 0), None);
    }
}
