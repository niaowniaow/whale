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
}
