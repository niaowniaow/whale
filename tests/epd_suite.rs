use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use whale::board::state::BoardState;
use whale::search::iterative_deepening;
use whale::search::metrics;
use whale::search::position_state::PositionState;
use whale::search::search_state::SearchState;

fn parse_epd_fens(file_path: &str) -> Vec<String> {
    let p = Path::new(file_path);
    if !p.exists() {
        return Vec::new();
    }
    let content = fs::read_to_string(p).unwrap_or_default();
    content
        .lines()
        .filter_map(|l| {
            let l = l.trim();
            if l.is_empty() || l.starts_with('#') {
                None
            } else {
                let parts: Vec<&str> = l.split(" c0 ").collect();
                Some(parts[0].trim_end_matches(';').trim().to_string())
            }
        })
        .collect()
}

fn search_fen(fen: &str, depth: u8) -> metrics::SearchMetrics {
    let mut board = BoardState::parse_fen(fen);
    let mut state = SearchState::new();
    let token = AtomicBool::new(false);
    let mut debug = false;
    iterative_deepening::search(&mut board, depth, &token, &mut debug, &mut state, 1);
    metrics::collect(&state)
}

#[test]
fn test_epd_defense_benchmarks() {
    let fens = parse_epd_fens("data/benchmarks/defense.epd");
    assert!(!fens.is_empty(), "defense.epd should exist");
    for fen in fens {
        let m = search_fen(&fen, 2);
        assert_eq!(
            m.state,
            PositionState::Defend,
            "FEN {} must classify as Defend, got {:?}",
            fen,
            m.state
        );
    }
}

#[test]
fn test_epd_quiet_benchmarks() {
    let fens = parse_epd_fens("data/benchmarks/quiet.epd");
    assert!(!fens.is_empty(), "quiet.epd should exist");
    for fen in fens {
        let m = search_fen(&fen, 2);
        assert!(
            !matches!(m.state, PositionState::Attack | PositionState::Crush),
            "Quiet FEN {} should not attack, got {:?}",
            fen,
            m.state
        );
    }
}

#[test]
fn test_epd_conversion_benchmarks() {
    let fens = parse_epd_fens("data/benchmarks/conversion.epd");
    assert!(!fens.is_empty(), "conversion.epd should exist");
    for fen in fens {
        let m = search_fen(&fen, 2);
        assert!(
            m.simplify || matches!(m.state, PositionState::Convert | PositionState::Crush),
            "Conversion FEN {} must trigger simplify/convert, score={}",
            fen,
            m.score
        );
    }
}

#[test]
fn test_epd_must_try_benchmarks() {
    let fens = parse_epd_fens("data/benchmarks/must_try.epd");
    assert!(!fens.is_empty(), "must_try.epd should exist");
    for fen in fens {
        let m = search_fen(&fen, 1);
        assert!(
            m.musttry_fired || m.verified.is_some() || m.score > 100,
            "MustTry FEN {} should trigger must-try or high score",
            fen
        );
    }
}
