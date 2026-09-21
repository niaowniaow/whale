use std::sync::atomic::AtomicBool;
use whale::board::state::BoardState;
use whale::common::helpers::STARTING_FEN;
use whale::search::iterative_deepening;
use whale::search::search_state::SearchState;

fn run_search(fen: &str, depth: u8, multipv: usize) -> (BoardState, SearchState) {
    let mut board = BoardState::parse_fen(fen);
    let mut state = SearchState::new();
    state.multipv = multipv;
    let token = AtomicBool::new(false);
    let mut debug = true;
    iterative_deepening::search(&mut board, depth, &token, &mut debug, &mut state, 1);
    (board, state)
}

#[test]
fn multipv_two_returns_two_distinct_legal_lines() {
    let (board, state) = run_search(STARTING_FEN, 3, 2);
    assert_eq!(state.multipv_lines.len(), 2);
    assert_eq!(state.multipv_lines[0].mv, state.best_move);
    assert_ne!(state.multipv_lines[0].mv, state.multipv_lines[1].mv);
    for line in &state.multipv_lines {
        assert!(
            board.is_legal(line.mv),
            "illegal multipv move {:?}",
            line.mv
        );
        assert!(!line.pv.is_empty());
    }
}

#[test]
fn multipv_one_default_records_single_primary_line() {
    let (board, state) = run_search(STARTING_FEN, 3, 1);
    assert_eq!(state.multipv, 1);
    assert_eq!(state.multipv_lines.len(), 1);
    assert_eq!(state.multipv_lines[0].mv, state.best_move);
    assert!(board.is_legal(state.best_move));
}

#[test]
fn root_diagnostics_populated() {
    let (_board, state) = run_search(STARTING_FEN, 3, 1);
    assert!(state.prev_root_score.is_some());
    assert!(state.prev_opp_cpi.is_some());
    assert!(state.prev_opp_freedom.is_some());

    assert_eq!(state.prev_opp_freedom.unwrap(), 20);
}

#[test]
fn all_aprm_off_still_finds_legal_move() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let mut state = SearchState::new();
    state.params.cpi_enabled = false;
    state.params.state_enabled = false;
    state.params.risk_enabled = false;
    state.params.pressure_enabled = false;
    state.params.attack_enabled = false;
    state.params.conversion_enabled = false;
    let token = AtomicBool::new(false);
    let mut debug = true;
    iterative_deepening::search(&mut board, 3, &token, &mut debug, &mut state, 1);
    assert!(board.is_legal(state.best_move));
    assert_eq!(state.multipv_lines.len(), 1);
}

#[test]
fn mate_position_arms_verification_and_simplify() {
    let fen = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4";
    let (board, state) = run_search(fen, 1, 1);
    let best_lan = {
        let m = state.best_move;
        let promo = m
            .promotion_char()
            .map(|c| c.to_string())
            .unwrap_or_default();
        format!("{}{}{}", m.source, m.target, promo)
    };
    assert_eq!(best_lan, "f3f7");
    assert!(board.is_legal(state.best_move));
    assert!(
        state.simplify_bias,
        "mate score must enable simplification bias"
    );
    assert!(
        state.verification_budget > 0,
        "mate-score jump must fund verification budget"
    );
    assert!(
        state.last_musttry,
        "mate-score jump inside the window must fire must-try"
    );
    assert!(
        state.last_verified.is_some(),
        "fired must-try must run a verification search"
    );
}

#[test]
fn cpi_toggle_keeps_search_legal() {
    for cpi in [true, false] {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let mut state = SearchState::new();
        state.params.cpi_enabled = cpi;
        let token = AtomicBool::new(false);
        let mut debug = false;
        iterative_deepening::search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert!(board.is_legal(state.best_move), "cpi={cpi}");
    }
}
