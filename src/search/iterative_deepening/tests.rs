use super::*;
use crate::common::side::Side;

#[test]
fn test_format_score() {
    assert_eq!(format_score(100), "cp 100");
    assert_eq!(format_score(-500), "cp -500");
    assert_eq!(format_score(MAX_CENTIPAWN_EVAL - 1), "mate 1");
    assert_eq!(format_score(MAX_CENTIPAWN_EVAL - 3), "mate 2");
}

#[test]
fn search_records_engine_side_from_root_stm() {
    for (fen, expected) in [
        (crate::common::helpers::STARTING_FEN, Side::White),
        (
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            Side::Black,
        ),
    ] {
        let mut board = BoardState::parse_fen(fen);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert_eq!(state.engine_side, expected, "fen {fen}");
    }
}

#[test]
fn contempt_is_engine_relative_at_the_root() {
    for fen in ["8/8/8/8/8/8/8/K6k w - - 0 1", "8/8/8/8/8/8/8/K6k b - - 0 1"] {
        let mut board = BoardState::parse_fen(fen);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.contempt_cp = 50;
        search(&mut board, 3, &token, &mut debug, &mut state, 1);
        assert!(
            (-51..=-50).contains(&state.score),
            "fen {fen}: contempt must devalue the draw for the engine, got {}",
            state.score
        );
    }
}

#[test]
fn test_smp_search_multi_threaded() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    search(&mut board, 4, &token, &mut debug, &mut state, 4);
    assert_ne!(state.best_move, Move::NO_MOVE);
}

#[test]
fn all_experimentals_off_still_finds_legal_move() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.params.extension_cap_enabled = false;
    state.params.cpi_enabled = false;
    search(&mut board, 3, &token, &mut debug, &mut state, 1);
    assert_ne!(state.best_move, Move::NO_MOVE);
    assert!(board.is_legal(state.best_move));
    assert!(state.nodes > 0);
}

#[test]
fn depth_one_finds_legal_move_single_thread() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    search(&mut board, 1, &token, &mut debug, &mut state, 1);
    assert_ne!(state.best_move, Move::NO_MOVE);
    assert!(board.is_legal(state.best_move));
    assert!(state.nodes > 0);
}

#[test]
fn searchmoves_filter_pins_best_move() {
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    let pinned = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
    state.searchmoves = vec![pinned];
    search(&mut board, 1, &token, &mut debug, &mut state, 1);
    assert_eq!(state.best_move, pinned);
}

#[test]
fn illegal_searchmoves_filter_still_completes() {
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.searchmoves = vec![Move::new(Square::A1, Square::A8, MoveType::Quiet)];
    search(&mut board, 1, &token, &mut debug, &mut state, 1);

    assert!(state.nodes > 0);
}

#[test]
fn max_nodes_stops_between_iterations() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.max_nodes = 1;
    search(&mut board, 2, &token, &mut debug, &mut state, 1);
    assert!(state.nodes >= 1);
}

#[test]
fn mate_in_caps_depth_and_debug_prints() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = true;
    let mut state = SearchState::new();
    state.mate_in = 1;
    search(&mut board, 2, &token, &mut debug, &mut state, 1);
    assert_ne!(state.best_move, Move::NO_MOVE);
}

#[test]
fn cancelled_entry_returns_no_move() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(true);
    let mut debug = false;
    let mut state = SearchState::new();
    search(&mut board, 2, &token, &mut debug, &mut state, 1);
    assert_eq!(state.best_move, Move::NO_MOVE);
}

#[test]
fn single_move_time_manager_caps_depth() {
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;
    let mut board = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.opt_time = 100;
    state.max_time = 500;
    let pinned = Move::new(Square::H1, Square::G1, MoveType::Quiet);
    if board.is_legal(pinned) {
        state.searchmoves = vec![pinned];
    }
    search(&mut board, 2, &token, &mut debug, &mut state, 1);
    assert_ne!(state.best_move, Move::NO_MOVE);
}

#[test]
fn mate_score_early_exit_with_time_manager() {
    let mut board = BoardState::parse_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.opt_time = 100;
    state.max_time = 500;
    search(&mut board, 2, &token, &mut debug, &mut state, 1);
    assert_ne!(state.best_move, Move::NO_MOVE);
    assert!(state.score > MAX_CENTIPAWN_EVAL - 100);
}

#[test]
fn aspiration_sweep_over_volatile_positions() {
    for fen in [
        crate::common::helpers::STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "7k/8/8/8/8/8/QQQ5/K7 w - - 0 1",
    ] {
        let mut board = BoardState::parse_fen(fen);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert!(state.score.abs() < MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }
}

#[test]
fn worker_max_nodes_abort_across_staggered_depths() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.max_nodes = 1;
    search(&mut board, 1, &token, &mut debug, &mut state, 3);
    assert!(state.nodes >= 1);
}

#[test]
fn multithreaded_mate_finds_move_fast() {
    let mut board = BoardState::parse_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    search(&mut board, 1, &token, &mut debug, &mut state, 2);
    assert_ne!(state.best_move, Move::NO_MOVE);
    assert!(board.is_legal(state.best_move));
}

#[test]
fn time_manager_without_stop_covers_falling_eval() {
    let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    state.opt_time = 100;
    state.max_time = 500;
    search(&mut board, 1, &token, &mut debug, &mut state, 1);
    assert_ne!(state.best_move, Move::NO_MOVE);
    assert!(board.is_legal(state.best_move));
}
