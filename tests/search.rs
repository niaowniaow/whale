use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;
use whale::board::state::BoardState;
use whale::common::helpers::{ADVANCED_MOVE_FEN, ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN};
use whale::common::move_type::MoveType;
use whale::common::moves::Move;
use whale::search::search_state::SearchState;

fn find_move_from_move_list(board: &mut BoardState, expected_move: Move) -> Move {
    let mut move_list = whale::common::move_list::MoveList::new();
    board.generate_moves(&mut move_list);

    for m in move_list.iter() {
        if m.mv.source == expected_move.source
            && m.mv.target == expected_move.target
            && (expected_move.move_type == MoveType::Quiet
                || ((m.mv.move_type.value() & !8) == expected_move.move_type.value()))
        {
            return m.mv;
        }
    }

    Move::NO_MOVE
}

fn assert_search_completes(position: &str, depth: u8) -> (u64, i16) {
    let mut board_state = BoardState::parse_fen(position);
    let cancellation_token = AtomicBool::new(false);
    let mut debug_mode = false;
    let mut search_state = SearchState::new();

    let best_move = board_state.find_best_move(
        depth,
        &cancellation_token,
        &mut debug_mode,
        &mut search_state,
        1,
    );

    assert_ne!(best_move, Move::NO_MOVE, "engine must return a valid move");
    assert!(
        search_state.nodes > 0,
        "engine must visit at least one node"
    );
    assert!(
        search_state.score > -30000 && search_state.score < 30000,
        "score {} out of valid range",
        search_state.score
    );

    (search_state.nodes, search_state.score)
}

fn assert_tactic_best_move(fen: &str, move_lan: &str) {
    assert_tactic_best_move_timed(fen, move_lan, 3000, 20);
}

fn assert_tactic_best_move_timed(fen: &str, move_lan: &str, time_ms: u64, depth: u8) {
    let mut board_state = BoardState::parse_fen(fen);

    let cancellation_token = Arc::new(AtomicBool::new(false));
    let cancellation_writer = Arc::clone(&cancellation_token);
    let cancellation_worker = thread::spawn(move || {
        thread::sleep(Duration::from_millis(time_ms));
        cancellation_writer.store(true, Ordering::Relaxed);
    });

    let mut debug_mode = false;
    let mut search_state = SearchState::new();
    let best_move = board_state.find_best_move(
        depth,
        cancellation_token.as_ref(),
        &mut debug_mode,
        &mut search_state,
        1,
    );

    let expected_move =
        Move::parse_long_algebraic(move_lan).expect("Failed to parse expected move");
    let expected_move = find_move_from_move_list(&mut board_state, expected_move);

    cancellation_worker.join().unwrap();

    assert_eq!(expected_move, best_move);
}

macro_rules! search_completion_test_case {
    ($name:ident, $fen:expr, $depth:expr) => {
        #[test]
        fn $name() {
            assert_search_completes($fen, $depth);
        }
    };
}

macro_rules! tactic_test_case {
    ($name:ident, $fen:expr, $best_move:expr) => {
        #[test]
        fn $name() {
            assert_tactic_best_move($fen, $best_move);
        }
    };
    ($name:ident, skip = $reason:literal, $fen:expr, $best_move:expr) => {
        #[test]
        #[ignore = $reason]
        fn $name() {
            assert_tactic_best_move($fen, $best_move);
        }
    };
}

search_completion_test_case!(traversal_starting, STARTING_FEN, 6);
search_completion_test_case!(traversal_endgame, ENDGAME_FEN, 6);
search_completion_test_case!(traversal_advanced, ADVANCED_MOVE_FEN, 6);
search_completion_test_case!(traversal_kiwi_pete, KIWI_PETE_FEN, 6);

tactic_test_case!(
    tactic_mate_in_one_back_rank,
    "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
    "e1e8"
);
tactic_test_case!(
    tactic_mate_in_one_scholar,
    "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
    "f3f7"
);
tactic_test_case!(
    tactic_random_puzzle_position,
    "r4r2/pb4kp/1p4p1/1P6/2P1pRp1/P3B3/7P/5RK1 w - - 0 29",
    "f4f8"
);
tactic_test_case!(
    tactic_zugzwang_verification_5,
    "1q1k4/2Rr4/8/2Q3K1/8/8/8/8 w - - 0 1",
    "g5h6"
);

tactic_test_case!(
    tactic_transposition_table_verification,
    skip = "Deep pawn endgame requires depth > 30",
    "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - -",
    "a1b1"
);
tactic_test_case!(
    tactic_zugzwang_verification_1,
    skip = "Deep zugzwang requires null-move verification search",
    "8/8/1p1r1k2/p1pPN1p1/P3KnP1/1P6/8/3R4 b - - 0 1",
    "f4d5"
);
tactic_test_case!(
    tactic_zugzwang_verification_2,
    skip = "Deep zugzwang requires null-move verification search",
    "7k/5K2/5P1p/3p4/6P1/3p4/8/8 w - - 0 1",
    "g4g5"
);
tactic_test_case!(
    tactic_zugzwang_verification_3,
    skip = "Deep zugzwang requires null-move verification search",
    "8/6B1/p5p1/Pp4kp/1P5r/5P1Q/4q1PK/8 w - - 0 32",
    "h3h4"
);
tactic_test_case!(
    tactic_zugzwang_verification_4,
    skip = "Deep zugzwang requires null-move verification search",
    "8/8/p1p5/1p5p/1P5p/8/PPP2K1p/4R1rk w - - 0 1",
    "e1f1"
);

#[test]
fn models_family_search_integration() {
    for model_name in ["whale_small", "whale_medium", "whale_big"] {
        let path = format!("models/{model_name}.nnue");
        if !std::path::Path::new(&path).exists() {
            continue;
        }
        let res = whale::eval::nnue::v16::set_eval_file("Model", model_name);
        assert!(res.is_ok(), "Failed to activate {model_name}: {:?}", res);
        assert_eq!(whale::eval::nnue::v16::active_model_name(), model_name);

        // Run search with the loaded model on tactical position
        assert_tactic_best_move("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8");
    }
    let _ = whale::eval::nnue::v16::set_eval_file("Model", "embedded");
    assert_eq!(whale::eval::nnue::v16::active_model_name(), "embedded");
}

#[test]
fn test_game_position_depth_8() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let moves = "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f6e4 d2d4 b7b5 a4b3 d7d5 d4e5 c8e6 c2c3 f8e7 c1e3 e8g8 b1d2 e6g4 d2e4 d5e4 d1d5 d8d5 b3d5 e4f3 d5c6 f3g2 g1g2 a8b8 f1e1 g4e6 e1d1 b5b4 c6d7 b4c3 b2c3 b8b2 d7e6 f7e6 d1d7 e7h4 a1f1 b2a2 d7d4";
    for m_str in moves.split_whitespace() {
        let mut list = whale::common::move_list::MoveList::new();
        board.generate_moves(&mut list);
        let m = list
            .iter()
            .map(|e| e.mv)
            .find(|mv| {
                let promo = mv
                    .promotion_char()
                    .map(|c| c.to_string())
                    .unwrap_or_default();
                format!("{}{}{}", mv.source, mv.target, promo) == m_str
            })
            .expect("move must be legal");
        board.make_move(m);
    }
    let cancellation_token = AtomicBool::new(false);
    let mut debug_mode = true;
    let mut search_state = SearchState::new();
    let start = std::time::Instant::now();
    let best = board.find_best_move(
        12,
        &cancellation_token,
        &mut debug_mode,
        &mut search_state,
        4,
    );
    println!(
        "Elapsed: {:?}, best: {:?}, nodes: {}",
        start.elapsed(),
        best,
        search_state.nodes
    );
}
