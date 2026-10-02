use std::sync::atomic::AtomicBool;
use whale::board::state::BoardState;
use whale::common::helpers::{ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN};
use whale::common::moves::Move;
use whale::search::search_state::SearchState;

const SIGNATURE_FENS: [&str; 3] = [STARTING_FEN, KIWI_PETE_FEN, ENDGAME_FEN];
const SIGNATURE_DEPTH: u8 = 6;

fn run_signature_once() -> (u64, Vec<Move>) {
    whale::common::random::reset_seed();
    whale::init();
    let shared_tt = std::sync::Arc::new(whale::common::tt::TranspositionTable::new_mb(1));
    let mut total_nodes = 0u64;
    let mut bestmoves = Vec::new();
    for fen in SIGNATURE_FENS {
        let mut board = BoardState::parse_fen(fen);
        let cancel = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.tt = std::sync::Arc::clone(&shared_tt);
        let best = board.find_best_move(SIGNATURE_DEPTH, &cancel, &mut debug, &mut state, 1);
        assert_ne!(best, Move::NO_MOVE);
        total_nodes += state.nodes;
        bestmoves.push(best);
    }
    (total_nodes, bestmoves)
}

#[test]
fn bench_signature_is_deterministic() {
    let (nodes_a, moves_a) = run_signature_once();
    let (nodes_b, moves_b) = run_signature_once();
    assert!(nodes_a > 0);
    assert_eq!(nodes_a, nodes_b);
    assert_eq!(moves_a, moves_b);
    println!("Nodes searched  : {nodes_a}");
}

#[test]
fn repro_fixed_depth_gives_same_bestmove() {
    whale::common::random::reset_seed();
    whale::init();
    let run_once = || {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let cancel = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.tt = std::sync::Arc::new(whale::common::tt::TranspositionTable::new_mb(1));
        let best = board.find_best_move(5, &cancel, &mut debug, &mut state, 1);
        (best, state.nodes)
    };
    let (best_a, nodes_a) = run_once();
    whale::common::random::reset_seed();
    let (best_b, nodes_b) = run_once();
    assert_ne!(best_a, Move::NO_MOVE);
    assert_eq!(best_a, best_b);
    assert_eq!(nodes_a, nodes_b);
}
