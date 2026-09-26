use super::context::SearchContext;
use super::core::{search, search_internal};
use super::early::{
    apply_iir, coarse_pass, null_move_search, probcut_search, rfp_gate, singular_search,
    tablebase_probe, tt_cutoff,
};
use super::history::beta_cutoff;
use super::moves::{MoveOut, extension_depth, finish_node, gtp_gate, move_scores, search_move};
use super::*;
use crate::common::helpers::STARTING_FEN;
use crate::common::square::Square;
use crate::common::tt::TranspositionTableEntry;
use crate::search::intent::SearchIntent;
use crate::search::nmp;

const MATE_IN_ONE: &str = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
const STALEMATE: &str = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1";
const IN_CHECK_ESCAPE: &str = "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1";
const FEW_PIECE_ENDGAME: &str = "4k3/8/8/8/8/8/8/4K2R w K - 0 1";

fn run_search(fen: &str, depth: u8, alpha: i16, beta: i16) -> (i16, u64) {
    let mut board = BoardState::parse_fen(fen);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let score = search(
        &mut board,
        depth,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    (score, state.nodes)
}

#[test]
fn depth_one_startpos_returns_legal_score() {
    let (score, nodes) = run_search(STARTING_FEN, 1, i16::MIN + 1, i16::MAX - 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn depth_two_endgame_stays_fast_and_bounded() {
    let (score, nodes) = run_search(FEW_PIECE_ENDGAME, 2, i16::MIN + 1, i16::MAX - 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn mate_in_one_scores_near_mate() {
    let (score, _) = run_search(MATE_IN_ONE, 1, i16::MIN + 1, i16::MAX - 1);
    assert!(score > constants::MAX_CENTIPAWN_EVAL - 100);
}

#[test]
fn stalemate_returns_zero() {
    let (score, _) = run_search(STALEMATE, 1, i16::MIN + 1, i16::MAX - 1);

    assert!(score.abs() <= 2, "stalemate score {score}");
}

#[test]
fn cancelled_root_returns_zero() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let cancel = AtomicBool::new(true);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let score = search(
        &mut board,
        1,
        i16::MIN + 1,
        i16::MAX - 1,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert_eq!(score, 0);
}

#[test]
fn fifty_move_draw_returns_zero() {
    let (score, _) = run_search(
        "7k/8/5K2/8/8/8/8/8 b - - 100 150",
        1,
        i16::MIN + 1,
        i16::MAX - 1,
    );
    assert!(score.abs() <= 2, "fifty-move score {score}");
}

#[test]
fn depth_zero_delegates_to_quiescence() {
    let (score, nodes) = run_search(FEW_PIECE_ENDGAME, 0, i16::MIN + 1, i16::MAX - 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn tt_exact_cutoff_hits_on_non_pv() {
    let mut board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(250, 0, board.half_move_clock),
        5,
        Move::NO_MOVE,
        TranspositionEntryType::Exact,
    );
    let score = search(&mut board, 1, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert_eq!(score, 250);
    assert_eq!(state.nodes, 1);
}

#[test]
fn rfp_prunes_with_depressed_beta() {
    let (score, nodes) = run_search(STARTING_FEN, 1, -1000, -999);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert_eq!(nodes, 1);
}

#[test]
fn in_check_evasion_searches_moves() {
    let board = BoardState::parse_fen(IN_CHECK_ESCAPE);
    assert!(board.is_in_check(board.side_to_move));
    let (score, nodes) = run_search(IN_CHECK_ESCAPE, 1, i16::MIN + 1, i16::MAX - 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn root_searchmoves_filter_restricts_to_one() {
    use crate::common::move_type::MoveType;
    let mut board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let only = Move::new(Square::H1, Square::G1, MoveType::Quiet);
    assert!(board.is_legal(only));
    state.searchmoves = vec![only];
    let score = search(
        &mut board,
        1,
        i16::MIN + 1,
        i16::MAX - 1,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert_eq!(pv_table.line().first(), Some(&only));
}

#[test]
fn narrow_window_covers_beta_cutoff_histories() {
    let (score, nodes) = run_search("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1", 2, 0, 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn mate_distance_clamp_returns_alpha_immediately() {
    let (score, nodes) = run_search(STARTING_FEN, 1, 30_000, 30_000);
    assert_eq!(score, 30_000);
    assert_eq!(nodes, 1);
}

#[test]
fn beta_is_mate_skips_rfp_and_nmp_gates() {
    let (score, nodes) = run_search(STARTING_FEN, 1, 30_999, 31_000);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn small_prob_beta_returns_without_full_search() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(1000, 0, board.half_move_clock),
        0,
        Move::NO_MOVE,
        TranspositionEntryType::Exact,
    );
    let score = search(&mut board, 2, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert_eq!(score, 381);
}

#[test]
fn tt_alpha_cutoff_hits_on_shallow_non_pv() {
    let mut board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(-100, 0, board.half_move_clock),
        5,
        Move::NO_MOVE,
        TranspositionEntryType::Alpha,
    );
    let score = search(&mut board, 1, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert_eq!(score, -100);
    assert_eq!(state.nodes, 1);
}

#[test]
fn tt_beta_deep_cutoff_hits_on_stalemate() {
    let mut board = BoardState::parse_fen(STALEMATE);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(500, 0, board.half_move_clock),
        7,
        Move::NO_MOVE,
        TranspositionEntryType::Beta,
    );
    let score = search(&mut board, 6, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert_eq!(score, 500);
    assert_eq!(state.nodes, 1);
}

#[test]
fn tt_shallow_depth_miss_falls_through_to_search() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(100, 0, board.half_move_clock),
        1,
        Move::NO_MOVE,
        TranspositionEntryType::Beta,
    );
    let score = search(&mut board, 2, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 1);
}

#[test]
fn tt_cutoff_skipped_when_halfmove_above_90() {
    let mut board = BoardState::parse_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 97 150");
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(250, 0, board.half_move_clock),
        5,
        Move::NO_MOVE,
        TranspositionEntryType::Exact,
    );
    let score = search(&mut board, 1, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert!(score.abs() <= 2, "gated TT score {score}");
}

#[test]
fn null_move_prune_triggers_on_material_up() {
    let (score, nodes) = run_search("7k/8/8/8/8/8/QQQ5/K7 w - - 0 1", 2, 0, 1);
    assert!(score > 0);
    assert!(score < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn futility_prunes_late_quiets_with_high_alpha() {
    let (score, nodes) = run_search(STARTING_FEN, 2, 20_000, 20_001);
    assert!(score < 20_000);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn see_prune_skips_second_losing_capture() {
    let (score, nodes) = run_search("7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1", 2, 700, 701);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn lmr_reduces_late_quiets_with_pvs_research() {
    let (score, nodes) = run_search(STARTING_FEN, 3, i16::MIN + 1, i16::MAX - 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

fn singular_stalemate_search(tt_score: i16, alpha: i16, beta: i16) -> (i16, u64) {
    use crate::common::move_type::MoveType;
    let mut board = BoardState::parse_fen(STALEMATE);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let tt_best = Move::new(Square::H8, Square::G8, MoveType::Quiet);
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(tt_score, 0, board.half_move_clock),
        6,
        tt_best,
        TranspositionEntryType::Exact,
    );
    let score = search(
        &mut board,
        6,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    (score, state.nodes)
}

#[test]
fn singular_double_extension_on_hopeless_exclusion() {
    let (score, nodes) = singular_stalemate_search(300, 0, 1);
    assert!(score.abs() <= 2, "singular score {score}");
    assert!(nodes > 0);
}

#[test]
fn singular_single_extension_near_margin() {
    let (score, nodes) = singular_stalemate_search(13, 0, 1);
    assert!(score.abs() <= 2, "singular score {score}");
    assert!(nodes > 0);
}

#[test]
fn singular_negative_extension_when_original_above_beta() {
    let (score, nodes) = singular_stalemate_search(10, 0, 1);
    assert!(score.abs() <= 2, "singular score {score}");
    assert!(nodes > 0);
}

#[test]
fn singular_fail_high_returns_with_correction_bonus() {
    let (score, nodes) = singular_stalemate_search(0, -5, -4);
    assert!(score.abs() <= 2, "singular score {score}");
    assert!(nodes > 0);
}

#[test]
fn extension_cap_never_exceeds_two_in_a_row() {
    let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
    crate::eval::nnue::v16::unload_nets();
    let mut board = BoardState::parse_fen(crate::common::helpers::KIWI_PETE_FEN);
    let cancel = AtomicBool::new(false);
    let mut debug = false;
    let mut state = SearchState::new();
    let best = board.find_best_move(8, &cancel, &mut debug, &mut state, 1);
    assert!(state.score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
    assert_ne!(best, Move::NO_MOVE);
    assert!(
        state.nodes < 5_000_000,
        "extension cap failed, visited {} nodes",
        state.nodes
    );
    assert!(
        state.seldepth <= 48,
        "extension cap failed, seldepth {}",
        state.seldepth
    );
}

#[test]
fn extension_streak_slot_roundtrips() {
    let mut state = SearchState::new();
    state.extension_streak[3] = 2;
    assert_eq!(state.extension_streak[3], 2);
    state.reset_search();
    assert_eq!(state.extension_streak[3], 0);
    let worker = state.clone_for_worker(1);
    assert_eq!(worker.extension_streak[3], 0);
}

#[test]
fn check_evasion_deep_copies_eval_stack() {
    let (score, nodes) = run_search(IN_CHECK_ESCAPE, 3, i16::MIN + 1, i16::MAX - 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn mate_against_returns_mated_score() {
    let (score, _) = run_search(
        "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
        1,
        i16::MIN + 1,
        i16::MAX - 1,
    );
    assert_eq!(score, -constants::MAX_CENTIPAWN_EVAL);
}

#[test]
fn coarse_pass_runs_on_deep_stalemate_both_arms() {
    let (lo_score, lo_nodes) = run_search(STALEMATE, 7, 0, 1);
    assert!(lo_score.abs() <= 2, "coarse lo {lo_score}");
    assert!(lo_nodes > 0);
    let (hi_score, hi_nodes) = run_search(STALEMATE, 7, 1000, 1001);
    assert!(hi_score.abs() <= 2, "coarse hi {hi_score}");
    assert!(hi_nodes > 0);
}

#[test]
fn probcut_verifies_captures_on_tiny_board() {
    let (score, nodes) = run_search("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1", 4, 0, 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

fn run_search_at_ply(fen: &str, depth: u8, ply: u8, alpha: i16, beta: i16) -> (i16, u64) {
    let mut board = BoardState::parse_fen(fen);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let score = {
        let mut ctx = SearchContext {
            allow_null_move: true,
            on_pv_path: true,
            previous_pv: &[],
            excluded_move: None,
            cut_node: false,
            gtp_graph: gtp::GtpTreeGraph::new(),
            gtp_parent: None,
            pv_table: &mut pv_table,
            cancellation_token: &cancel,
            search_state: &mut state,
        };
        search_internal(&mut board, depth, ply, alpha, beta, None, &mut ctx)
    };
    (score, state.nodes)
}

#[test]
fn max_ply_delegates_to_quiescence() {
    let (score, nodes) = run_search_at_ply(
        FEW_PIECE_ENDGAME,
        1,
        constants::MAX_PLY as u8,
        i16::MIN + 1,
        i16::MAX - 1,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn psm_disabled_takes_zero_delta() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.params.psm_enabled = false;
    let score = search(
        &mut board,
        1,
        i16::MIN + 1,
        i16::MAX - 1,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
}

#[test]
fn singular_beta_clamped_to_mate_floor() {
    use crate::common::move_type::MoveType;
    let mut board = BoardState::parse_fen(STALEMATE);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let tt_best = Move::new(Square::H8, Square::G8, MoveType::Quiet);
    state.tt.submit_entry(
        board.board_hash,
        tt::TranspositionTable::adjust_score(-30990, 0, board.half_move_clock),
        5,
        tt_best,
        TranspositionEntryType::Exact,
    );
    let score = search(&mut board, 6, 0, 1, &cancel, &[], &mut pv_table, &mut state);
    assert!(score.abs() <= 2, "singular clamp score {score}");
    assert!(state.nodes > 0);
}

const PROBCUT_FEN: &str = "7k/8/8/3pp3/8/8/3R4/K3Q3 w - - 0 1";

fn probcut_window(fen: &str, depth: u8, sub: i16) -> (i16, i16) {
    let mut tmp = BoardState::parse_fen(fen);
    let raw = crate::eval::evaluate_with_depth(&mut tmp, 0, depth);
    let beta = raw.saturating_sub(sub);
    let alpha = beta.saturating_sub(1);
    (alpha, beta)
}

#[test]
fn probcut_improving_and_capture_sort_hit() {
    let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
    crate::eval::nnue::v16::unload_nets();
    let (alpha, beta) = probcut_window(PROBCUT_FEN, 4, 300);
    let mut board = BoardState::parse_fen(PROBCUT_FEN);
    let hash = board.board_hash;
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let score = search(
        &mut board,
        4,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
    let entry = state.tt.probe(hash).expect("probcut stores TT");
    assert_eq!(entry.entry_type, TranspositionEntryType::Beta);
    assert_eq!(entry.depth, 1);
}

#[test]
fn probcut_verify_with_depth_six_hits_tce() {
    let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
    crate::eval::nnue::v16::unload_nets();
    let (alpha, beta) = probcut_window(PROBCUT_FEN, 6, 300);
    let mut board = BoardState::parse_fen(PROBCUT_FEN);
    let hash = board.board_hash;
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let score = search(
        &mut board,
        6,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
    let entry = state.tt.probe(hash).expect("probcut stores TT");
    assert_eq!(entry.entry_type, TranspositionEntryType::Beta);
    assert_eq!(entry.depth, 1);
}

#[test]
fn coarse_pass_updates_tt_best() {
    let (score, nodes) = run_search_at_ply("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1", 7, 60, 0, 1);
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

#[test]
fn pawn_push_to_seventh_gets_extension() {
    let (score, nodes) = run_search(
        "4k3/8/4P3/8/8/8/8/4K3 w - - 0 1",
        2,
        i16::MIN + 1,
        i16::MAX - 1,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(nodes > 0);
}

const PAWN_KING_FEN: &str = "4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1";

fn fail_low_window(fen: &str, depth: u8, add: i16) -> (i16, i16) {
    let mut tmp = BoardState::parse_fen(fen);
    let raw = crate::eval::evaluate_with_depth(&mut tmp, 0, depth);
    let alpha = raw.saturating_add(add);
    let beta = alpha.saturating_add(1);
    (alpha, beta)
}

#[test]
fn quiet_history_negative_prunes_late_quiets() {
    let (alpha, beta) = fail_low_window(PAWN_KING_FEN, 4, 200);
    let mut board = BoardState::parse_fen(PAWN_KING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.params.alp_threshold = 101;
    state.params.gtp_threshold = 0;
    for row in state.move_ordering.history_moves.iter_mut() {
        for s in row.iter_mut() {
            *s = -1000;
        }
    }
    for side in state.move_ordering.quiet_history.iter_mut() {
        for row in side.iter_mut() {
            for s in row.iter_mut() {
                *s = -1000;
            }
        }
    }
    for piece in state.move_ordering.continuation_history.iter_mut() {
        for row in piece.iter_mut() {
            for s in row.iter_mut() {
                *s = -1000;
            }
        }
    }
    let score = search(
        &mut board,
        4,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
}

#[test]
fn psm_sibling_prune_triggers() {
    let (alpha, beta) = fail_low_window(PAWN_KING_FEN, 4, 200);
    let mut board = BoardState::parse_fen(PAWN_KING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.params.alp_threshold = 101;
    state.psm_stack.stack[0].consecutive_fail_lows = 5;
    state.psm_stack.stack[0].hidden = [-256; 128];
    let score = search(
        &mut board,
        4,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
}

#[test]
fn gtp_subtree_prune_triggers() {
    let (alpha, beta) = fail_low_window(PAWN_KING_FEN, 4, 200);
    let mut board = BoardState::parse_fen(PAWN_KING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.params.alp_threshold = 101;
    state.params.gtp_threshold = 100;
    let score = search(
        &mut board,
        4,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
}

#[test]
fn late_history_prune_triggers_on_startpos() {
    let (alpha, beta) = fail_low_window(STARTING_FEN, 3, 200);
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    state.params.alp_threshold = 101;
    for row in state.move_ordering.history_moves.iter_mut() {
        for s in row.iter_mut() {
            *s = -10000;
        }
    }
    for side in state.move_ordering.quiet_history.iter_mut() {
        for row in side.iter_mut() {
            for s in row.iter_mut() {
                *s = -10000;
            }
        }
    }
    for piece in state.move_ordering.continuation_history.iter_mut() {
        for row in piece.iter_mut() {
            for s in row.iter_mut() {
                *s = -10000;
            }
        }
    }
    let score = search(
        &mut board,
        3,
        alpha,
        beta,
        &cancel,
        &[],
        &mut pv_table,
        &mut state,
    );
    assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
    assert!(state.nodes > 0);
}

#[test]
fn beta_cutoff_cancelled_returns_score() {
    let board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
    let cancel = AtomicBool::new(true);
    let mut state = SearchState::new();
    let mv = Move::new(Square::H1, Square::G1, MoveType::Quiet);
    let score = beta_cutoff(
        123,
        mv,
        0,
        &board,
        2,
        None,
        &mut state,
        &cancel,
        &[],
        &[],
        None,
    );
    assert_eq!(score, 123);
}

#[test]
fn beta_cutoff_excluded_returns_score() {
    let board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
    let cancel = AtomicBool::new(false);
    let mut state = SearchState::new();
    let mv = Move::new(Square::H1, Square::G1, MoveType::Quiet);
    let excluded = Move::new(Square::H1, Square::H2, MoveType::Quiet);
    let score = beta_cutoff(
        77,
        mv,
        0,
        &board,
        2,
        None,
        &mut state,
        &cancel,
        &[],
        &[],
        Some(excluded),
    );
    assert_eq!(score, 77);
}

#[test]
fn beta_cutoff_en_passant_updates_capture_history() {
    use crate::common::move_type::MoveType;
    let board = BoardState::parse_fen("7k/8/2b5/3pP3/8/8/8/K2RQ3 w - d6 0 1");
    let cancel = AtomicBool::new(false);
    let mut state = SearchState::new();
    let ep = Move::new(Square::E5, Square::D6, MoveType::EnPassant);
    assert!(board.piece_mapping[Square::E5 as usize] == Piece::Pawn);
    let normal = Move::new(Square::D1, Square::D5, MoveType::Capture);
    let tried = [ep, normal];
    let score = beta_cutoff(
        250,
        ep,
        0,
        &board,
        2,
        None,
        &mut state,
        &cancel,
        &[],
        &tried,
        None,
    );
    assert_eq!(score, 250);
    assert!(state.tt.probe(board.board_hash).is_some());
}

fn test_context<'a>(
    cancel: &'a AtomicBool,
    pv_table: &'a mut PvTable,
    state: &'a mut SearchState,
) -> SearchContext<'a> {
    SearchContext {
        allow_null_move: true,
        on_pv_path: true,
        previous_pv: &[],
        excluded_move: None,
        cut_node: false,
        gtp_graph: gtp::GtpTreeGraph::new(),
        gtp_parent: None,
        pv_table,
        cancellation_token: cancel,
        search_state: state,
    }
}

fn test_tt_entry() -> TranspositionTableEntry {
    TranspositionTableEntry {
        hash: 1,
        score: 0,
        best_move: Move::new(Square::E2, Square::E4, MoveType::DoublePush),
        depth: 10,
        entry_type: TranspositionEntryType::Exact,
        generation: 0,
        eval: 0,
        raw_eval: 0,
    }
}

#[test]
fn early_exit_guards_cover_all_arms() {
    let entry = test_tt_entry();
    assert!(tt_cutoff(entry, true, false, 5, 0, 10, false, 0, 0).is_none());
    assert!(tt_cutoff(entry, false, true, 5, 0, 10, false, 0, 0).is_none());
    let mut none_entry = entry;
    none_entry.entry_type = TranspositionEntryType::None;
    assert!(tt_cutoff(none_entry, false, false, 5, 0, 10, false, 0, 0).is_none());

    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let nt = NodeThreats::compute(&board);
    let mut ctx = test_context(&cancel, &mut pv_table, &mut state);
    assert!(
        rfp_gate(
            &nt, 0, 10, 8, 0, false, true, true, false, 0, None, 0, 30000, false, &mut ctx
        )
        .is_none()
    );
    let tb = tablebase_probe(&mut board, 0, 8, -10, 10, -32000, false, 0, &mut ctx);
    assert!(tb.score.is_none());
    assert_eq!((tb.alpha, tb.best), (-10, -32000));
    let coarse = coarse_pass(&mut board, 4, 0, -10, 10, true, false, None, None, &mut ctx);
    assert!(!coarse.cancelled);
    assert_eq!(coarse.depth, 4);
    assert!(!coarse.failed_low);
    assert!(coarse.tt_best.is_none());
    assert_eq!(apply_iir(10, true, false, true, false, false), 10);

    let mut check_board = BoardState::parse_fen(IN_CHECK_ESCAPE);
    let sing = singular_search(
        &mut check_board,
        8,
        0,
        0,
        0,
        true,
        true,
        false,
        None,
        None,
        false,
        None,
        0,
        30000,
        &mut ctx,
    );
    assert!(sing.score.is_none());
    assert_eq!(sing.extension, 0);
    assert_eq!(sing.depth_bonus, 0);
}

#[test]
fn cancelled_helpers_bail_out_deterministically() {
    let cancel = AtomicBool::new(true);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let before = board.clone();
    let mut ctx = test_context(&cancel, &mut pv_table, &mut state);

    let tb = tablebase_probe(&mut board, 1, 8, -10, 10, -32000, false, 0, &mut ctx);
    assert_eq!(tb.score, Some(0));

    let tiny = "7k/8/8/8/3p4/8/3R4/K7 w - - 0 1";
    let mut cap_board = BoardState::parse_fen(tiny);
    let cap_nt = NodeThreats::compute(&cap_board);
    let prob = probcut_search(
        &mut cap_board,
        8,
        0,
        -500,
        0,
        false,
        false,
        false,
        0,
        30000,
        &cap_nt,
        &mut ctx,
    );
    assert_eq!(prob, Some(0));
    assert_eq!(cap_board, BoardState::parse_fen(tiny));

    nmp::clear_nmp_state();
    let material = "7k/8/8/8/8/8/QQQ5/K7 w - - 0 1";
    let mut null_board = BoardState::parse_fen(material);
    let nul = null_move_search(
        &mut null_board,
        8,
        0,
        0,
        2000,
        0,
        false,
        false,
        false,
        true,
        None,
        0,
        30000,
        &mut ctx,
    );
    assert_eq!(nul, Some(0));
    assert_eq!(null_board, BoardState::parse_fen(material));

    let coarse = coarse_pass(
        &mut board, 10, 0, -10, 10, false, false, None, None, &mut ctx,
    );
    assert!(coarse.cancelled);

    let e2e4 = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
    let nt = NodeThreats::compute(&board);
    board.make_move(e2e4);
    let sm = search_move(
        &mut board,
        8,
        0,
        -50,
        50,
        false,
        e2e4,
        false,
        false,
        false,
        false,
        true,
        true,
        0,
        10,
        0,
        0,
        -1000,
        None,
        false,
        None,
        true,
        ctx.gtp_graph,
        0,
        5,
        0,
        &nt,
        &mut ctx,
    );
    assert!(matches!(sm, MoveOut::Cancelled));
    assert_eq!(board, before);

    let fin = finish_node(
        &mut board,
        0,
        0,
        30000,
        false,
        false,
        0,
        true,
        8,
        10,
        Move::NO_MOVE,
        TranspositionEntryType::Alpha,
        None,
        &[],
        0,
        false,
        crate::search::bmo::BanditArm::CapturesFirst,
        &mut ctx,
    );
    assert_eq!(fin, 0);
}

#[test]
fn nmp_verification_path_executes() {
    let material = "4k3/8/8/8/8/8/PPPP4/R3K3 w - - 0 1";
    nmp::clear_nmp_state();
    let mut board = BoardState::parse_fen(material);
    let before = board.clone();
    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let mut ctx = test_context(&cancel, &mut pv_table, &mut state);
    let mate_bound = constants::MAX_CENTIPAWN_EVAL - constants::MAX_PLY as i16;
    let out = null_move_search(
        &mut board, 16, 1, -100, 800, 0, false, false, false, true, None, 0, mate_bound, &mut ctx,
    );
    nmp::clear_nmp_state();
    assert!(out.is_some());
    assert_eq!(board, before);
}

#[test]
fn lmr_adjust_arms_covered() {
    let cancel = AtomicBool::new(true);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let before = board.clone();
    let nt = NodeThreats::compute(&board);
    let e2e4 = Move::new(Square::E2, Square::E4, MoveType::DoublePush);

    state.last_intent = SearchIntent::Attack;
    let out = {
        let mut ctx = test_context(&cancel, &mut pv_table, &mut state);
        board.make_move(e2e4);
        search_move(
            &mut board,
            8,
            0,
            -50,
            50,
            false,
            e2e4,
            false,
            false,
            false,
            false,
            true,
            true,
            0,
            10,
            0,
            0,
            -1000,
            None,
            false,
            None,
            true,
            ctx.gtp_graph,
            0,
            5,
            0,
            &nt,
            &mut ctx,
        )
    };
    assert!(matches!(out, MoveOut::Cancelled));

    let mut check_board = BoardState::parse_fen("k3r3/8/8/8/8/3n4/2P1Q3/4K3 w - - 0 1");
    let check_nt = NodeThreats::compute(&check_board);
    let capture = Move::new(Square::C2, Square::D3, MoveType::Capture);
    state.last_intent = SearchIntent::Improvement;
    let mut ctx = test_context(&cancel, &mut pv_table, &mut state);
    check_board.make_move(capture);
    let out = search_move(
        &mut check_board,
        8,
        0,
        -50,
        50,
        false,
        capture,
        true,
        false,
        false,
        false,
        true,
        true,
        -5000,
        10,
        0,
        0,
        -1000,
        None,
        false,
        None,
        true,
        ctx.gtp_graph,
        0,
        5,
        0,
        &check_nt,
        &mut ctx,
    );
    assert!(matches!(out, MoveOut::Cancelled));
    assert_eq!(board, before);
}

#[test]
fn extension_max_ply_streak_capped() {
    let mut state = SearchState::new();
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let e2e4 = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
    let max_ply = constants::MAX_PLY as u8;
    let (depth, extension) = extension_depth(
        &mut board, &mut state, e2e4, None, 0, 8, false, false, false, max_ply, None,
    );
    assert_eq!((depth, extension), (8, 0));
}

#[test]
fn move_scores_capture_to_empty() {
    let state = SearchState::new();
    let board = BoardState::parse_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    let mv = Move::new(Square::D1, Square::D5, MoveType::Capture);
    assert_eq!(move_scores(&board, &state, mv, true, None), (0, 0));
}

#[test]
fn gtp_gate_taken_prunes() {
    let mut graph = gtp::GtpTreeGraph::new();
    let idx = graph.add_node(gtp::GtpNode {
        depth: 3,
        eval_margin: -400,
        history_score: -2000,
        ..gtp::GtpNode::default()
    });
    let score = gtp::GtpModel::message_passing(&graph)[idx];
    assert!(gtp_gate(
        true,
        false,
        false,
        false,
        false,
        false,
        3,
        10,
        false,
        &graph,
        idx,
        score.saturating_add(1),
    ));
}

#[test]
fn tablebase_probe_hits_real_tables() {
    use crate::syzygy::{SYZYGY_TEST_LOCK, set_path, set_probe_depth, set_probe_limit};
    let _serial = SYZYGY_TEST_LOCK.lock().unwrap();
    if !std::path::Path::new("tables/KQvK.rtbw").exists() {
        return;
    }
    let (old_limit, old_depth) = crate::syzygy::probe_config();
    set_path("tables").expect("load tables");
    set_probe_limit(7);
    set_probe_depth(1);

    let cancel = AtomicBool::new(false);
    let mut pv_table = PvTable::new();
    let mut state = SearchState::new();
    let mut ctx = test_context(&cancel, &mut pv_table, &mut state);

    let mut win = BoardState::parse_fen("8/8/8/8/8/2K5/2Q5/7k w - - 0 1");
    let hit = tablebase_probe(&mut win, 1, 8, -100, 100, -32000, false, 0, &mut ctx);
    assert!(hit.score.is_some());

    let raise = tablebase_probe(&mut win, 1, 8, 0, 31000, -32000, true, 0, &mut ctx);
    assert!(raise.score.is_none());
    assert!(raise.alpha > 0 && raise.best > 0 && raise.alpha == raise.best);

    let ret = tablebase_probe(&mut win, 1, 8, 0, 100, -32000, true, 0, &mut ctx);
    assert!(ret.score.is_some());

    let mut draw = BoardState::parse_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1");
    let dull = tablebase_probe(&mut draw, 1, 8, -100, 100, -32000, false, 0, &mut ctx);
    assert_eq!(dull.score, Some(0));

    let mut lost = BoardState::parse_fen("7k/8/8/8/8/2K5/2Q5/8 b - - 0 1");
    let doomed = tablebase_probe(&mut lost, 1, 8, -100, 100, -32000, false, 0, &mut ctx);
    assert!(doomed.score.is_some());

    set_probe_limit(old_limit);
    set_probe_depth(old_depth);
    set_path("<empty>").ok();
}
