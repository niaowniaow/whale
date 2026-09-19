use crate::common::constants::MAX_PLY;
use crate::common::move_list::MoveList;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::tt::{self, TranspositionEntryType};
use crate::eval::evaluate_with_optimism;
use crate::search::bmo::BanditArm;
use crate::search::draw;
use crate::search::move_picker::MovePicker;
use crate::search::search_state::{SearchState, stm_is_white};
use crate::{board::state::BoardState, common::constants::MAX_CENTIPAWN_EVAL};
use std::sync::atomic::{AtomicBool, Ordering};

fn has_legal_move(board_state: &BoardState) -> bool {
    let stm = board_state.side_to_move;
    let checkers = board_state.checkers(stm).0;
    let pinned = board_state.pinned_pieces(stm).0;
    let mut moves = MoveList::new();
    board_state.generate_captures(&mut moves);
    if moves
        .iter()
        .any(|entry| board_state.is_legal_with(entry.mv, checkers, pinned))
    {
        return true;
    }
    board_state.generate_quiets(&mut moves);
    moves
        .iter()
        .any(|entry| board_state.is_legal_with(entry.mv, checkers, pinned))
}

pub fn search(
    board_state: &mut BoardState,
    mut alpha: i16,
    beta: i16,
    ply: u8,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
) -> i16 {
    // FIX ABORT-PROPAGATION: 0 is not a valid score; callers re-check cancelled.
    if cancellation_token.load(Ordering::Relaxed) {
        return 0;
    }

    if board_state.is_draw_in_search(ply as u16) {
        let base = draw::draw_score(search_state.nodes);
        return draw::apply_contempt(
            base,
            stm_is_white(board_state.side_to_move),
            search_state.contempt_cp,
            search_state.draw_score_cp,
        );
    }

    // PERF: same once-per-node checkers/pinners cache as negamax.
    let node_checkers = board_state.checkers(board_state.side_to_move).0;
    let node_pinned = board_state.pinned_pieces(board_state.side_to_move).0;
    let in_check = node_checkers != 0;
    if board_state.occupancy().count_ones() <= 10 && !has_legal_move(board_state) {
        return if in_check {
            -MAX_CENTIPAWN_EVAL + ply as i16
        } else {
            let base = draw::draw_score(search_state.nodes);
            draw::apply_contempt(
                base,
                stm_is_white(board_state.side_to_move),
                search_state.contempt_cp,
                search_state.draw_score_cp,
            )
        };
    }

    let optimism = search_state.optimism[board_state.side_to_move as usize];
    let halfmove = board_state.half_move_clock;
    // PV-ness derived from the window (negamax passes no explicit flag).
    let is_pv = beta > alpha + 1;

    if ply as usize >= MAX_PLY {
        return evaluate_with_optimism(&mut *board_state, optimism);
    }

    // FIX NODES-SELDEPTH: single count per node (removed double count on TT
    // cutoff); seldepth tracks deepest PV ply including qsearch.
    search_state.nodes += 1;
    if is_pv {
        let sd = ply as u32 + 1;
        if sd > search_state.seldepth {
            search_state.seldepth = sd;
        }
    }

    let original_alpha = alpha;
    let tt_entry = search_state.tt.probe(board_state.board_hash);
    // FIX QSEARCH-TT: cut only outside PV (Stockfish search.cpp:1723-1726;
    // qsearch entries live at depth 0 so any stored depth suffices, but PV
    // nodes must never cut).
    if let Some(entry) = tt_entry
        && !is_pv
    {
        let tt_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
        let cutoff = match entry.entry_type {
            TranspositionEntryType::Exact => true,
            TranspositionEntryType::Alpha => tt_score <= alpha,
            TranspositionEntryType::Beta => tt_score >= beta,
            TranspositionEntryType::None => false,
        };
        if cutoff {
            return tt_score;
        }
    }

    let mut futility_base = -MAX_CENTIPAWN_EVAL;
    // FIX QSEARCH-TT: best move/value tracked so fail-high stores carry the
    // move (never NO_MOVE once a move improved the score) and fail-low stores
    // carry the most accurate bound (Stockfish-style bestValue).
    let mut best_move = Move::NO_MOVE;
    let mut best_value: i16 = -MAX_CENTIPAWN_EVAL;

    if !in_check {
        // FIX QSEARCH-LERP: single eval path (removed fast_eval +/-200/1200
        // prunes that used a different eval than the stand-pat below).
        let eval = if let Some(entry) = tt_entry {
            let s = tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
            match entry.entry_type {
                TranspositionEntryType::Exact => s,
                TranspositionEntryType::Beta if s >= beta => s,
                _ => evaluate_with_optimism(&mut *board_state, optimism),
            }
        } else {
            evaluate_with_optimism(&mut *board_state, optimism)
        };
        let continue_qs = search_state.params.lqt_enabled
            && crate::search::lqt::should_continue_quiescence(board_state, eval, alpha, beta, ply);
        // FIX QSEARCH-TT: stand-pat fail-high stores LOWER before returning.
        if eval >= beta && !continue_qs {
            let mut stand_pat = beta;
            if eval.abs() < MAX_CENTIPAWN_EVAL - 200 {
                stand_pat = ((441 * eval as i32 + 583 * beta as i32) / 1024) as i16;
            }
            if !cancellation_token.load(Ordering::Relaxed) {
                search_state.tt.submit_entry(
                    board_state.board_hash,
                    tt::TranspositionTable::adjust_score(stand_pat, ply as i32, halfmove),
                    0,
                    best_move,
                    TranspositionEntryType::Beta,
                );
            }
            return stand_pat;
        }
        if eval > alpha {
            alpha = eval;
        }
        // FIX QSEARCH-LERP: top-level delta prune removed — reaching alpha by
        // queen-win alone is handled per-move below via futility_base, which
        // updates bestValue instead of returning alpha early.
        futility_base = eval + 306;
        best_value = eval;
    }

    let mut move_picker = if !in_check {
        MovePicker::new_qsearch(ply as usize)
    } else {
        MovePicker::new(
            None,
            None,
            None,
            ply as usize,
            None,
            BanditArm::CapturesFirst,
        )
    };

    let mut has_legal_moves = false;
    let mut move_count = 0;

    while let Some(move_obj) = move_picker.next(
        board_state,
        &search_state.move_ordering,
        &mut search_state.captures_stack[ply as usize],
        &mut search_state.quiets_stack[ply as usize],
    ) {
        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if !board_state.is_legal_with(move_obj, node_checkers, node_pinned) {
            continue;
        }

        if !in_check {
            if board_state.see(move_obj) < -74 {
                continue;
            }

            if !move_obj.is_promotion() {
                move_count += 1;
                if move_count > 2 {
                    continue;
                }

                let captured_piece = if move_obj.move_type == MoveType::EnPassant {
                    Piece::Pawn
                } else {
                    board_state.piece_mapping[move_obj.target as usize]
                };
                let futility_val = futility_base + captured_piece.see_value();
                if futility_val <= alpha {
                    best_value = best_value.max(futility_val);
                    continue;
                }
                if board_state.see(move_obj) < alpha.saturating_sub(futility_base) {
                    best_value = best_value.max(alpha.min(futility_base));
                    continue;
                }
            }
        }

        board_state.make_move(move_obj);

        has_legal_moves = true;

        let score = -search(
            board_state,
            -beta,
            -alpha,
            ply + 1,
            cancellation_token,
            search_state,
        );
        board_state.unmake_move(move_obj);

        if cancellation_token.load(Ordering::Relaxed) {
            return 0;
        }

        // FIX QSEARCH-TT: fail-high stores LOWER with the best move first.
        if score >= beta {
            best_move = move_obj;
            let mut fail_high = beta;
            if score.abs() < MAX_CENTIPAWN_EVAL - 200 && score > beta {
                fail_high = ((462 * score as i32 + 562 * beta as i32) / 1024) as i16;
            }
            if !cancellation_token.load(Ordering::Relaxed) {
                search_state.tt.submit_entry(
                    board_state.board_hash,
                    tt::TranspositionTable::adjust_score(fail_high, ply as i32, halfmove),
                    0,
                    best_move,
                    TranspositionEntryType::Beta,
                );
            }
            return fail_high;
        }
        if score > best_value {
            best_value = score;
            if score > alpha {
                alpha = score;
                best_move = move_obj;
            }
        }
    }

    if !in_check && !cancellation_token.load(Ordering::Relaxed) {
        let mut promos = [Move::NO_MOVE; 32];
        let mut promo_count = 0usize;
        {
            let quiets = &mut search_state.quiets_stack[ply as usize];
            board_state.generate_quiets(quiets);
            for i in (0..quiets.len()).rev() {
                let move_obj = quiets[i].mv;
                if move_obj.is_promotion() && promo_count < promos.len() {
                    promos[promo_count] = move_obj;
                    promo_count += 1;
                }
            }
        }
        for &move_obj in promos.iter().take(promo_count) {
            if !board_state.is_legal_with(move_obj, node_checkers, node_pinned) {
                continue;
            }
            board_state.make_move(move_obj);
            let score = -search(
                board_state,
                -beta,
                -alpha,
                ply + 1,
                cancellation_token,
                search_state,
            );
            board_state.unmake_move(move_obj);

            if cancellation_token.load(Ordering::Relaxed) {
                return 0;
            }

            if score >= beta {
                if score.abs() < MAX_CENTIPAWN_EVAL - 200 && score > beta {
                    return ((462 * score as i32 + 562 * beta as i32) / 1024) as i16;
                }
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
    }

    if in_check && !has_legal_moves {
        return -MAX_CENTIPAWN_EVAL + ply as i16;
    }

    // Reconcile with the untouched rescue-scan above (it raises alpha only).
    let final_value = best_value.max(alpha);
    let entry_type = if final_value >= beta {
        TranspositionEntryType::Beta
    } else if final_value > original_alpha {
        TranspositionEntryType::Exact
    } else {
        TranspositionEntryType::Alpha
    };
    // FIX QSEARCH-TT: never store NO_MOVE once a move improved the score; skip
    // the store entirely on abort (no valid bound to report).
    if !cancellation_token.load(Ordering::Relaxed) {
        search_state.tt.submit_entry(
            board_state.board_hash,
            tt::TranspositionTable::adjust_score(final_value, ply as i32, halfmove),
            0,
            best_move,
            entry_type,
        );
    }

    final_value
}

#[cfg(test)]
mod tests {
    use super::*;

    const STALEMATE: &str = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1";
    const QUIET_ONLY: &str = "7k/8/5K2/8/8/8/Q7/8 b - - 0 1";
    const CHECKMATE: &str = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1";

    #[test]
    fn stalemate_precedes_stand_pat_and_window_bounds() {
        let mut board = BoardState::parse_fen(STALEMATE);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        assert!(!board.is_in_check(board.side_to_move));
        assert!(!board.is_draw_in_search(4));
        assert!(!has_legal_move(&board));
        let eval = evaluate_with_optimism(&mut board, 0);
        assert_ne!(eval, 0);
        for (alpha, beta) in [
            (eval - 2, eval - 1),
            (eval - 100, eval - 1),
            (100, 101),
            (-MAX_CENTIPAWN_EVAL, MAX_CENTIPAWN_EVAL),
        ] {
            state.tt.clear();
            // Draw score with node parity: near-zero.
            let s = search(&mut board, alpha, beta, 4, &cancel, &mut state);
            assert!(s.abs() <= 2, "stalemate qsearch {s}");
        }
    }

    #[test]
    fn stalemate_precedes_nonzero_tt_cutoffs() {
        let mut board = BoardState::parse_fen(STALEMATE);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        for (entry_type, score) in [
            (TranspositionEntryType::Exact, 321),
            (TranspositionEntryType::Alpha, -321),
            (TranspositionEntryType::Beta, 321),
        ] {
            for (alpha, beta) in [(0, 1), (-100, 100)] {
                state.tt.clear();
                state.tt.submit_entry(
                    board.board_hash,
                    tt::TranspositionTable::adjust_score(score, 4, board.half_move_clock),
                    12,
                    Move::NO_MOVE,
                    entry_type,
                );
                assert_eq!(state.tt.probe(board.board_hash).unwrap().score, score);
                let s = search(&mut board, alpha, beta, 4, &cancel, &mut state);
                assert!(s.abs() <= 2, "stalemate qsearch {s}");
            }
        }
    }

    #[test]
    fn quiet_legal_moves_are_not_stalemate() {
        let mut board = BoardState::parse_fen(QUIET_ONLY);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let mut captures = MoveList::new();
        board.generate_captures(&mut captures);
        assert!(captures.is_empty());
        assert!(!board.is_in_check(board.side_to_move));
        assert!(has_legal_move(&board));
        let eval = evaluate_with_optimism(&mut board, 0);
        assert_ne!(eval, 0);
        for ply in [4, MAX_PLY as u8] {
            state.tt.clear();
            assert_eq!(
                search(
                    &mut board,
                    -MAX_CENTIPAWN_EVAL,
                    MAX_CENTIPAWN_EVAL,
                    ply,
                    &cancel,
                    &mut state,
                ),
                eval,
            );
        }
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(321, 4, board.half_move_clock),
            12,
            Move::NO_MOVE,
            TranspositionEntryType::Exact,
        );
        assert_eq!(search(&mut board, 0, 1, 4, &cancel, &mut state), 321);
    }

    #[test]
    fn terminal_scores_remain_distinct_at_max_ply_and_with_tt() {
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        for (fen, in_check) in [(STALEMATE, false), (CHECKMATE, true)] {
            let mut board = BoardState::parse_fen(fen);
            assert_eq!(board.is_in_check(board.side_to_move), in_check);
            assert!(!has_legal_move(&board));
            for ply in [0, 4, MAX_PLY as u8] {
                state.tt.clear();
                state.tt.submit_entry(
                    board.board_hash,
                    tt::TranspositionTable::adjust_score(321, ply as i32, board.half_move_clock),
                    12,
                    Move::NO_MOVE,
                    TranspositionEntryType::Exact,
                );
                let s = search(&mut board, 0, 1, ply, &cancel, &mut state);
                if in_check {
                    assert_eq!(s, -MAX_CENTIPAWN_EVAL + ply as i16);
                } else {
                    assert!(s.abs() <= 2, "stalemate qsearch {s}");
                }
            }
        }
    }

    #[test]
    fn cancelled_search_returns_zero_immediately() {
        let mut board = BoardState::parse_fen(QUIET_ONLY);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(true);
        assert_eq!(
            search(
                &mut board,
                -MAX_CENTIPAWN_EVAL,
                MAX_CENTIPAWN_EVAL,
                0,
                &cancel,
                &mut state
            ),
            0
        );
        let mut draw = BoardState::parse_fen("7k/8/5K2/8/8/8/8/8 b - - 100 150");
        let live = AtomicBool::new(false);
        assert_eq!(
            search(
                &mut draw,
                -MAX_CENTIPAWN_EVAL,
                MAX_CENTIPAWN_EVAL,
                4,
                &live,
                &mut state
            ),
            0
        );
    }

    #[test]
    fn non_pv_tt_bounds_cut() {
        let mut board = BoardState::parse_fen(QUIET_ONLY);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        state.tt.clear();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(500, 4, board.half_move_clock),
            0,
            Move::NO_MOVE,
            TranspositionEntryType::Beta,
        );
        assert_eq!(search(&mut board, 0, 1, 4, &cancel, &mut state), 500);
        state.tt.clear();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(-500, 4, board.half_move_clock),
            0,
            Move::NO_MOVE,
            TranspositionEntryType::Alpha,
        );
        assert_eq!(search(&mut board, 0, 1, 4, &cancel, &mut state), -500);
        state.tt.clear();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(500, 4, board.half_move_clock),
            0,
            Move::NO_MOVE,
            TranspositionEntryType::Beta,
        );
        let eval = evaluate_with_optimism(&mut board, 0);
        assert_eq!(
            search(
                &mut board,
                -MAX_CENTIPAWN_EVAL,
                MAX_CENTIPAWN_EVAL,
                4,
                &cancel,
                &mut state
            ),
            eval
        );
    }

    #[test]
    fn capture_search_covers_see_and_futility() {
        let mut board = BoardState::parse_fen("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1");
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let score = search(
            &mut board,
            -MAX_CENTIPAWN_EVAL,
            MAX_CENTIPAWN_EVAL,
            2,
            &cancel,
            &mut state,
        );
        assert!(score.abs() < MAX_CENTIPAWN_EVAL);
        let tight = search(&mut board, 900, 901, 2, &cancel, &mut state);
        assert!(tight.abs() < MAX_CENTIPAWN_EVAL);
    }

    #[test]
    fn quiet_promotions_are_rescued() {
        let mut board = BoardState::parse_fen("7k/5P2/5K2/8/8/8/8/8 w - - 0 1");
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let eval = evaluate_with_optimism(&mut board, 0);
        let score = search(
            &mut board,
            -MAX_CENTIPAWN_EVAL,
            MAX_CENTIPAWN_EVAL,
            2,
            &cancel,
            &mut state,
        );
        assert!(score >= eval);
    }

    #[test]
    fn check_evasion_searches_all_moves() {
        let mut board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        assert!(board.is_in_check(board.side_to_move));
        assert!(has_legal_move(&board));
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let score = search(
            &mut board,
            -MAX_CENTIPAWN_EVAL,
            MAX_CENTIPAWN_EVAL,
            2,
            &cancel,
            &mut state,
        );
        assert!(score.abs() < MAX_CENTIPAWN_EVAL);
    }

    #[test]
    fn stand_pat_fail_high_stores_beta_bound() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let score = search(&mut board, -2000, -1999, 2, &cancel, &mut state);
        assert!(score >= -1999);
        assert!(score.abs() < MAX_CENTIPAWN_EVAL);
        let entry = state.tt.probe(board.board_hash).expect("stand-pat stores");
        assert_eq!(entry.entry_type, TranspositionEntryType::Beta);
    }

    #[test]
    fn promo_rescue_fail_high_above_stand_pat() {
        let mut board = BoardState::parse_fen("7k/5P2/5K2/8/8/8/8/8 w - - 0 1");
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let eval = evaluate_with_optimism(&mut board, 0);
        let alpha = eval.saturating_add(99);
        let beta = eval.saturating_add(100);
        let score = search(&mut board, alpha, beta, 2, &cancel, &mut state);
        assert!(score >= beta);
    }

    #[test]
    fn capture_fail_high_blends_toward_beta() {
        let mut board = BoardState::parse_fen("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1");
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let score = search(&mut board, 0, 1, 2, &cancel, &mut state);
        assert!(score >= 1);
        assert!(score.abs() < MAX_CENTIPAWN_EVAL);
    }

    #[test]
    fn losing_captures_hit_see_prune_in_tight_window() {
        let mut board = BoardState::parse_fen("7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1");
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let score = search(&mut board, 700, 701, 2, &cancel, &mut state);
        assert!(score.abs() < MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }

    #[test]
    fn alpha_bound_returned_when_nothing_improves() {
        let mut board = BoardState::parse_fen(QUIET_ONLY);
        let mut state = SearchState::new();
        let cancel = AtomicBool::new(false);
        let eval = evaluate_with_optimism(&mut board, 0);
        let alpha = eval.saturating_add(100);
        let beta = eval.saturating_add(101);
        let score = search(&mut board, alpha, beta, 2, &cancel, &mut state);
        assert_eq!(score, alpha);
    }
}
