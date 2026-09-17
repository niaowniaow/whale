use crate::common::constants::MAX_PLY;
use crate::common::move_list::MoveList;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::tt::{self, TranspositionEntryType};
use crate::eval::evaluate_with_optimism;
use crate::search::bmo::BanditArm;
use crate::search::move_picker::MovePicker;
use crate::search::search_state::SearchState;
use crate::{board::state::BoardState, common::constants::MAX_CENTIPAWN_EVAL};
use std::sync::atomic::{AtomicBool, Ordering};

fn has_legal_move(board_state: &BoardState) -> bool {
    let mut moves = MoveList::new();
    board_state.generate_quiets(&mut moves);
    if moves.iter().any(|entry| board_state.is_legal(entry.mv)) {
        return true;
    }
    board_state.generate_captures(&mut moves);
    moves.iter().any(|entry| board_state.is_legal(entry.mv))
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
        return 0;
    }

    let in_check = board_state.is_in_check(board_state.side_to_move);
    if !has_legal_move(board_state) {
        return if in_check {
            -MAX_CENTIPAWN_EVAL + ply as i16
        } else {
            0
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
        let continue_qs =
            crate::search::lqt::should_continue_quiescence(board_state, eval, alpha, beta, ply);
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

    while let Some(move_obj) = move_picker.next(
        board_state,
        &search_state.move_ordering,
        &mut search_state.captures_stack[ply as usize],
        &mut search_state.quiets_stack[ply as usize],
    ) {
        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if !in_check {
            if board_state.see(move_obj) < -74 {
                continue;
            }

            if !move_obj.is_promotion() {
                let captured_piece = if move_obj.move_type == MoveType::EnPassant {
                    Piece::Pawn
                } else {
                    board_state.piece_mapping[move_obj.target as usize]
                };
                // FIX QSEARCH-LERP: futility updates bestValue instead of
                // silently skipping (Stockfish search.cpp:1815-1818).
                let futility_val = futility_base + captured_piece.see_value();
                if futility_val <= alpha {
                    best_value = best_value.max(futility_val);
                    continue;
                }
            }
        }

        if !board_state.is_legal(move_obj) {
            continue;
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
            if !board_state.is_legal(move_obj) {
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
            assert_eq!(search(&mut board, alpha, beta, 4, &cancel, &mut state), 0);
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
                assert_eq!(search(&mut board, alpha, beta, 4, &cancel, &mut state), 0);
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
                assert_eq!(
                    search(&mut board, 0, 1, ply, &cancel, &mut state),
                    if in_check {
                        -MAX_CENTIPAWN_EVAL + ply as i16
                    } else {
                        0
                    },
                );
            }
        }
    }
}
