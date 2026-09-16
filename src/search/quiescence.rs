use crate::common::constants::MAX_PLY;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::tt::{self, TranspositionEntryType};
use crate::eval::{evaluate_fast, evaluate_with_optimism};
use crate::search::move_picker::MovePicker;
use crate::search::search_state::SearchState;
use crate::{board::state::BoardState, common::constants::MAX_CENTIPAWN_EVAL};
use std::sync::atomic::{AtomicBool, Ordering};

pub fn search(
    board_state: &mut BoardState,
    mut alpha: i16,
    beta: i16,
    ply: u8,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
) -> i16 {
    if cancellation_token.load(Ordering::Relaxed) {
        return 0;
    }

    if board_state.is_draw_in_search(ply as u16) {
        return 0;
    }

    let optimism = search_state.optimism[board_state.side_to_move as usize];

    if ply as usize >= MAX_PLY {
        return evaluate_with_optimism(&mut *board_state, optimism);
    }

    search_state.nodes += 1;

    let original_alpha = alpha;
    let tt_entry = search_state.tt.probe(board_state.board_hash);
    if let Some(entry) = tt_entry {
        let tt_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
        let cutoff = match entry.entry_type {
            TranspositionEntryType::Exact => true,
            TranspositionEntryType::Alpha => tt_score <= alpha,
            TranspositionEntryType::Beta => tt_score >= beta,
            TranspositionEntryType::None => false,
        };
        if cutoff {
            search_state.nodes += 1;
            return tt_score;
        }
    }

    let in_check = board_state.is_in_check(board_state.side_to_move);
    let mut futility_base = -MAX_CENTIPAWN_EVAL;

    if !in_check {
        let fast_eval = evaluate_fast(&mut *board_state, optimism);
        if fast_eval >= beta + 200 {
            return beta;
        }
        if fast_eval + 1200 < alpha {
            return alpha;
        }

        let eval = if let Some(entry) = tt_entry {
            let s = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
            match entry.entry_type {
                TranspositionEntryType::Exact => s,
                TranspositionEntryType::Beta if s >= beta => s,
                _ => evaluate_with_optimism(&mut *board_state, optimism),
            }
        } else {
            evaluate_with_optimism(&mut *board_state, optimism)
        };
        if eval >= beta {
            if eval.abs() < MAX_CENTIPAWN_EVAL - 200 {
                return ((441 * eval as i32 + 583 * beta as i32) / 1024) as i16;
            }
            return beta;
        }
        if eval > alpha {
            alpha = eval;
        }
        const QUEEN_DELTA: i16 = 1000;
        if eval + QUEEN_DELTA < alpha {
            return alpha;
        }
        futility_base = eval + 306;
    }

    let mut move_picker = if !in_check {
        MovePicker::new_qsearch(ply as usize)
    } else {
        MovePicker::new(None, None, None, ply as usize, None)
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
                let futility_val = futility_base + captured_piece.see_value();
                if futility_val <= alpha {
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

    let entry_type = if alpha >= beta {
        TranspositionEntryType::Beta
    } else if alpha > original_alpha {
        TranspositionEntryType::Exact
    } else {
        TranspositionEntryType::Alpha
    };
    search_state.tt.submit_entry(
        board_state.board_hash,
        tt::TranspositionTable::adjust_score(alpha, ply as i32),
        0,
        Move::NO_MOVE,
        entry_type,
    );

    alpha
}
