use crate::common::constants::MAX_PLY;
use crate::common::moves::Move;
use crate::eval::evaluate;
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

    if ply as usize >= MAX_PLY {
        return evaluate(&mut *board_state);
    }

    search_state.nodes += 1;

    let in_check = board_state.is_in_check(board_state.side_to_move);

    if !in_check {
        let eval = evaluate(&mut *board_state);
        if eval >= beta {
            return beta;
        }
        if eval > alpha {
            alpha = eval;
        }
        // DELTA PRUNING:
        // If standing pat + Queen value cannot exceed alpha, normal captures cannot raise alpha.
        const QUEEN_DELTA: i16 = 1000;
        if eval + QUEEN_DELTA < alpha {
            return alpha;
        }
    }

    let mut move_picker = if !in_check {
        MovePicker::new_qsearch(ply as usize)
    } else {
        MovePicker::new(None, None, None, None, ply as usize, None)
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

        board_state.make_move(move_obj);
        if board_state.is_in_check(board_state.side_to_move.other()) {
            board_state.unmake_move(move_obj);
            continue;
        }

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
            board_state.make_move(move_obj);
            if board_state.is_in_check(board_state.side_to_move.other()) {
                board_state.unmake_move(move_obj);
                continue;
            }
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

    alpha
}
