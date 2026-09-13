use crate::board::state::BoardState;
use crate::common::constants::{ASPIRATION_WINDOW_MARGIN, MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::moves::Move;
use crate::search::negamax;
use crate::search::pv_table::PvTable;
use crate::search::search_state::SearchState;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

pub fn search(
    board_state: &mut BoardState,
    depth: u8,
    cancellation_token: &AtomicBool,
    debug_mode: &mut bool,
    search_state: &mut SearchState,
) {
    search_state.reset_search();

    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    let mut last_score: i16 = 0;
    let mut best_move_so_far = Move::NO_MOVE;
    let mut bm_changes = 0;

    let timer = Instant::now();

    for current_depth in 1..=depth {
        // Aspiration Windows
        let mut alpha = i16::MIN + 1;
        let mut beta = i16::MAX - 1;

        if current_depth > 1 {
            if last_score.abs() as i32 > MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32 {
                alpha = i16::MIN + 1;
                beta = i16::MAX - 1;
            } else {
                alpha = last_score
                    .saturating_sub(ASPIRATION_WINDOW_MARGIN)
                    .max(i16::MIN + 1);
                beta = last_score
                    .saturating_add(ASPIRATION_WINDOW_MARGIN)
                    .min(i16::MAX - 1);
            }
        }

        let mut current_score = last_score;
        let mut completed = false;
        let mut delta = ASPIRATION_WINDOW_MARGIN;

        loop {
            let score = negamax::search(
                board_state,
                current_depth,
                alpha,
                beta,
                cancellation_token,
                &previous_pv,
                &mut pv_table,
                search_state,
            );

            if cancellation_token.load(Ordering::Relaxed) {
                break;
            }

            current_score = score;
            // Geometric widening: each fail strictly expands the window, so
            // this terminates (worst case falls back to a full window).
            if current_score <= alpha {
                if alpha == i16::MIN + 1 {
                    completed = true;
                    break;
                }
                delta = delta.saturating_mul(4).min(1500);
                alpha = current_score.saturating_sub(delta).max(i16::MIN + 1);
            } else if current_score >= beta {
                if beta == i16::MAX - 1 {
                    completed = true;
                    break;
                }
                delta = delta.saturating_mul(4).min(1500);
                beta = current_score.saturating_add(delta).min(i16::MAX - 1);
            } else {
                completed = true;
                break;
            }
        }

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if completed {
            let current_pv = pv_table.line().to_vec();
            let new_best_move = current_pv.first().copied().unwrap_or(Move::NO_MOVE);
            if current_depth > 1
                && best_move_so_far != Move::NO_MOVE
                && new_best_move != best_move_so_far
                && new_best_move != Move::NO_MOVE
            {
                bm_changes += 1;
            }
            // Never throw away a known best move for an empty PV.
            if new_best_move != Move::NO_MOVE {
                best_move_so_far = new_best_move;
            }

            last_score = current_score;
            search_state.score = current_score;
            if !current_pv.is_empty() {
                previous_pv = current_pv;
            }
            search_state.best_move = best_move_so_far;
        }

        // TM: Soft limit with BM instability
        if search_state.opt_time > 0 {
            let elapsed = timer.elapsed().as_millis() as i32;
            // Add up to 50% extra time if best move is unstable
            let dynamic_opt =
                search_state.opt_time + (search_state.opt_time / 4) * bm_changes.min(2);
            if elapsed >= dynamic_opt {
                cancellation_token.store(true, Ordering::Relaxed);
                break;
            }
        }

        let time_ms = timer.elapsed().as_millis().max(1) as f64;
        let nps = (search_state.nodes as f64 / time_ms * 1000.0) as i32;

        if *debug_mode {
            let pv_string = previous_pv
                .iter()
                .map(|m| {
                    let promotion = m
                        .promotion_char()
                        .map(|c| c.to_string())
                        .unwrap_or_else(String::new);
                    format!("{}{}{}", m.source, m.target, promotion)
                })
                .collect::<Vec<String>>()
                .join(" ");
            let score_str = format_score(search_state.score);
            println!(
                "info depth {} score {} nodes {} time {} nps {} pv {}",
                current_depth, score_str, search_state.nodes, time_ms, nps, pv_string
            );
        }
    }
}

pub fn format_score(score: i16) -> String {
    let score_abs = (score as i32).abs();
    if (MAX_CENTIPAWN_EVAL as i32 - score_abs) <= MAX_PLY as i32 {
        let d = crate::common::constants::MAX_CENTIPAWN_EVAL as i32 - score_abs;
        let y = (d + 1) / 2;
        let sign = if score < 0 { -1 } else { 1 };
        format!("mate {}", y * sign)
    } else {
        format!("cp {}", score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_score() {
        assert_eq!(format_score(100), "cp 100");
        assert_eq!(format_score(-500), "cp -500");
        assert_eq!(format_score(MAX_CENTIPAWN_EVAL - 1), "mate 1");
        assert_eq!(format_score(MAX_CENTIPAWN_EVAL - 3), "mate 2");
    }
}
