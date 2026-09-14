use crate::board::state::BoardState;
use crate::common::constants::{ASPIRATION_WINDOW_MARGIN, MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_list::MoveList;
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
    search_state.tt.new_search();

    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    let mut last_score: i16 = 0;
    let mut best_move_so_far = Move::NO_MOVE;
    let mut bm_changes = 0;
    let mut stable_count = 0;
    let mut last_best_move_depth = 1u8;
    let mut iter_scores = [0i16; 4];
    let mut iter_idx = 0usize;

    // Time management is only active in clock-based searches; fixed-depth
    // searches always run to full depth for predictable testing.
    let use_tm = search_state.opt_time > 0;

    // TM 1 — Easy move: a single legal move needs no deep thought. Still run
    // one shallow iteration so the score/PV stay sane, then stop.
    let mut legal_root_moves = 0;
    {
        let mut root_moves = MoveList::new();
        board_state.generate_moves(&mut root_moves);
        for i in 0..root_moves.len() {
            let m = root_moves[i].mv;
            if board_state.is_legal(m) {
                legal_root_moves += 1;
                if legal_root_moves > 1 {
                    break;
                }
            }
        }
    }
    let max_depth = if use_tm && legal_root_moves <= 1 {
        1
    } else {
        depth
    };

    let timer = Instant::now();

    for current_depth in 1..=max_depth {
        let iter_start_nodes = search_state.nodes;
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
            let move_changed = current_depth > 1
                && best_move_so_far != Move::NO_MOVE
                && new_best_move != Move::NO_MOVE
                && new_best_move != best_move_so_far;
            let score_swing =
                current_depth > 1 && (current_score as i32 - last_score as i32).abs() > 80;
            if move_changed || score_swing {
                bm_changes += 1;
            }
            if move_changed {
                last_best_move_depth = current_depth;
            }
            let stable = use_tm
                && !move_changed
                && new_best_move != Move::NO_MOVE
                && current_depth > 2
                && (current_score as i32 - last_score as i32).abs() <= 12;
            if stable {
                stable_count += 1;
            } else {
                stable_count = 0;
            }
            if new_best_move != Move::NO_MOVE {
                best_move_so_far = new_best_move;
            }

            last_score = current_score;
            search_state.score = current_score;
            if !current_pv.is_empty() {
                previous_pv = current_pv;
            }
            search_state.best_move = best_move_so_far;

            iter_scores[iter_idx] = current_score;
            iter_idx = (iter_idx + 1) & 3;

            if use_tm && stable_count >= 2 && current_depth >= 6 {
                break;
            }
        }

        if use_tm {
            let elapsed = timer.elapsed().as_millis() as i32;
            let iter_nodes = (search_state.nodes - iter_start_nodes).max(1);
            let effort =
                (search_state.root_best_move_nodes as f64 / iter_nodes as f64).clamp(0.0, 1.0);
            let high_effort_discount = if effort >= 0.70 {
                (1.0 - 0.30 * ((effort - 0.70) / 0.30)).clamp(0.70, 1.0)
            } else {
                1.0
            };

            let prev_diff = search_state
                .best_previous_score
                .map(|p| p as f64 - current_score as f64)
                .unwrap_or(0.0);
            let iter_diff = iter_scores[iter_idx] as f64 - current_score as f64;
            let falling_eval = if current_depth > 2 {
                (1.0 + 0.015 * prev_diff + 0.010 * iter_diff).clamp(0.60, 1.75)
            } else {
                1.0
            };

            let best_move_instability = (1.0 + 0.35 * bm_changes as f64).clamp(1.0, 2.0);

            let stability_depth = current_depth.saturating_sub(last_best_move_depth) as f64;
            let stability_discount = if stability_depth >= 3.0 {
                (1.0 - 0.05 * (stability_depth - 3.0)).clamp(0.65, 1.0)
            } else {
                1.0
            };

            let total_scale =
                falling_eval * best_move_instability * stability_discount * high_effort_discount;
            let dynamic_opt = ((search_state.opt_time as f64) * total_scale) as i32;
            let max_limit = if search_state.max_time > 0 {
                search_state.max_time
            } else {
                search_state.opt_time.saturating_mul(5)
            };
            let capped_opt = dynamic_opt.clamp(10, max_limit);

            if elapsed >= capped_opt
                || (elapsed as f64 >= capped_opt as f64 * 0.55 && current_depth >= 6)
            {
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
    search_state.best_previous_score = Some(search_state.score);
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
