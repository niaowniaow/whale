use crate::board::state::BoardState;
use crate::common::constants::{ASPIRATION_WINDOW_MARGIN, MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::search::negamax;
use crate::search::pv_table::PvTable;
use crate::search::search_state::SearchState;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

fn worker_search(
    mut board: BoardState,
    max_depth: u8,
    cancellation_token: &AtomicBool,
    mut search_state: SearchState,
    thread_id: usize,
) -> (u64, u64, u32) {
    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    // FIX ASPIRATION: Lazy SMP helpers run full-window searches only (no
    // private aspiration); depth stagger (odd/even) is the sole diversity
    // source so every thread shares identical params/TT view.
    let start_depth = if thread_id % 2 == 1 { 1 } else { 2 };
    for current_depth in start_depth..=max_depth {
        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }
        // FIX LIMIT max_nodes: helpers abort cleanly on the shared token.
        if search_state.max_nodes > 0 && search_state.nodes >= search_state.max_nodes {
            break;
        }

        let score = negamax::search(
            &mut board,
            current_depth,
            i16::MIN + 1,
            i16::MAX - 1,
            cancellation_token,
            &previous_pv,
            &mut pv_table,
            &mut search_state,
        );

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        // Root discards nothing here (helpers only warm the shared TT), but a
        // cancelled score must not poison move ordering.
        if search_state.max_nodes > 0 && search_state.nodes >= search_state.max_nodes {
            break;
        }
        let _ = score;
        let current_pv = pv_table.line().to_vec();
        if !current_pv.is_empty() {
            previous_pv = current_pv;
        }
    }
    (
        search_state.nodes,
        search_state.tbhits,
        search_state.seldepth,
    )
}

#[allow(clippy::too_many_arguments)]
fn search_primary(
    board_state: &mut BoardState,
    _depth: u8,
    cancellation_token: &AtomicBool,
    debug_mode: &mut bool,
    search_state: &mut SearchState,
    worker_nodes: &AtomicU64,
    max_depth: u8,
    legal_root_moves: i32,
    use_tm: bool,
) {
    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    let mut last_score: i16 = 0;
    let mut best_move_so_far = Move::NO_MOVE;
    let mut last_best_move_depth = 1u8;
    let prev_score = search_state.best_previous_score.unwrap_or(0);
    let mut iter_scores = [prev_score; 4];
    let mut iter_idx = 0usize;

    let timer = Instant::now();

    for current_depth in 1..=max_depth {
        let iter_start_nodes = search_state.nodes;

        // FIX LIMIT max_nodes: stop cleanly between iterations (root discards
        // the aborted iteration below via the cancellation check).
        if search_state.max_nodes > 0
            && search_state.nodes + worker_nodes.load(Ordering::Relaxed) >= search_state.max_nodes
        {
            cancellation_token.store(true, Ordering::Relaxed);
            break;
        }

        let us = board_state.side_to_move;
        let avg_sf = (last_score as i32) * 208 / 100;
        let opt = if avg_sf != 0 {
            114 * avg_sf / (avg_sf.abs() + 85)
        } else {
            0
        };
        search_state.optimism[us as usize] = opt;
        search_state.optimism[us.other() as usize] = -opt;

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

        // FIX ASPIRATION: Stockfish search.cpp:376-444 window dynamics. On
        // fail-low the window slides down with beta=alpha overlap; on
        // fail-high alpha tightens to beta-delta overlap; delta grows gently.
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
            if current_score <= alpha {
                if alpha == i16::MIN + 1 {
                    completed = true;
                    break;
                }
                beta = alpha;
                alpha = current_score.saturating_sub(delta).max(i16::MIN + 1);
            } else if current_score >= beta {
                if beta == i16::MAX - 1 {
                    completed = true;
                    break;
                }
                alpha = beta.saturating_sub(delta).max(alpha);
                beta = current_score.saturating_add(delta).min(i16::MAX - 1);
            } else {
                completed = true;
                break;
            }
            delta = delta
                .saturating_add(delta.saturating_mul(47) / 128)
                .min(1500);
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
            if move_changed {
                last_best_move_depth = current_depth;
            }
            if new_best_move != Move::NO_MOVE {
                best_move_so_far = new_best_move;
                search_state.ponder_move = current_pv.get(1).copied().unwrap_or(Move::NO_MOVE);
            }

            last_score = current_score;
            search_state.score = current_score;
            if !current_pv.is_empty() {
                previous_pv = current_pv;
            }
            search_state.best_move = best_move_so_far;

            iter_scores[iter_idx] = current_score;
            iter_idx = (iter_idx + 1) & 3;
        }

        // FIX NODES-SELDEPTH: consistent totals (primary + helpers) for NPS and
        // effort; u64 throughout with i64 casts before signed arithmetic.
        let total_nodes_now = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
        if use_tm {
            let elapsed = timer.elapsed().as_millis() as f64;
            let iter_nodes = search_state.nodes.saturating_sub(iter_start_nodes).max(1);
            let nodes_effort = (search_state.root_best_move_nodes.saturating_mul(100_000)
                / iter_nodes.max(1))
            .clamp(0, 100_000);

            let high_best_move_effort = {
                let t = (nodes_effort as i64 - 75800) as f64 / (104510 - 75800) as f64;
                (0.969 + (0.714 - 0.969) * t).clamp(0.693, 0.838)
            };

            let prev_diff = (prev_score as f64 - current_score as f64) * 2.08;
            let iter_diff = (iter_scores[iter_idx] as f64 - current_score as f64) * 2.08;
            let falling_eval =
                ((11.48 + 2.30 * prev_diff + 1.10 * iter_diff) / 100.0).clamp(0.70, 1.40);

            let stability_depth = current_depth.saturating_sub(last_best_move_depth) as f64;
            let t = (stability_depth - 4.96) / (18.79 - 4.96);
            let time_reduction = (0.639 + (1.712 - 0.639) * t).clamp(0.629, 1.544);
            let reduction =
                (1.468 + search_state.previous_time_reduction) / (2.284 * time_reduction);
            search_state.previous_time_reduction = time_reduction;

            let tot_best_move_changes = search_state.best_move_changes;
            search_state.best_move_changes = 0;
            let best_move_instability = (1.0 + 0.35 * (tot_best_move_changes as f64)).min(1.6);

            let disagreement = (prev_score as i32 - current_score as i32).abs();
            let dad_time_factor = if current_depth >= 6 {
                if disagreement > 60 {
                    1.25
                } else if disagreement < 15 {
                    0.85
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let mut total_time = (search_state.opt_time as f64)
                * falling_eval
                * reduction
                * best_move_instability
                * high_best_move_effort
                * dad_time_factor;

            if legal_root_moves <= 1 {
                total_time = total_time.min(500.0);
            }

            let max_limit = if search_state.max_time > 0 {
                search_state.max_time as f64
            } else {
                (search_state.opt_time as f64) * 2.5
            };

            let stop_time = total_time.min(max_limit);
            let is_mate = current_score.abs() as i32 >= (MAX_CENTIPAWN_EVAL as i32 - 5);
            // FIX LIMIT max_nodes: abort cleanly inside the time check as well.
            let nodes_exceeded =
                search_state.max_nodes > 0 && total_nodes_now >= search_state.max_nodes;

            if elapsed > stop_time || is_mate || elapsed > total_time * 0.60 || nodes_exceeded {
                cancellation_token.store(true, Ordering::Relaxed);
                break;
            }
        }

        let time_ms = timer.elapsed().as_millis().max(1) as f64;
        let total_nodes = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
        let nps = (total_nodes as f64 / time_ms * 1000.0) as i32;

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
                "info depth {} seldepth {} score {} nodes {} tbhits {} time {} nps {} pv {}",
                current_depth,
                search_state.seldepth,
                score_str,
                total_nodes,
                search_state.tbhits,
                time_ms,
                nps,
                pv_string
            );
        }
    }
    search_state.best_previous_score = Some(search_state.score);
}

pub fn search(
    board_state: &mut BoardState,
    depth: u8,
    cancellation_token: &AtomicBool,
    debug_mode: &mut bool,
    search_state: &mut SearchState,
    num_threads: usize,
) {
    search_state.reset_search();
    search_state.tt.new_search();

    let use_tm = search_state.opt_time > 0;

    // FIX LIMIT searchmoves: an empty filter means "all moves" (skip filtering).
    let mut legal_root_moves = 0;
    {
        let mut root_moves = MoveList::new();
        board_state.generate_moves(&mut root_moves);
        let filter = !search_state.searchmoves.is_empty();
        for i in 0..root_moves.len() {
            let m = root_moves[i].mv;
            if filter && !search_state.searchmoves.contains(&m) {
                continue;
            }
            if board_state.is_legal(m) {
                legal_root_moves += 1;
                if legal_root_moves > 1 {
                    break;
                }
            }
        }
        // A non-empty filter with no legal moves must not kill the search.
        if filter && legal_root_moves == 0 {
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
    }
    let mut max_depth = if use_tm && legal_root_moves <= 1 {
        1
    } else {
        depth
    };
    // FIX LIMIT mate_in: cap depth to ~2*plies+1 so mate search terminates.
    if search_state.mate_in > 0 {
        let mate_cap = search_state
            .mate_in
            .saturating_mul(2)
            .saturating_add(1)
            .max(1);
        max_depth = max_depth.min(mate_cap);
    }

    let worker_nodes = AtomicU64::new(0);
    let worker_tbhits = AtomicU64::new(0);
    let worker_seldepth = AtomicU32::new(0);
    let num_workers = num_threads.saturating_sub(1);

    if num_workers > 0 {
        std::thread::scope(|s| {
            for thread_id in 1..=num_workers {
                let worker_board = board_state.clone();
                let worker_search_state = search_state.clone_for_worker(thread_id);
                let worker_nodes_ref = &worker_nodes;
                let worker_tbhits_ref = &worker_tbhits;
                let worker_seldepth_ref = &worker_seldepth;
                s.spawn(move || {
                    let (nodes, tbhits, seldepth) = worker_search(
                        worker_board,
                        max_depth,
                        cancellation_token,
                        worker_search_state,
                        thread_id,
                    );
                    worker_nodes_ref.fetch_add(nodes, Ordering::Relaxed);
                    worker_tbhits_ref.fetch_add(tbhits, Ordering::Relaxed);
                    worker_seldepth_ref.fetch_max(seldepth, Ordering::Relaxed);
                });
            }

            search_primary(
                board_state,
                depth,
                cancellation_token,
                debug_mode,
                search_state,
                &worker_nodes,
                max_depth,
                legal_root_moves,
                use_tm,
            );

            cancellation_token.store(true, Ordering::Relaxed);
        });
    } else {
        search_primary(
            board_state,
            depth,
            cancellation_token,
            debug_mode,
            search_state,
            &worker_nodes,
            max_depth,
            legal_root_moves,
            use_tm,
        );
    }
    // FIX NODES-SELDEPTH: consistent totals include helper threads.
    search_state.nodes += worker_nodes.load(Ordering::Relaxed);
    search_state.tbhits += worker_tbhits.load(Ordering::Relaxed);
    search_state.seldepth = search_state
        .seldepth
        .max(worker_seldepth.load(Ordering::Relaxed));
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

    #[test]
    fn test_smp_search_multi_threaded() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 4, &token, &mut debug, &mut state, 4);
        assert_ne!(state.best_move, Move::NO_MOVE);
    }
}
