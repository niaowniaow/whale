use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn should_stop(
    timer: &Instant,
    search_state: &mut SearchState,
    cancellation_token: &AtomicBool,
    worker_nodes: &AtomicU64,
    iter_start_nodes: u64,
    prev_score: i16,
    current_score: i16,
    iter_scores: [i16; 4],
    iter_idx: usize,
    last_best_move_depth: u8,
    current_depth: u8,
    legal_root_moves: i32,
    aprm_boost: f64,
) -> bool {
    let total_nodes_now = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
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
    let falling_eval = ((11.48 + 2.30 * prev_diff + 1.10 * iter_diff) / 100.0).clamp(0.70, 1.40);

    let stability_depth = current_depth.saturating_sub(last_best_move_depth) as f64;
    let t = (stability_depth - 4.96) / (18.79 - 4.96);
    let time_reduction = (0.639 + (1.712 - 0.639) * t).clamp(0.629, 1.544);
    let reduction = (1.468 + search_state.previous_time_reduction) / (2.284 * time_reduction);
    search_state.previous_time_reduction = time_reduction;

    let tot_best_move_changes = search_state.best_move_changes;
    search_state.best_move_changes = 0;
    let best_move_instability = (1.0 + 0.35 * (tot_best_move_changes as f64)).min(1.6);

    let disagreement = (prev_score as i32 - current_score as i32).abs();
    let dad_time_factor = if current_depth >= 6 && search_state.params.dad_enabled {
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
        * dad_time_factor
        * aprm_boost;

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

    let nodes_exceeded = search_state.max_nodes > 0 && total_nodes_now >= search_state.max_nodes;

    if search_state.max_time > 0 && search_state.opt_time == search_state.max_time {
        let fixed_stop = (search_state.max_time as f64) * 0.85;
        if elapsed >= fixed_stop || is_mate || nodes_exceeded {
            cancellation_token.store(true, Ordering::Relaxed);
            return true;
        }
        return false;
    }

    if elapsed > stop_time || is_mate || elapsed > total_time * 0.60 || nodes_exceeded {
        cancellation_token.store(true, Ordering::Relaxed);
        return true;
    }
    false
}
