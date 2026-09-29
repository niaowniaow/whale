use super::aspiration::run_aspiration;
use super::diagnostics::update_diagnostics;
use super::reporting::{
    conclude_search, report_diagnostics, report_iteration, run_multipv, run_verification,
};
use super::timing::should_stop;
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn search_primary(
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
    let mut completed_depth = 0u8;
    let prev_score = search_state.best_previous_score.unwrap_or(0);
    let mut iter_scores = [prev_score; 4];
    let mut iter_idx = 0usize;
    let mut score_avg = crate::eval::optimism::ScoreAverage::new();
    score_avg.push(prev_score as i32);

    let timer = Instant::now();

    let mut root_moves = MoveList::new();
    board_state.generate_moves(&mut root_moves);
    let root_threats = NodeThreats::compute(board_state);
    for entry in root_moves.iter() {
        if board_state.is_legal_with(entry.mv, root_threats.checkers, root_threats.pinned) {
            best_move_so_far = entry.mv;
            search_state.best_move = entry.mv;
            break;
        }
    }

    for current_depth in 1..=max_depth {
        let iter_start_nodes = search_state.nodes;

        if search_state.max_nodes > 0
            && search_state.nodes + worker_nodes.load(Ordering::Relaxed) >= search_state.max_nodes
        {
            cancellation_token.store(true, Ordering::Relaxed);
            break;
        }

        let us = board_state.side_to_move;
        let pair = score_avg.pair();
        search_state.optimism[us as usize] = pair[0];
        search_state.optimism[us.other() as usize] = pair[1];

        let (current_score, completed) = run_aspiration(
            board_state,
            current_depth,
            last_score,
            &previous_pv,
            cancellation_token,
            &mut pv_table,
            search_state,
        );

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        let mut aprm_boost = 1.0f64;
        let mut aprm_state_txt = "improve";
        let mut aprm_cpi_us = 0i32;
        let mut aprm_cpi_opp = 0i32;
        let mut aprm_free_us = 0u32;
        let mut aprm_free_opp = 0u32;
        let mut aprm_breaks_opp = 0u32;
        let mut aprm_momentum = 0i16;
        let mut aprm_concession_txt = String::new();
        let mut aprm_musttry = false;
        let mut aprm_move_changed = false;
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
            aprm_move_changed = move_changed;
            if new_best_move != Move::NO_MOVE {
                best_move_so_far = new_best_move;
                search_state.ponder_move = current_pv.get(1).copied().unwrap_or(Move::NO_MOVE);
            }

            let prev_iter_score = last_score;
            last_score = current_score;
            score_avg.push(current_score as i32);
            search_state.score = current_score;
            completed_depth = current_depth;
            if !current_pv.is_empty() {
                previous_pv = current_pv;
            }
            search_state.best_move = best_move_so_far;

            iter_scores[iter_idx] = current_score;
            iter_idx = (iter_idx + 1) & 3;

            let diag = update_diagnostics(
                board_state,
                current_score,
                prev_iter_score,
                best_move_so_far,
                move_changed,
                current_depth,
                search_state,
            );
            aprm_boost = diag.boost;
            aprm_state_txt = diag.state_txt;
            aprm_cpi_us = diag.cpi_us;
            aprm_cpi_opp = diag.cpi_opp;
            aprm_free_us = diag.free_us;
            aprm_free_opp = diag.free_opp;
            aprm_breaks_opp = diag.breaks_opp;
            aprm_momentum = diag.momentum;
            aprm_concession_txt = diag.concession.clone();
            aprm_musttry = diag.musttry;
        }

        if use_tm
            && should_stop(
                &timer,
                search_state,
                cancellation_token,
                worker_nodes,
                iter_start_nodes,
                prev_score,
                current_score,
                iter_scores,
                iter_idx,
                last_best_move_depth,
                current_depth,
                legal_root_moves,
                aprm_boost,
            )
        {
            break;
        }

        report_iteration(
            *debug_mode,
            &previous_pv,
            current_depth,
            search_state,
            &timer,
            worker_nodes,
        );

        if completed && best_move_so_far != Move::NO_MOVE {
            let verified_txt = run_verification(
                board_state,
                current_depth,
                current_score,
                best_move_so_far,
                aprm_musttry,
                aprm_move_changed,
                cancellation_token,
                search_state,
            );
            run_multipv(
                board_state,
                current_depth,
                current_score,
                best_move_so_far,
                previous_pv.clone(),
                cancellation_token,
                search_state,
                &timer,
                worker_nodes,
            );
            report_diagnostics(
                *debug_mode,
                board_state,
                current_depth,
                current_score,
                verified_txt,
                aprm_state_txt,
                aprm_cpi_us,
                aprm_cpi_opp,
                aprm_free_us,
                aprm_free_opp,
                aprm_breaks_opp,
                aprm_momentum,
                &aprm_concession_txt,
                aprm_musttry,
                search_state,
            );
        }
    }
    if search_state.best_move == Move::NO_MOVE {
        search_state.best_move = best_move_so_far;
    }
    conclude_search(board_state, search_state, completed_depth);
}
