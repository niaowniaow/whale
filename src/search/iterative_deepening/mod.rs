use crate::board::node_threats::NodeThreats;
use crate::board::plans;
use crate::board::state::BoardState;
use crate::common::constants::{
    ASPIRATION_WINDOW_MARGIN, MAX_CENTIPAWN_EVAL, MAX_PLY, SEARCH_THREAD_STACK_SIZE,
};
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::side::Side;
use crate::search::attack;
use crate::search::concession;
use crate::search::conversion;
use crate::search::counterplay;
use crate::search::multipv::MultipvLine;
use crate::search::negamax;
use crate::search::position_state;
use crate::search::pressure;
use crate::search::pv_table::PvTable;
use crate::search::risk;
use crate::search::search_state::SearchState;
use crate::search::split_root;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

pub mod aspiration;
pub mod core;
pub mod diagnostics;
pub mod reporting;
pub mod roots;
#[cfg(test)]
mod tests;
pub mod timing;
pub mod workers;

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

    search_state.engine_side = board_state.side_to_move;

    if search_state.params.split_root_enabled && num_threads > 1 {
        workers::search_split_root(
            board_state,
            depth,
            cancellation_token,
            debug_mode,
            search_state,
            num_threads,
        );
        return;
    }

    let use_tm = search_state.opt_time > 0;

    let mut legal_root_moves = 0;
    {
        let mut root_moves = MoveList::new();
        board_state.generate_moves(&mut root_moves);

        let root_threats = NodeThreats::compute(board_state);
        let filter = !search_state.searchmoves.is_empty();
        for i in 0..root_moves.len() {
            let m = root_moves[i].mv;
            if filter && !search_state.searchmoves.contains(&m) {
                continue;
            }
            if board_state.is_legal_with(m, root_threats.checkers, root_threats.pinned) {
                legal_root_moves += 1;
                if legal_root_moves > 1 {
                    break;
                }
            }
        }

        if filter && legal_root_moves == 0 {
            for i in 0..root_moves.len() {
                let m = root_moves[i].mv;
                if board_state.is_legal_with(m, root_threats.checkers, root_threats.pinned) {
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
                std::thread::Builder::new()
                    .stack_size(SEARCH_THREAD_STACK_SIZE)
                    .spawn_scoped(s, move || {
                        let (nodes, tbhits, seldepth) = workers::worker_search(
                            worker_board,
                            max_depth,
                            cancellation_token,
                            worker_search_state,
                            thread_id,
                        );
                        worker_nodes_ref.fetch_add(nodes, Ordering::Relaxed);
                        worker_tbhits_ref.fetch_add(tbhits, Ordering::Relaxed);
                        worker_seldepth_ref.fetch_max(seldepth, Ordering::Relaxed);
                    })
                    .unwrap();
            }

            core::search_primary(
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
        core::search_primary(
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
