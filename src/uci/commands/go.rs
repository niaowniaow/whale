use crate::common::constants;
use crate::common::moves::Move;
use crate::common::side::Side;
use crate::uci::{SEARCH_STATE, UciClient, get_parameter, output_best_move, time_management};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

impl UciClient {
    pub(crate) fn run_go(&mut self, parameters: &[&str]) {
        if let Some(cancel) = &self.current_search {
            cancel.store(true, Ordering::Relaxed);
        }

        let cancel_token = Arc::new(AtomicBool::new(false));
        self.current_search = Some(Arc::clone(&cancel_token));

        {
            let mut state = SEARCH_STATE.lock().unwrap();
            state.best_move = Move::NO_MOVE;
        }

        let has_depth = parameters.contains(&"depth");
        let depth = get_parameter("depth", parameters, 8)
            .clamp(1, constants::MAX_SEARCH_DEPTH as i32) as u8;
        let winc = get_parameter("winc", parameters, 0);
        let binc = get_parameter("binc", parameters, 0);
        let wtime = get_parameter("wtime", parameters, -1);
        let btime = get_parameter("btime", parameters, -1);
        let movetime = get_parameter("movetime", parameters, -1);
        let movestogo = get_parameter("movestogo", parameters, -1);
        let infinite = parameters.contains(&"infinite");

        let (clock, increment) = {
            let board = self.board.lock().unwrap();
            if board.side_to_move == Side::White {
                (wtime, winc)
            } else {
                (btime, binc)
            }
        };

        let allotted_time = if movetime == -1 {
            if clock == -1 {
                -1
            } else {
                time_management::calculate_move_time_with_moves(clock, increment, movestogo)
            }
        } else {
            movetime
        };

        let board_snapshot = self.board.lock().unwrap().clone();
        let debug = Arc::clone(&self.debug_mode);
        let cancel_for_search = Arc::clone(&cancel_token);
        let search_state = Arc::clone(&self.search_state);

        if allotted_time != -1 {
            let cancel_for_timer = std::sync::Arc::clone(&cancel_token);
            // max_time is 150% of opt_time for stability extensions
            let max_time = (allotted_time as f32 * 1.5) as u64;
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(max_time));
                cancel_for_timer.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }

        let search_depth = if infinite {
            constants::MAX_SEARCH_DEPTH
        } else if has_depth {
            depth
        } else if allotted_time == -1 {
            depth
        } else {
            constants::MAX_SEARCH_DEPTH
        };

        thread::spawn(move || {
            let mut board = board_snapshot;
            let mut debug_mode = debug.load(Ordering::Relaxed);
            let mut search_state_guard = search_state.lock().unwrap();
            search_state_guard.opt_time = allotted_time;
            let best_move = board.find_best_move(
                search_depth,
                &cancel_for_search,
                &mut debug_mode,
                &mut search_state_guard,
            );
            debug.store(debug_mode, Ordering::Relaxed);

            {
                let mut state = SEARCH_STATE.lock().unwrap();
                state.best_move = best_move;
            }

            output_best_move(best_move);
        });
    }
}
