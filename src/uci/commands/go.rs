use crate::common::constants;
use crate::common::moves::Move;
use crate::common::side::Side;
use crate::uci::{SEARCH_STATE, UciClient, get_parameter, output_best_move, time_management};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

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

        let ply = {
            let board = self.board.lock().unwrap();
            board.move_count
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
        let (opt_time_tmp, max_time_tmp) = if movetime == -1 && clock != -1 {
            time_management::calculate_optimum_with_ply(clock, increment, movestogo, ply)
        } else {
            (allotted_time, allotted_time)
        };

        let board_snapshot = self.board.lock().unwrap().clone();
        let debug = Arc::clone(&self.debug_mode);
        let cancel_for_search = Arc::clone(&cancel_token);
        let search_state = Arc::clone(&self.search_state);

        let is_movetime = movetime != -1;
        let opt_time = if is_movetime {
            (allotted_time * 9) / 10
        } else if allotted_time == -1 {
            -1
        } else {
            opt_time_tmp
        };
        let max_time = if is_movetime {
            (allotted_time as u64).saturating_sub(10)
        } else if allotted_time == -1 {
            u64::MAX
        } else {
            max_time_tmp as u64
        };

        if allotted_time != -1 {
            let cancel_for_timer = std::sync::Arc::clone(&cancel_token);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(max_time));
                cancel_for_timer.store(true, std::sync::atomic::Ordering::Relaxed);
            });
        }

        let search_depth = if infinite {
            constants::MAX_SEARCH_DEPTH
        } else if has_depth || allotted_time == -1 {
            depth
        } else {
            constants::MAX_SEARCH_DEPTH
        };

        thread::spawn(move || {
            let mut board = board_snapshot;
            let mut debug_mode = debug.load(Ordering::Relaxed);
            let mut search_state_guard = search_state.lock().unwrap();
            search_state_guard.opt_time = opt_time;
            search_state_guard.max_time = max_time.min(i32::MAX as u64) as i32;
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
