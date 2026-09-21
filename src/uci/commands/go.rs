use crate::common::constants;
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::side::Side;
use crate::uci::{SEARCH_STATE, UciClient, cli, get_parameter, get_u64, output_best_move};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

impl UciClient {
    pub(crate) fn run_go(&mut self, parameters: &[&str]) {
        if self.search_state.try_lock().is_err() {
            cli::write_line("info string busy");
            return;
        }
        if let Some(cancel) = &self.current_search {
            cancel.store(true, Ordering::Relaxed);
        }
        self.precompute_cancel.store(true, Ordering::Relaxed);
        self.precompute_cancel = Arc::new(AtomicBool::new(false));

        let cancel_token = Arc::new(AtomicBool::new(false));
        self.current_search = Some(Arc::clone(&cancel_token));

        {
            let mut state = SEARCH_STATE.lock().unwrap();
            state.best_move = Move::NO_MOVE;
            state.ponder_move = Move::NO_MOVE;
        }

        let start_time = Instant::now();

        let is_ponder = parameters.contains(&"ponder");
        self.is_pondering.store(is_ponder, Ordering::Relaxed);

        let has_depth = parameters.contains(&"depth");
        let depth = get_parameter("depth", parameters, 8)
            .clamp(1, constants::MAX_SEARCH_DEPTH as i32) as u8;

        let wtime = get_u64("wtime", parameters);
        let btime = get_u64("btime", parameters);
        let winc = get_u64("winc", parameters).unwrap_or(0);
        let binc = get_u64("binc", parameters).unwrap_or(0);
        let movetime = get_u64("movetime", parameters);
        let nodes = get_u64("nodes", parameters).unwrap_or(0);
        let mate = get_u64("mate", parameters).unwrap_or(0);
        let movestogo = get_parameter("movestogo", parameters, -1);
        let infinite = parameters.contains(&"infinite");
        let ponder_option = self.ponder_enabled;

        let searchmoves: Vec<Move> = parameters
            .iter()
            .position(|&t| t == "searchmoves")
            .map(|idx| {
                let wanted: Vec<Move> = parameters[idx + 1..]
                    .iter()
                    .filter_map(|s| Move::parse_long_algebraic(s))
                    .collect();
                if wanted.is_empty() {
                    return Vec::new();
                }
                let board = self.board.lock().unwrap();
                let mut generated = MoveList::new();
                board.generate_moves(&mut generated);
                wanted
                    .into_iter()
                    .filter_map(|w| {
                        generated
                            .iter()
                            .map(|e| e.mv)
                            .find(|m| *m == w)
                            .or_else(|| {
                                generated
                                    .iter()
                                    .map(|e| e.mv)
                                    .find(|m| m.source == w.source && m.target == w.target)
                            })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let (clock_opt, increment, opp_clock_opt) = {
            let board = self.board.lock().unwrap();
            if board.side_to_move == Side::White {
                (wtime, winc, btime)
            } else {
                (btime, binc, wtime)
            }
        };

        let ply = {
            let board = self.board.lock().unwrap();
            board.move_count
        };

        {
            let mut guard = self.search_state.lock().unwrap();
            guard.max_nodes = nodes;
            guard.mate_in = mate.min(u8::MAX as u64) as u8;
            guard.searchmoves = searchmoves;
        }

        let (mut opt_ms, mut max_ms): (i64, u64) = match (movetime, clock_opt) {
            (Some(mt), _) => {
                let t = (mt as i64 - self.move_overhead as i64).max(1);
                (t, t.max(1) as u64)
            }
            (None, Some(clock)) => {
                let clock_i = clock.min(i32::MAX as u64) as i32;
                let inc_i = increment.min(i32::MAX as u64) as i32;
                let opp_i = opp_clock_opt
                    .map(|o| o.min(i32::MAX as u64) as i32)
                    .unwrap_or(-1);
                let (opt, max) = crate::uci::time_management::calculate_optimum_with_ply(
                    clock_i,
                    inc_i,
                    movestogo,
                    ply,
                    opp_i,
                    self.move_overhead,
                    ponder_option,
                );
                (opt as i64, max.max(1) as u64)
            }
            (None, None) => (-1, u64::MAX),
        };

        if self.max_move_time > 0 && max_ms != u64::MAX {
            max_ms = max_ms.min(self.max_move_time as u64);
            opt_ms = opt_ms.min(self.max_move_time as i64);
        }

        let (inner_opt, inner_max): (i32, i32) = if is_ponder || max_ms == u64::MAX {
            (-1, -1)
        } else {
            (
                opt_ms.clamp(1, i32::MAX as i64) as i32,
                max_ms.min(i32::MAX as u64) as i32,
            )
        };

        if max_ms != u64::MAX {
            let cancel_for_timer = Arc::clone(&cancel_token);
            let is_pondering_timer = Arc::clone(&self.is_pondering);
            thread::spawn(move || {
                while is_pondering_timer.load(Ordering::Relaxed) {
                    if cancel_for_timer.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(std::time::Duration::from_millis(5));
                }

                let elapsed_ms = start_time.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
                if max_ms > elapsed_ms {
                    thread::sleep(std::time::Duration::from_millis(max_ms - elapsed_ms));
                }
                cancel_for_timer.store(true, Ordering::Relaxed);
            });
        }

        let board_snapshot = self.board.lock().unwrap().clone();
        let debug = Arc::clone(&self.debug_mode);
        let cancel_for_search = Arc::clone(&cancel_token);
        let search_state = Arc::clone(&self.search_state);
        let is_pondering_search = Arc::clone(&self.is_pondering);

        let search_depth = if infinite {
            constants::MAX_SEARCH_DEPTH
        } else if has_depth || max_ms == u64::MAX {
            depth
        } else {
            constants::MAX_SEARCH_DEPTH
        };

        let num_threads = self.num_threads;
        let precompute_cancel = Arc::clone(&self.precompute_cancel);
        let search_state_for_precompute = Arc::clone(&search_state);

        thread::spawn(move || {
            let mut board = board_snapshot;
            let mut debug_mode = debug.load(Ordering::Relaxed);
            let mut search_state_guard = search_state.lock().unwrap();
            search_state_guard.opt_time = inner_opt;
            search_state_guard.max_time = inner_max;
            let best_move = board.find_best_move(
                search_depth,
                &cancel_for_search,
                &mut debug_mode,
                &mut search_state_guard,
                num_threads,
            );
            debug.store(debug_mode, Ordering::Relaxed);

            let ponder_move = search_state_guard.ponder_move;

            {
                let mut state = SEARCH_STATE.lock().unwrap();
                state.best_move = best_move;
                state.ponder_move = ponder_move;
            }

            while is_pondering_search.load(Ordering::Relaxed) {
                if cancel_for_search.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }

            output_best_move(best_move, ponder_move);
            drop(search_state_guard);

            if best_move != Move::NO_MOVE
                && !is_pondering_search.load(Ordering::Relaxed)
                && ponder_option
            {
                crate::search::smp_precompute::run_precomputation(
                    board,
                    best_move,
                    search_state_for_precompute,
                    precompute_cancel,
                    6,
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    static GO_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn wait_for_best_move(timeout: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < timeout {
            if SEARCH_STATE.lock().unwrap().best_move != Move::NO_MOVE {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    #[test]
    fn go_reports_busy_while_search_state_locked() {
        let mut client = UciClient::new();
        let search_state = std::sync::Arc::clone(&client.search_state);
        let _guard = search_state.lock().unwrap();
        client.run_go(&["depth", "1"]);
        assert!(client.current_search.is_none());
    }

    #[test]
    fn go_depth_one_finds_best_move() {
        let _serial = GO_TEST_LOCK.lock().unwrap();
        let mut client = UciClient::new();
        client.run_go(&["depth", "1"]);
        assert!(client.current_search.is_some());
        assert!(wait_for_best_move(Duration::from_secs(60)));
        client.run_stop(&[]);
    }

    #[test]
    fn go_searchmoves_resolves_to_generated_moves() {
        use crate::common::square::Square;

        let _serial = GO_TEST_LOCK.lock().unwrap();
        let mut client = UciClient::new();

        client.run_go(&["depth", "1", "searchmoves", "e2e4", "xxxx", "e7e5"]);
        {
            let guard = client.search_state.lock().unwrap();
            assert_eq!(guard.searchmoves.len(), 1);
            assert_eq!(guard.searchmoves[0].source, Square::E2);
            assert_eq!(guard.searchmoves[0].target, Square::E4);
        }
        assert!(wait_for_best_move(Duration::from_secs(60)));
        let best = SEARCH_STATE.lock().unwrap().best_move;
        assert_eq!(best.source, Square::E2);
        assert_eq!(best.target, Square::E4);
        client.run_stop(&[]);
    }

    #[test]
    fn go_movetime_timer_cancels_search() {
        let _serial = GO_TEST_LOCK.lock().unwrap();
        let mut client = UciClient::new();
        client.run_go(&["movetime", "50"]);
        let cancel = client.current_search.clone().expect("search token");
        let start = Instant::now();
        while !cancel.load(Ordering::Relaxed) {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "movetime timer did not fire"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        client.run_stop(&[]);
    }
}
