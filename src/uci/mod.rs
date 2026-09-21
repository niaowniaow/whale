pub mod cli;
mod commands;
pub mod time_management;

use crate::board::state::BoardState;
use crate::common::helpers::STARTING_FEN;
use crate::common::moves::Move;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, LazyLock, Mutex};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SearchState {
    pub best_move: Move,
    pub ponder_move: Move,
}

pub(crate) static SEARCH_STATE: LazyLock<Mutex<SearchState>> = LazyLock::new(|| {
    Mutex::new(SearchState {
        best_move: Move::NO_MOVE,
        ponder_move: Move::NO_MOVE,
    })
});

pub fn run(_parameters: &[&str]) {
    let mut client = UciClient::new();
    client.write_id();
    client.run();
}

pub(crate) struct UciClient {
    pub board: Arc<Mutex<BoardState>>,
    pub debug_mode: Arc<AtomicBool>,
    pub current_search: Option<Arc<AtomicBool>>,
    pub search_state: Arc<Mutex<crate::search::search_state::SearchState>>,
    pub is_ready: bool,
    pub is_pondering: Arc<AtomicBool>,
    pub ponder_enabled: bool,
    pub precompute_cancel: Arc<AtomicBool>,
    pub num_threads: usize,
    pub move_overhead: i32,
    pub max_move_time: i32,
}

impl UciClient {
    pub(crate) fn new() -> Self {
        let mut search_state = crate::search::search_state::SearchState::new();
        search_state.tt = std::sync::Arc::new(crate::common::tt::TranspositionTable::new_mb(16));
        Self {
            board: Arc::new(Mutex::new(BoardState::parse_fen(STARTING_FEN))),
            debug_mode: Arc::new(AtomicBool::new(true)),
            current_search: None,
            search_state: Arc::new(Mutex::new(search_state)),
            is_ready: false,
            is_pondering: Arc::new(AtomicBool::new(false)),
            ponder_enabled: false,
            precompute_cancel: Arc::new(AtomicBool::new(false)),
            num_threads: 1,
            move_overhead: 10,
            max_move_time: 0,
        }
    }

    fn run(&mut self) {
        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            match stdin.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => continue,
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            let command = parts[0];
            let parameters = &parts[1..];

            if command == "quit" {
                self.run_stop(&[]);
                std::process::exit(0);
            }

            self.handle_command(command, parameters);
        }
    }

    pub(crate) fn handle_command(&mut self, command: &str, parameters: &[&str]) {
        match command {
            "isready" => self.run_isready(parameters),
            "position" => self.run_position(parameters),
            "go" => self.run_go(parameters),
            "stop" => self.run_stop(parameters),
            "ponderhit" => self.run_ponderhit(),
            "ucinewgame" => self.run_ucinewgame(parameters),
            "debug" => self.run_debug(parameters),
            "setoption" => self.run_setoption(parameters),
            "uci" => self.write_id(),
            "perft" => self.run_perft(parameters),
            "bench" => self.run_bench(parameters),
            "model" => self.run_model(parameters),

            _ => eprintln!("Unknown command {command}"),
        }
    }

    pub(crate) fn run_ponderhit(&mut self) {
        self.is_pondering
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn write_id(&self) {
        cli::write_line(&format!("id name Whale {}", env!("CARGO_PKG_VERSION")));
        cli::write_line("id author Vishnu B");
        cli::write_line("option name Hash type spin default 16 min 1 max 2048");
        cli::write_line("option name Threads type spin default 1 min 1 max 256");
        cli::write_line("option name Move Overhead type spin default 10 min 0 max 5000");
        cli::write_line("option name MaxMoveTime type spin default 0 min 0 max 300000");
        cli::write_line("option name Ponder type check default false");
        cli::write_line("option name Clear Hash type button");
        cli::write_line(
            "option name Model type combo default whale_big var whale_big var whale_medium var whale_small var embedded",
        );
        cli::write_line("option name EvalFile type string default <empty>");
        cli::write_line("option name EvalFileSmall type string default <empty>");
        cli::write_line("option name SyzygyPath type string default <empty>");
        cli::write_line("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
        cli::write_line("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
        cli::write_line("option name Syzygy50MoveRule type check default true");
        cli::write_line("option name Contempt type spin default 0 min -200 max 200");
        cli::write_line("option name DrawScore type spin default 0 min -200 max 200");
        cli::write_line("option name ShowWDL type check default true");
        cli::write_line("option name RFP_Margin type spin default 110 min 50 max 300");
        cli::write_line("option name Futility_Margin type spin default 120 min 50 max 300");
        cli::write_line("option name Singular_Margin type spin default 2 min 1 max 5");
        cli::write_line("option name ProbCut_Margin type spin default 170 min 50 max 400");
        cli::write_line("option name NMP_Base type spin default 3 min 1 max 6");
        cli::write_line("option name NMP_Depth_Div type spin default 3 min 2 max 8");
        cli::write_line("option name LMR_Base type spin default 65 min 10 max 150");
        cli::write_line("option name LMR_Div type spin default 215 min 100 max 350");
        cli::write_line("option name History_Weight type spin default 2 min 1 max 4");
        cli::write_line("option name ALP_Enabled type check default true");
        cli::write_line("option name ALP_Threshold type spin default 75 min 50 max 95");
        cli::write_line("option name PSM_Enabled type check default true");
        cli::write_line("option name GTP_Enabled type check default true");
        cli::write_line("option name GTP_Threshold type spin default 15 min 5 max 50");
        cli::write_line("option name CFSS_Enabled type check default true");
        cli::write_line("option name RAS_Enabled type check default true");
        cli::write_line("option name BMO_Enabled type check default true");
        cli::write_line("option name TCE_Enabled type check default true");
        cli::write_line("option name LQT_Enabled type check default true");
        cli::write_line("option name SPS_Enabled type check default true");
        cli::write_line("option name DAD_Enabled type check default true");
        cli::write_line("option name Extension_Cap_Enabled type check default true");
        cli::write_line("option name MultiPV type spin default 1 min 1 max 8");
        cli::write_line("option name CPI_Enabled type check default true");
        cli::write_line("option name State_Enabled type check default true");
        cli::write_line("option name Risk_Enabled type check default true");
        cli::write_line("option name Pressure_Enabled type check default true");
        cli::write_line("option name Attack_Enabled type check default true");
        cli::write_line("option name Conversion_Enabled type check default true");
        cli::write_line("option name QS_Checks_Enabled type check default true");
        cli::write_line("option name MustTryGain type spin default 30 min 10 max 100");
        cli::write_line("option name MustTryRisk type spin default 120 min 40 max 250");
        cli::write_line("option name ConvertScore type spin default 250 min 100 max 600");
        cli::write_line("option name CrushScore type spin default 600 min 300 max 1200");
        cli::write_line("option name DefendCPI type spin default 120 min 40 max 250");
        cli::write_line("option name DualNet type check default true");

        if let Some(path) = crate::eval::nnue::v16::try_load_default_path() {
            cli::write_line(&format!("info string SFNNv16 network loaded: {path}"));
        }
        cli::write_line("uciok");
    }

    pub(crate) fn run_perft(&mut self, parameters: &[&str]) {
        let depth: u8 = parameters
            .first()
            .and_then(|d| d.parse::<u8>().ok())
            .unwrap_or(5);
        let mut board = self.board.lock().unwrap().clone();
        let start = std::time::Instant::now();
        let nodes = perft_count(&mut board, depth);
        let elapsed = start.elapsed();
        let nps = if elapsed.as_secs_f64() > 0.0 {
            (nodes as f64 / elapsed.as_secs_f64()) as u64
        } else {
            0
        };
        cli::write_line(&format!(
            "info string perft depth {depth}: {nodes} nodes in {} ms ({nps} nps)",
            elapsed.as_millis()
        ));
    }

    pub(crate) fn run_bench(&mut self, parameters: &[&str]) {
        use crate::common::helpers::BENCH_FENS;
        let hash_mb: usize = parameters
            .first()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16);
        let threads: usize = parameters
            .get(1)
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let depth: u8 = parameters
            .get(2)
            .and_then(|v| v.parse::<u8>().ok())
            .unwrap_or(12);
        if let Some(&model) = parameters.get(3) {
            let _ = crate::eval::nnue::v16::set_eval_file("Model", model);
        }
        let positions = BENCH_FENS;
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let mut debug = false;
        let mut total_nodes: u64 = 0;
        let total_start = std::time::Instant::now();

        let shared_tt = std::sync::Arc::new(crate::common::tt::TranspositionTable::new_mb(
            hash_mb.clamp(1, 2048),
        ));
        for (index, fen) in positions.iter().enumerate() {
            let mut board = BoardState::parse_fen(fen);
            let mut state = crate::search::search_state::SearchState::new();
            state.tt = std::sync::Arc::clone(&shared_tt);
            let start = std::time::Instant::now();
            board.find_best_move(
                depth,
                &cancel,
                &mut debug,
                &mut state,
                threads.clamp(1, 256),
            );
            let elapsed = start.elapsed();
            total_nodes += state.nodes;
            cli::write_line(&format!(
                "info string bench {}: {} nodes in {} ms",
                index + 1,
                state.nodes,
                elapsed.as_millis()
            ));
        }
        let total_elapsed = total_start.elapsed();
        let nps = if total_elapsed.as_secs_f64() > 0.0 {
            (total_nodes as f64 / total_elapsed.as_secs_f64()) as u64
        } else {
            0
        };
        cli::write_line(&format!(
            "info string bench done: {total_nodes} nodes in {} ms ({nps} nps) hashfull {}",
            total_elapsed.as_millis(),
            shared_tt.hashfull()
        ));
    }

    pub(crate) fn run_model(&mut self, parameters: &[&str]) {
        if let Some(&name) = parameters.first() {
            let res = crate::eval::nnue::v16::set_eval_file("Model", name);
            match res {
                Ok(msg) => cli::write_line(&format!("info string {msg}: {name}")),
                Err(e) => cli::write_line(&format!("info string Failed to load model {name}: {e}")),
            }
        } else {
            let current = crate::eval::nnue::v16::active_model_name();
            cli::write_line(&format!("info string Active model: {current}"));
        }
    }
}

fn perft_count(board: &mut BoardState, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    let mut moves = crate::common::move_list::MoveList::new();
    board.generate_moves(&mut moves);
    for m in moves.iter() {
        board.make_move(m.mv);
        if !board.is_in_check(board.side_to_move.other()) {
            nodes += perft_count(board, depth - 1);
        }
        board.unmake_move(m.mv);
    }
    nodes
}

pub(crate) fn output_best_move(move_obj: Move, ponder_obj: Move) {
    if move_obj == Move::NO_MOVE {
        cli::write_line("bestmove (none)");
        return;
    }

    let promotion = move_obj
        .promotion_char()
        .map(|c| c.to_string())
        .unwrap_or_default();

    if ponder_obj != Move::NO_MOVE {
        let ponder_promo = ponder_obj
            .promotion_char()
            .map(|c| c.to_string())
            .unwrap_or_default();
        cli::write_line(&format!(
            "bestmove {}{}{} ponder {}{}{}",
            move_obj.source,
            move_obj.target,
            promotion,
            ponder_obj.source,
            ponder_obj.target,
            ponder_promo
        ));
    } else {
        cli::write_line(&format!(
            "bestmove {}{}{}",
            move_obj.source, move_obj.target, promotion
        ));
    }
}

pub(crate) fn get_parameter(name: &str, parameters: &[&str], fallback: i32) -> i32 {
    for i in 0..parameters.len() {
        if parameters[i] == name
            && i + 1 < parameters.len()
            && let Ok(value) = parameters[i + 1].parse::<i32>()
        {
            return value;
        }
    }
    fallback
}

pub(crate) fn get_u64(name: &str, parameters: &[&str]) -> Option<u64> {
    for i in 0..parameters.len() {
        if parameters[i] == name && i + 1 < parameters.len() {
            return parameters[i + 1].parse::<u64>().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn perft_counts_legal_nodes_without_changing_position() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let original = board.clone();
        for (depth, expected) in [(0, 1), (1, 20), (2, 400), (3, 8902)] {
            assert_eq!(perft_count(&mut board, depth), expected);
            assert!(board == original);
        }
    }

    #[test]
    fn diagnostic_commands_preserve_client_position() {
        let mut client = UciClient::new();
        let original = client.board.lock().unwrap().clone();
        client.run_perft(&["2"]);
        client.run_bench(&["1", "1", "1"]);
        assert!(*client.board.lock().unwrap() == original);
    }

    #[test]
    fn ponderhit_ends_pondering_without_cancelling_search() {
        let mut client = UciClient::new();
        let cancel = Arc::new(AtomicBool::new(false));
        client.current_search = Some(Arc::clone(&cancel));
        client.is_pondering.store(true, Ordering::Relaxed);
        client.run_ponderhit();
        assert!(!client.is_pondering.load(Ordering::Relaxed));
        assert!(!cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn write_id_lists_options_without_changing_position() {
        let client = UciClient::new();
        let original = client.board.lock().unwrap().clone();
        client.write_id();
        assert!(*client.board.lock().unwrap() == original);
    }

    #[test]
    fn get_parameter_parses_and_falls_back() {
        assert_eq!(get_parameter("depth", &["depth", "5"], 8), 5);
        assert_eq!(get_parameter("depth", &[], 8), 8);
        assert_eq!(get_parameter("depth", &["depth"], 8), 8);
        assert_eq!(get_parameter("depth", &["depth", "xx"], 8), 8);
        assert_eq!(get_parameter("depth", &["movetime", "5"], 8), 8);
        assert_eq!(get_parameter("depth", &["depth", "5", "depth", "7"], 8), 5);
    }

    #[test]
    fn output_best_move_handles_all_shapes() {
        use crate::common::moves::Move;

        output_best_move(Move::NO_MOVE, Move::NO_MOVE);
        output_best_move(Move::parse_long_algebraic("e2e4").unwrap(), Move::NO_MOVE);
        output_best_move(
            Move::parse_long_algebraic("e7e8q").unwrap(),
            Move::parse_long_algebraic("e2e4").unwrap(),
        );
    }

    #[test]
    fn get_u64_parses_unsigned_only() {
        assert_eq!(get_u64("wtime", &["wtime", "1000"]), Some(1000));
        assert_eq!(get_u64("wtime", &[]), None);
        assert_eq!(get_u64("wtime", &["wtime"]), None);
        assert_eq!(get_u64("wtime", &["wtime", "xx"]), None);

        assert_eq!(get_u64("wtime", &["wtime", "-5"]), None);
        assert_eq!(get_u64("wtime", &["btime", "5"]), None);
        assert_eq!(get_u64("nodes", &["nodes", "0"]), Some(0));
    }

    #[test]
    fn get_parameter_handles_negative_and_duplicates() {
        assert_eq!(get_parameter("movestogo", &["movestogo", "-1"], 0), -1);
        assert_eq!(get_parameter("depth", &["depth", "-3"], 8), -3);

        assert_eq!(get_parameter("depth", &["depth", "3", "depth", "9"], 8), 3);
    }

    #[test]
    fn output_best_move_promotion_shapes() {
        use crate::common::moves::Move;

        output_best_move(Move::parse_long_algebraic("a7a8n").unwrap(), Move::NO_MOVE);

        output_best_move(
            Move::parse_long_algebraic("e2e4").unwrap(),
            Move::parse_long_algebraic("a7a8r").unwrap(),
        );

        output_best_move(
            Move::parse_long_algebraic("e7e8q").unwrap(),
            Move::parse_long_algebraic("a2a1n").unwrap(),
        );
    }

    #[test]
    fn perft_depth_zero_and_one_are_stable() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let original = board.clone();
        assert_eq!(perft_count(&mut board, 0), 1);
        assert_eq!(perft_count(&mut board, 1), 20);
        assert!(board == original);
    }

    #[test]
    fn output_best_move_plain_with_plain_ponder() {
        use crate::common::moves::Move;

        output_best_move(
            Move::parse_long_algebraic("e2e4").unwrap(),
            Move::parse_long_algebraic("e7e5").unwrap(),
        );
    }

    #[test]
    fn output_best_move_none_ignores_ponder() {
        use crate::common::moves::Move;

        output_best_move(Move::NO_MOVE, Move::parse_long_algebraic("e2e4").unwrap());
    }

    #[test]
    fn get_u64_max_value_and_first_occurrence_wins() {
        assert_eq!(
            get_u64("wtime", &["wtime", "18446744073709551615"]),
            Some(u64::MAX)
        );
        assert_eq!(
            get_u64("wtime", &["wtime", "100", "wtime", "200"]),
            Some(100)
        );

        assert_eq!(get_u64("wtime", &["wtime", "xx", "wtime", "200"]), None);
    }

    #[test]
    fn perft_kiwipete_shallow_depths() {
        use crate::common::helpers::KIWI_PETE_FEN;

        let mut board = BoardState::parse_fen(KIWI_PETE_FEN);
        let original = board.clone();
        for (depth, expected) in [(0, 1), (1, 48), (2, 2039)] {
            assert_eq!(perft_count(&mut board, depth), expected);
            assert!(board == original);
        }
    }

    #[test]
    fn run_perft_on_kiwipete_preserves_position() {
        use crate::common::helpers::KIWI_PETE_FEN;

        let mut client = UciClient::new();
        let parts: Vec<&str> = KIWI_PETE_FEN.split_whitespace().collect();
        let mut params = vec!["fen"];
        params.extend(parts);
        client.run_position(&params);
        let original = client.board.lock().unwrap().clone();
        client.run_perft(&["1"]);
        client.run_perft(&["2"]);
        assert!(*client.board.lock().unwrap() == original);
    }

    #[test]
    fn run_bench_tiny_hash_preserves_position() {
        let mut client = UciClient::new();
        let original = client.board.lock().unwrap().clone();
        client.run_bench(&["2", "1", "1"]);
        assert!(*client.board.lock().unwrap() == original);
    }

    #[test]
    fn run_perft_small_depths_preserve_position() {
        let mut client = UciClient::new();
        let original = client.board.lock().unwrap().clone();
        client.run_perft(&["0"]);
        client.run_perft(&["1"]);
        assert!(*client.board.lock().unwrap() == original);
    }
}
