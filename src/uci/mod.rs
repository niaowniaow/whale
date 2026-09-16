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
    client.run();
}

pub(crate) struct UciClient {
    pub board: Arc<Mutex<BoardState>>,
    pub debug_mode: Arc<AtomicBool>,
    pub current_search: Option<Arc<AtomicBool>>,
    pub search_state: Arc<Mutex<crate::search::search_state::SearchState>>,
    pub is_ready: bool,
    pub is_pondering: Arc<AtomicBool>,
    pub num_threads: usize,
    pub move_overhead: i32,
}

impl UciClient {
    pub(crate) fn new() -> Self {
        Self {
            board: Arc::new(Mutex::new(BoardState::parse_fen(STARTING_FEN))),
            debug_mode: Arc::new(AtomicBool::new(true)),
            current_search: None,
            search_state: Arc::new(Mutex::new(crate::search::search_state::SearchState::new())),
            is_ready: false,
            is_pondering: Arc::new(AtomicBool::new(false)),
            num_threads: 1,
            move_overhead: 200,
        }
    }

    fn run(&mut self) {
        self.write_id();
        if let Some(path) = crate::eval::nnue::v16::try_load_default_path() {
            cli::write_line(&format!("info string SFNNv16 network loaded: {path}"));
        }

        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            if stdin.read_line(&mut line).is_err() {
                continue;
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            let command = parts[0];
            let parameters = &parts[1..];

            if command == "quit" {
                std::process::exit(0);
            }

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
                _ => cli::write_line(&format!("Unknown command {command}")),
            }
        }
    }

    pub(crate) fn run_ponderhit(&mut self) {
        self.is_pondering.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    fn write_id(&self) {
        cli::write_line(&format!("id name Rudim {}", env!("CARGO_PKG_VERSION")));
        cli::write_line("id author Vishnu B");
        cli::write_line("option name Hash type spin default 64 min 1 max 2048");
        cli::write_line("option name Threads type spin default 1 min 1 max 256");
        cli::write_line("option name Move Overhead type spin default 200 min 0 max 5000");
        cli::write_line("option name Ponder type check default true");
        cli::write_line("option name EvalFile type string default <empty>");
        cli::write_line("option name EvalFileSmall type string default <empty>");
        cli::write_line("option name SyzygyPath type string default <empty>");
        cli::write_line("option name RFP_Margin type spin default 110 min 50 max 300");
        cli::write_line("option name Futility_Margin type spin default 120 min 50 max 300");
        cli::write_line("option name Singular_Margin type spin default 2 min 1 max 5");
        cli::write_line("option name ProbCut_Margin type spin default 170 min 50 max 400");
        cli::write_line("option name NMP_Base type spin default 3 min 1 max 6");
        cli::write_line("option name NMP_Depth_Div type spin default 3 min 2 max 8");
        cli::write_line("option name LMR_Base type spin default 65 min 10 max 150");
        cli::write_line("option name LMR_Div type spin default 215 min 100 max 350");
        cli::write_line("option name History_Weight type spin default 2 min 1 max 4");

        cli::write_line("uciok");
    }
}

pub(crate) fn output_best_move(move_obj: Move, ponder_obj: Move) {
    if move_obj == Move::NO_MOVE {
        cli::write_line("bestmove 0000");
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
            move_obj.source, move_obj.target, promotion,
            ponder_obj.source, ponder_obj.target, ponder_promo
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

#[allow(dead_code)]
pub(crate) fn has_flag(name: &str, parameters: &[&str]) -> bool {
    parameters.contains(&name)
}
