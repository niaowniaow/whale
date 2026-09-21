use std::env::args;
use std::process::exit;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use whale::bitboard::magics::generate_all_magic_numbers;
use whale::board::state::BoardState;
use whale::common::helpers::{
    ADVANCED_MOVE_FEN, BENCH_FENS, ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN,
};
use whale::init;
use whale::search::search_state::SearchState;
use whale::train::{run as train_run, run_smoke as train_smoke_run};
use whale::uci::cli::run as uci_run;

fn main() {
    let builder = std::thread::Builder::new()
        .name("whale-main".into())
        .stack_size(16 * 1024 * 1024);
    let handler = builder.spawn(real_main).unwrap();
    if let Err(e) = handler.join() {
        std::panic::resume_unwind(e);
    }
}

fn real_main() {
    let raw_args: Vec<String> = args().collect();

    match raw_args.get(1).map(String::as_str) {
        Some("--generate-magics") => {
            generate_all_magic_numbers();
        }
        Some("--train") => {
            init();
            let dataset_path = raw_args.get(2).map(String::as_str);
            train_run(dataset_path);
        }
        Some("--train-smoke") => {
            init();
            let dataset_path = raw_args.get(2).map(String::as_str);
            train_smoke_run(dataset_path);
        }
        Some("--profile") => {
            run_searches();
        }
        Some("bench") | Some("--bench") => {
            init();
            let hash_mb: usize = raw_args
                .get(2)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(16)
                .clamp(1, 2048);
            let threads: usize = raw_args
                .get(3)
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 256);
            let depth: u8 = raw_args
                .get(4)
                .and_then(|v| v.parse::<u8>().ok())
                .unwrap_or(12)
                .clamp(1, 64);
            let model = raw_args.get(5).map(String::as_str);
            if let Some(m) = model {
                let _ = whale::eval::nnue::v16::set_eval_file("Model", m);
            } else if std::path::Path::new("models/whale_big.nnue").exists() {
                let _ = whale::eval::nnue::v16::set_eval_file("Model", "whale_big");
            }
            let active_net = whale::eval::nnue::v16::active_model_name();
            println!("Bench: hash={hash_mb}MB threads={threads} depth={depth} model={active_net}");
            let positions = BENCH_FENS;
            let cancellation_token = AtomicBool::new(false);
            let mut debug_mode = false;
            let mut total_nodes: u64 = 0;
            let total_start = Instant::now();
            let shared_tt =
                std::sync::Arc::new(whale::common::tt::TranspositionTable::new_mb(hash_mb));
            for (index, fen) in positions.iter().enumerate() {
                cancellation_token.store(false, std::sync::atomic::Ordering::Relaxed);
                let mut search_state = SearchState::new();
                search_state.tt = std::sync::Arc::clone(&shared_tt);
                let mut board = BoardState::parse_fen(fen);
                let start = Instant::now();
                let best = board.find_best_move(
                    depth,
                    &cancellation_token,
                    &mut debug_mode,
                    &mut search_state,
                    threads,
                );
                let elapsed_ms = start.elapsed().as_millis();
                total_nodes += search_state.nodes;
                let promo = best
                    .promotion_char()
                    .map(|c| c.to_string())
                    .unwrap_or_default();
                println!(
                    "Position {}/{}: bestmove {}{}{} {} nodes {} ms",
                    index + 1,
                    positions.len(),
                    best.source,
                    best.target,
                    promo,
                    search_state.nodes,
                    elapsed_ms
                );
            }
            let total_ms = total_start.elapsed().as_millis();
            let total_secs = total_start.elapsed().as_secs_f64();
            let nps = if total_secs > 0.0 {
                (total_nodes as f64 / total_secs) as u64
            } else {
                0
            };
            println!(
                "Total: {total_nodes} nodes {total_ms} ms {nps} nps hashfull {}",
                shared_tt.hashfull()
            );
            exit(0);
        }
        Some("datagen") | Some("--datagen") | Some("datagen-teacher") => {
            let uses_teacher =
                matches!(raw_args.get(1).map(String::as_str), Some("datagen-teacher"));
            if raw_args.len() < if uses_teacher { 8 } else { 5 } {
                let command_name = if uses_teacher {
                    "datagen-teacher"
                } else {
                    "datagen"
                };
                let teacher_suffix = if uses_teacher {
                    " <stockfish_binary>"
                } else {
                    ""
                };
                eprintln!(
                    "Usage: {} {} <output.binpack> <number_of_games> <opening_book.fen> [depth] [threads]{}",
                    raw_args[0], command_name, teacher_suffix
                );
                exit(1);
            }
            init();
            let output_path = &raw_args[2];
            let num_games = match raw_args[3].parse::<usize>() {
                Ok(n) => n,
                Err(_) => {
                    eprintln!("Error: invalid number of games");
                    exit(1);
                }
            };
            let book_path = &raw_args[4];
            let depth = if raw_args.len() > 5 {
                match raw_args[5].parse::<u8>() {
                    Ok(d) => d,
                    Err(_) => {
                        eprintln!("Error: invalid depth");
                        exit(1);
                    }
                }
            } else {
                8
            };
            let threads = if raw_args.len() > 6 {
                match raw_args[6].parse::<usize>() {
                    Ok(t) => t,
                    Err(_) => {
                        eprintln!("Error: invalid thread count");
                        exit(1);
                    }
                }
            } else {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(4)
            };
            let teacher_path = if uses_teacher {
                raw_args.get(7).map(String::as_str)
            } else {
                None
            };
            whale::datagen::run_with_teacher(
                output_path,
                num_games,
                book_path,
                depth,
                threads,
                teacher_path,
            );
            exit(0);
        }
        Some("--model") => {
            init();
            let model_name = raw_args.get(2).map(String::as_str).unwrap_or("whale_big");
            let _ = whale::eval::nnue::v16::set_eval_file("Model", model_name);
            uci_run();
        }
        _ => {
            init();
            if std::path::Path::new("models/whale_big.nnue").exists() {
                let _ = whale::eval::nnue::v16::set_eval_file("Model", "whale_big");
            }
            uci_run();
        }
    }
}

fn run_searches() {
    const PROFILE_DEPTH: u8 = 13;

    init();

    let positions = [
        ("Starting Position", STARTING_FEN),
        ("Kiwi Pete", KIWI_PETE_FEN),
        ("Endgame", ENDGAME_FEN),
        ("Advanced Position", ADVANCED_MOVE_FEN),
    ];

    let cancellation_token = AtomicBool::new(false);
    let mut debug_mode = true;
    let mut search_state = SearchState::new();

    for (name, fen) in positions {
        println!("\nProfiling Position: {}", name);
        println!("FEN: {}", fen);

        search_state.tt.clear();
        search_state.reset_heuristics();
        search_state.reset_search();

        let mut board = BoardState::parse_fen(fen);
        let start_time = Instant::now();
        let best_move = board.find_best_move(
            PROFILE_DEPTH,
            &cancellation_token,
            &mut debug_mode,
            &mut search_state,
            1,
        );
        let duration = start_time.elapsed();

        let promo = best_move
            .promotion_char()
            .map(|c| c.to_string())
            .unwrap_or_default();
        println!(
            "Best move: {}{}{}",
            best_move.source, best_move.target, promo
        );
        println!("Time taken: {:?}", duration);
    }
}
