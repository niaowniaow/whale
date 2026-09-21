use crate::datagen;
use crate::uci;
use std::collections::HashMap;
use std::io::{self, Write};

trait CliCommand {
    fn run(&self, parameters: &[&str]);
}

struct InfoCommand;

impl CliCommand for InfoCommand {
    fn run(&self, _parameters: &[&str]) {
        write_line(&format!("Whale v{} by Vishnu B", env!("CARGO_PKG_VERSION")));
    }
}

struct UciCommand;

impl CliCommand for UciCommand {
    fn run(&self, parameters: &[&str]) {
        uci::run(parameters);
    }
}

struct DatagenCommand;
impl CliCommand for DatagenCommand {
    fn run(&self, parameters: &[&str]) {
        if parameters.len() < 3 {
            write_line(
                "Usage: datagen <output.binpack> <number_of_games> <opening_book.fen> [depth] [threads]",
            );
            return;
        }
        let output_path = parameters[0];
        let num_games = match parameters[1].parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                write_line("Error: invalid number of games");
                return;
            }
        };
        let book_path = parameters[2];
        let depth = if parameters.len() > 3 {
            match parameters[3].parse::<u8>() {
                Ok(d) => d,
                Err(_) => {
                    write_line("Error: invalid depth");
                    return;
                }
            }
        } else {
            8
        };

        let threads = if parameters.len() > 4 {
            match parameters[4].parse::<usize>() {
                Ok(t) => t,
                Err(_) => {
                    write_line("Error: invalid thread count");
                    return;
                }
            }
        } else {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        };

        datagen::run(output_path, num_games, book_path, depth, threads);
    }
}

struct DatagenTeacherCommand;
impl CliCommand for DatagenTeacherCommand {
    fn run(&self, parameters: &[&str]) {
        if parameters.len() < 4 {
            write_line(
                "Usage: datagen-teacher <output.binpack> <number_of_games> <opening_book.fen> <stockfish_binary> [depth] [threads]",
            );
            return;
        }
        let output_path = parameters[0];
        let num_games = match parameters[1].parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                write_line("Error: invalid number of games");
                return;
            }
        };
        let book_path = parameters[2];
        let teacher_path = parameters[3];
        let depth = if parameters.len() > 4 {
            match parameters[4].parse::<u8>() {
                Ok(d) => d,
                Err(_) => {
                    write_line("Error: invalid depth");
                    return;
                }
            }
        } else {
            8
        };
        let threads = if parameters.len() > 5 {
            match parameters[5].parse::<usize>() {
                Ok(t) => t,
                Err(_) => {
                    write_line("Error: invalid thread count");
                    return;
                }
            }
        } else {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        };

        datagen::run_with_teacher(
            output_path,
            num_games,
            book_path,
            depth,
            threads,
            Some(teacher_path),
        );
    }
}

struct BenchCommand;
impl CliCommand for BenchCommand {
    fn run(&self, parameters: &[&str]) {
        let mut client = uci::UciClient::new();
        client.run_bench(parameters);
    }
}

struct ModelCommand;
impl CliCommand for ModelCommand {
    fn run(&self, parameters: &[&str]) {
        if parameters.is_empty() {
            let active = crate::eval::nnue::v16::active_model_name();
            write_line(&format!("Active model: {active}"));
            return;
        }
        let model_name = parameters[0];
        match crate::eval::nnue::v16::set_eval_file("Model", model_name) {
            Ok(msg) => write_line(&format!("info string {msg}: {model_name}")),
            Err(e) => write_line(&format!("Error loading model {model_name}: {e}")),
        }
    }
}

pub fn run() {
    let mut commands: HashMap<&str, Box<dyn CliCommand>> = HashMap::new();
    commands.insert("info", Box::new(InfoCommand));
    commands.insert("uci", Box::new(UciCommand));
    commands.insert("datagen", Box::new(DatagenCommand));
    commands.insert("datagen-teacher", Box::new(DatagenTeacherCommand));
    commands.insert("bench", Box::new(BenchCommand));
    commands.insert("model", Box::new(ModelCommand));

    let stdin = io::stdin();

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

        if command == "exit" || command == "quit" {
            break;
        }

        if let Some(cli_command) = commands.get(command) {
            cli_command.run(parameters);
        } else if matches!(
            command,
            "isready" | "position" | "go" | "stop" | "setoption" | "ucinewgame" | "perft"
        ) {
            let mut client = uci::UciClient::new();
            client.handle_command(command, parameters);
            client.run();
        } else {
            eprintln!("Unknown command {command}");
        }

        let _ = io::stdout().flush();
    }
}

pub fn write_line(message: &str) {
    println!("{message}");
    let _ = io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_command_prints_version() {
        InfoCommand.run(&[]);
    }

    #[test]
    fn datagen_rejects_bad_cli_args() {
        let cmd = DatagenCommand;
        cmd.run(&[]);
        cmd.run(&["out.binpack"]);
        cmd.run(&["out.binpack", "8"]);
        cmd.run(&["out.binpack", "many", "book.fen"]);
        cmd.run(&["out.binpack", "8", "book.fen", "deep"]);
        cmd.run(&["out.binpack", "8", "book.fen", "3", "lots"]);
    }

    #[test]
    fn datagen_teacher_rejects_bad_cli_args() {
        let cmd = DatagenTeacherCommand;
        cmd.run(&[]);
        cmd.run(&["out.binpack"]);
        cmd.run(&["out.binpack", "8"]);
        cmd.run(&["out.binpack", "many", "book.fen", "sf.exe"]);
        cmd.run(&["out.binpack", "8", "book.fen", "sf.exe", "deep"]);
        cmd.run(&["out.binpack", "8", "book.fen", "sf.exe", "3", "lots"]);
    }

    #[test]
    fn bench_command_runs_shallow() {
        let cmd = BenchCommand;
        cmd.run(&["1", "1", "1"]);
    }

    #[test]
    fn model_command_queries_and_sets() {
        let cmd = ModelCommand;
        cmd.run(&[]);
        cmd.run(&["embedded"]);
    }
}
