use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::board::state::BoardState;
use crate::common::constants::MAX_CENTIPAWN_EVAL;

pub struct StockfishTeacher {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl StockfishTeacher {
    pub fn new(path: &str) -> std::io::Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdin = child.stdin.take().expect("Stockfish stdin was not piped");
        let stdout = child.stdout.take().expect("Stockfish stdout was not piped");
        let mut reader = BufReader::new(stdout);

        writeln!(stdin, "uci")?;
        wait_for_token(&mut reader, "uciok")?;
        writeln!(stdin, "isready")?;
        wait_for_token(&mut reader, "readyok")?;

        Ok(Self {
            child,
            stdin,
            stdout: reader,
        })
    }

    pub fn evaluate(&mut self, board: &BoardState, depth: u8) -> std::io::Result<i16> {
        writeln!(self.stdin, "position fen {}", board.to_fen())?;
        writeln!(self.stdin, "go depth {}", depth)?;
        self.stdin.flush()?;

        let mut score = None;
        let mut line = String::new();
        loop {
            line.clear();
            if self.stdout.read_line(&mut line)? == 0 {
                break;
            }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if let Some(parsed) = parse_score(&tokens) {
                score = Some(parsed);
            }
            if tokens.first() == Some(&"bestmove") {
                break;
            }
        }

        Ok(score.unwrap_or(0))
    }
}

impl Drop for StockfishTeacher {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn wait_for_token(reader: &mut BufReader<ChildStdout>, expected: &str) -> std::io::Result<()> {
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("Stockfish exited before sending {expected}"),
            ));
        }
        if line.split_whitespace().any(|token| token == expected) {
            return Ok(());
        }
    }
}

fn parse_score(tokens: &[&str]) -> Option<i16> {
    let score_index = tokens.iter().position(|token| *token == "score")?;
    let score_type = *tokens.get(score_index + 1)?;
    let value = tokens.get(score_index + 2)?.parse::<i32>().ok()?;

    let score = match score_type {
        "cp" => value.clamp(
            -i32::from(MAX_CENTIPAWN_EVAL),
            i32::from(MAX_CENTIPAWN_EVAL),
        ),
        "mate" => {
            let sign = value.signum();
            sign * (i32::from(MAX_CENTIPAWN_EVAL) - value.abs().saturating_mul(2).min(1000))
        }
        _ => return None,
    };
    Some(score as i16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_centipawn_and_mate_scores() {
        assert_eq!(parse_score(&["info", "score", "cp", "42"]), Some(42));
        assert_eq!(parse_score(&["info", "score", "mate", "3"]), Some(30994));
        assert_eq!(parse_score(&["info", "score", "cp", "-50"]), Some(-50));
    }
}
