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
        evaluate_on(&mut self.stdin, &mut self.stdout, board, depth)
    }
}

impl Drop for StockfishTeacher {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn evaluate_on(
    stdin: &mut impl Write,
    stdout: &mut impl BufRead,
    board: &BoardState,
    depth: u8,
) -> std::io::Result<i16> {
    writeln!(stdin, "position fen {}", board.to_fen())?;
    writeln!(stdin, "go depth {}", depth)?;
    stdin.flush()?;

    let mut score = None;
    let mut line = String::new();
    loop {
        line.clear();
        if stdout.read_line(&mut line)? == 0 {
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

fn wait_for_token(reader: &mut impl BufRead, expected: &str) -> std::io::Result<()> {
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
    use std::io::Cursor;

    #[test]
    fn parses_centipawn_and_mate_scores() {
        assert_eq!(parse_score(&["info", "score", "cp", "42"]), Some(42));
        assert_eq!(parse_score(&["info", "score", "mate", "3"]), Some(30994));
        assert_eq!(parse_score(&["info", "score", "cp", "-50"]), Some(-50));
    }

    #[test]
    fn parse_score_clamps_and_rejects_garbage() {
        assert_eq!(
            parse_score(&["info", "score", "cp", "99999"]),
            Some(MAX_CENTIPAWN_EVAL)
        );
        assert_eq!(
            parse_score(&["info", "score", "cp", "-99999"]),
            Some(-MAX_CENTIPAWN_EVAL)
        );

        assert_eq!(
            parse_score(&["info", "score", "mate", "-2"]),
            Some(-(MAX_CENTIPAWN_EVAL - 4))
        );

        assert_eq!(parse_score(&["info", "score", "lowerbound", "10"]), None);
        assert_eq!(parse_score(&["info", "score"]), None);
        assert_eq!(parse_score(&["info"]), None);
        assert_eq!(parse_score(&[]), None);
        assert_eq!(parse_score(&["info", "score", "cp"]), None);
        assert_eq!(parse_score(&["info", "score", "cp", "xx"]), None);
        assert_eq!(parse_score(&["info", "score", "mate"]), None);
    }

    #[test]
    fn wait_for_token_finds_token_and_reports_eof() {
        let mut found = Cursor::new("id name x\nuciok\n");
        assert!(wait_for_token(&mut found, "uciok").is_ok());

        let mut missing = Cursor::new("id name x\nreadyok\n");
        assert!(wait_for_token(&mut missing, "uciok").is_err());

        let mut empty = Cursor::new("");
        assert!(wait_for_token(&mut empty, "uciok").is_err());
    }

    #[test]
    fn teacher_rejects_missing_binary() {
        assert!(StockfishTeacher::new("definitely/not/a-chess-engine").is_err());
    }

    #[test]
    fn evaluate_on_returns_last_score_before_bestmove() {
        use crate::common::helpers::STARTING_FEN;

        let mut stdin = Vec::new();
        let transcript = "info depth 1 score cp 10 pv e2e4\n\
                          info depth 1 score cp 25 pv d2d4\n\
                          bestmove d2d4\n";
        let mut stdout = Cursor::new(transcript.as_bytes());
        let board = BoardState::parse_fen(STARTING_FEN);
        assert_eq!(evaluate_on(&mut stdin, &mut stdout, &board, 1).unwrap(), 25);
        let sent = String::from_utf8(stdin).unwrap();
        assert!(sent.contains("position fen"));
        assert!(sent.contains("go depth 1"));
    }

    #[test]
    fn evaluate_on_defaults_to_zero_without_score() {
        use crate::common::helpers::STARTING_FEN;

        let board = BoardState::parse_fen(STARTING_FEN);

        let mut stdin = Vec::new();
        let mut stdout = Cursor::new(b"info string hi\n" as &[u8]);
        assert_eq!(evaluate_on(&mut stdin, &mut stdout, &board, 1).unwrap(), 0);

        let mut stdin = Vec::new();
        let mut stdout = Cursor::new(b"bestmove e2e4\n" as &[u8]);
        assert_eq!(evaluate_on(&mut stdin, &mut stdout, &board, 1).unwrap(), 0);
    }

    #[test]
    fn drop_reaps_exited_child() {
        #[cfg(windows)]
        let mut child = Command::new("cmd")
            .args(["/C", "exit", "0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        #[cfg(not(windows))]
        let mut child = Command::new("true")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let teacher = StockfishTeacher {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        };
        drop(teacher);
    }
}
