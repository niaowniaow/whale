pub mod bitboard;
pub mod board;
pub mod common;
pub mod endgame;
pub mod eval;
pub mod opening;
pub mod opponent;
pub mod opportunity;
pub mod perception;
pub mod risk;
pub mod root;
pub mod search;
pub mod syzygy;
pub mod teacher;
pub mod uci;
pub mod world;

#[cfg(feature = "train")]
pub mod datagen;
#[cfg(not(feature = "train"))]
pub mod datagen {
    pub fn run(
        _output_path: &str,
        _num_games: usize,
        _book_path: &str,
        _depth: u8,
        _threads: usize,
    ) {
        eprintln!("Error: This build of whale was compiled without the 'train' feature.");
        std::process::exit(1);
    }

    pub fn run_with_teacher(
        _output_path: &str,
        _num_games: usize,
        _book_path: &str,
        _depth: u8,
        _threads: usize,
        _teacher_path: Option<&str>,
    ) {
        eprintln!("Error: This build of whale was compiled without the 'train' feature.");
        std::process::exit(1);
    }
}

#[cfg(feature = "train")]
pub mod train;
#[cfg(not(feature = "train"))]
pub mod train {
    pub fn run(_custom_dataset_path: Option<&str>) {
        eprintln!("Error: This build of whale was compiled without the 'train' feature.");
        std::process::exit(1);
    }

    pub fn run_smoke(_custom_dataset_path: Option<&str>) {
        eprintln!("Error: This build of whale was compiled without the 'train' feature.");
        std::process::exit(1);
    }
}

pub fn init() {
    let _ = crate::eval::nnue::v16::try_load_default();
}

#[cfg(test)]
mod tests {
    #[test]
    fn init_is_callable() {
        let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
        super::init();
    }

    #[test]
    fn exit_probe_child() {
        match std::env::var("WHALE_EXIT_PROBE").as_deref() {
            Ok("datagen_run") => crate::datagen::run("x", 0, "x", 0, 0),
            Ok("datagen_teacher") => crate::datagen::run_with_teacher("x", 0, "x", 0, 0, None),
            Ok("train_run") => crate::train::run(None),
            Ok("train_smoke") => crate::train::run_smoke(None),
            Ok("cli_run") => crate::uci::cli::run(),
            _ => {}
        }
    }

    #[cfg(test)]
    fn probe_with_mode(
        mode: &str,
        stdin_data: Option<&[u8]>,
    ) -> (std::process::ExitStatus, String, String) {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("tests::exit_probe_child")
            .arg("--nocapture")
            .env("WHALE_EXIT_PROBE", mode)
            .stdin(if stdin_data.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        if let Some(data) = stdin_data {
            child.stdin.as_mut().unwrap().write_all(data).unwrap();
            drop(child.stdin.take());
        }
        let out = child.wait_with_output().unwrap();
        (
            out.status,
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    #[cfg(not(feature = "train"))]
    #[test]
    fn exit_stubs_terminate_with_error() {
        for mode in ["datagen_run", "datagen_teacher", "train_run", "train_smoke"] {
            let (status, _, err) = probe_with_mode(mode, None);
            assert_eq!(status.code(), Some(1), "mode {mode}");
            assert!(err.contains("train"), "mode {mode}");
        }
    }

    #[test]
    fn cli_run_processes_info_and_exit() {
        let (status, out, _) = probe_with_mode("cli_run", Some(b"\nbogus\ninfo\nexit\n"));
        assert!(status.success());
        assert!(out.contains("Whale v"));
    }

    #[test]
    fn cli_run_eof_breaks_loop() {
        let (status, out, _) = probe_with_mode("cli_run", Some(b"info\n"));
        assert!(status.success());
        assert!(out.contains("Whale v"));
    }

    #[test]
    fn cli_run_dispatches_uci_and_quit() {
        let (status, _, _) = probe_with_mode("cli_run", Some(b"uci\nquit\n"));
        assert!(status.success());
    }

    #[cfg(not(feature = "train"))]
    #[test]
    fn cli_run_valid_datagen_reaches_stub() {
        let (status, _, err) = probe_with_mode("cli_run", Some(b"datagen out 1 book 1 1\n"));
        assert_eq!(status.code(), Some(1));
        assert!(err.contains("train"));
    }
}
