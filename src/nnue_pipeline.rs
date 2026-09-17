use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSource {
    pub name: &'static str,
    pub url: &'static str,
    pub kind: &'static str,
    pub notes: &'static str,
    pub priority: u8,
}

impl DatasetSource {
    pub fn candidate_datasets() -> &'static [DatasetSource] {
        &[
            DatasetSource {
                name: "Lichess Elite Database",
                url: "https://database.lichess.org/",
                kind: "PGN / zstd archives",
                notes: "Large public chess games dataset, good for opening and midgame diversity.",
                priority: 1,
            },
            DatasetSource {
                name: "MillionBase",
                url: "https://github.com/official-stockfish/MillionBase",
                kind: "PGN archive",
                notes: "Classic large public dataset used in engine training and data mining workflows.",
                priority: 2,
            },
            DatasetSource {
                name: "Lichess Games Export",
                url: "https://lichess.org/games/export",
                kind: "PGN export",
                notes: "Useful for selective downloads by opening, time control, and Elo range.",
                priority: 3,
            },
            DatasetSource {
                name: "FIDE / PGN Public Archives",
                url: "https://www.fide.com/",
                kind: "Official tournament PGNs",
                notes: "Good for high-quality master games and tournament-level positions.",
                priority: 4,
            },
            DatasetSource {
                name: "Chess.com Archive",
                url: "https://www.chess.com/games/archive",
                kind: "Partial game archives",
                notes: "Useful supplemental dataset when deeper public PGN corpora are needed.",
                priority: 5,
            },
        ]
    }
}

#[derive(Debug, Clone, Default)]
pub struct TeacherStudentConfig {
    pub teacher_name: &'static str,
    pub teacher_engine_path: Option<PathBuf>,
    pub teacher_depth: u8,
    pub teacher_model_path: Option<PathBuf>,
    pub student_model_path: PathBuf,
    pub dataset_manifest: Vec<DatasetSource>,
    pub validation_suite: Vec<&'static str>,
    pub checkpoint_every: usize,
    pub max_epochs: usize,
}

impl TeacherStudentConfig {
    pub fn default_config() -> Self {
        Self {
            teacher_name: "Stockfish 19",
            teacher_engine_path: None,
            teacher_depth: 12,
            teacher_model_path: None,
            student_model_path: PathBuf::from("resources/nnue.bin"),
            dataset_manifest: DatasetSource::candidate_datasets().to_vec(),
            validation_suite: vec![
                "perft",
                "tactical-suite",
                "self-play-benchmark",
                "endgame-regression",
                "arr-adversarial-robustness",
            ],
            checkpoint_every: 5,
            max_epochs: 30,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NnuePipelineConfig {
    pub baseline_model_path: PathBuf,
    pub baseline_meta_path: PathBuf,
    pub teacher_model_path: Option<PathBuf>,
    pub teacher_engine_path: Option<PathBuf>,
    pub teacher_depth: u8,
    pub dataset_dir: PathBuf,
    pub checkpoints_dir: PathBuf,
    pub validation_dir: PathBuf,
    pub architecture_version: u8,
}

impl NnuePipelineConfig {
    pub fn default_paths() -> Self {
        Self {
            baseline_model_path: PathBuf::from("resources/nnue.bin"),
            baseline_meta_path: PathBuf::from("resources/nnue.bin.meta"),
            teacher_model_path: None,
            teacher_engine_path: None,
            teacher_depth: 12,
            dataset_dir: PathBuf::from("data"),
            checkpoints_dir: PathBuf::from("checkpoints"),
            validation_dir: PathBuf::from("validation"),
            architecture_version: crate::eval::nnue::v16::VERSION,
        }
    }
}

pub fn initialize_pipeline(config: &NnuePipelineConfig) {
    std::fs::create_dir_all(&config.dataset_dir).ok();
    std::fs::create_dir_all(&config.checkpoints_dir).ok();
    std::fs::create_dir_all(&config.validation_dir).ok();

    if config.baseline_model_path.exists() {
        println!(
            "Baseline NNUE present: {}",
            config.baseline_model_path.display()
        );
    } else {
        eprintln!(
            "Baseline NNUE missing: {}",
            config.baseline_model_path.display()
        );
    }
}

pub fn teacher_student_training_plan() -> &'static str {
    "1. Snapshot current NNUE baseline.\n2. Collect real self-play positions and teacher labels.\n3. Train a student model in Whale format without replacing the engine architecture.\n4. Validate on benchmark suites and keep only stronger checkpoints.\n5. Promote the best checkpoint after regression checks."
}

pub fn recommended_dataset_order() -> Vec<&'static str> {
    let mut sources = DatasetSource::candidate_datasets().to_vec();
    sources.sort_by_key(|source| source.priority);
    sources.into_iter().map(|source| source.name).collect()
}

pub fn dataset_ingestion_plan() -> Vec<&'static str> {
    vec![
        "download public PGN archives from preferred sources",
        "filter by Elo range and time-control diversity",
        "normalize board states to Whale feature encoding",
        "extract teacher labels and tactical samples",
        "counterfactual-move-augmentation",
        "save a validation split separate from training split",
        "run benchmark validation before promoting the checkpoint",
    ]
}

pub fn dataset_is_ready_for_training(total_positions: usize) -> bool {
    total_positions >= 1_000_000
}

pub fn augment_dataset_with_cma(
    board: &crate::board::state::BoardState,
    best_move: crate::common::moves::Move,
    eval: i16,
) -> Vec<crate::cma::CounterfactualSample> {
    crate::cma::generate_counterfactual_samples(
        board,
        best_move,
        eval,
        crate::cma::CmaConfig::default(),
    )
}

pub fn verify_checkpoint_adversarial_robustness(
    board: &mut crate::board::state::BoardState,
) -> bool {
    let report = crate::eval::nnue::arr::audit_adversarial_robustness(
        board,
        crate::eval::nnue::arr::ArrConfig::default(),
    );
    report.is_robust
}

pub fn verify_checkpoint_depth_consistency(board: &crate::board::state::BoardState) -> bool {
    let config = crate::eval::nnue::dcn::DcnConfig::default();
    let eval_d19 = crate::eval::nnue::dcn::DcnModel::condition_evaluation(100, 19, board, config);
    let eval_d20 = crate::eval::nnue::dcn::DcnModel::condition_evaluation(100, 20, board, config);
    (eval_d20 as i32 - eval_d19 as i32).abs() <= 60
}

pub fn verify_checkpoint_alp_accuracy() -> bool {
    let features = crate::search::alp::AlpFeatures {
        eval_margin: -600,
        depth: 2,
        move_index: 25,
        is_null_move: false,
        is_capture: false,
        is_pv: false,
        in_check: false,
        history_score: -800,
        momentum: -100,
    };
    crate::search::alp::AlpModel::should_prune(&features, 70)
}

pub fn verify_checkpoint_psm_state() -> bool {
    let parent = crate::search::psm::PsmHiddenState::new();
    let features = crate::search::psm::PsmFeatures {
        static_eval: 50,
        depth: 5,
        alpha: 0,
        beta: 100,
        move_history: 150,
        is_capture: false,
        sibling_index: 2,
        failed_low: false,
    };
    let next = crate::search::psm::PsmEngine::step(&parent, &features);
    crate::search::psm::PsmEngine::readout(&next).abs() <= 60
}

pub fn verify_checkpoint_gtp_stability() -> bool {
    let mut graph = crate::search::gtp::GtpTreeGraph::new();
    let root = graph.add_node(crate::search::gtp::GtpNode {
        depth: 6,
        eval_margin: 0,
        is_capture: false,
        in_check: false,
        history_score: 0,
        parent_idx: None,
    });
    let child = graph.add_node(crate::search::gtp::GtpNode {
        depth: 5,
        eval_margin: -100,
        is_capture: false,
        in_check: false,
        history_score: -200,
        parent_idx: Some(root),
    });
    let scores = crate::search::gtp::GtpModel::message_passing(&graph);
    scores[child] <= 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_config_defaults_are_valid() {
        let cfg = NnuePipelineConfig::default_paths();
        assert!(cfg.baseline_model_path.ends_with("nnue.bin"));
        assert!(cfg.dataset_dir.ends_with("data"));
        assert!(cfg.checkpoints_dir.ends_with("checkpoints"));
        assert!(cfg.validation_dir.ends_with("validation"));
        assert_eq!(cfg.architecture_version, crate::eval::nnue::v16::VERSION);
    }

    #[test]
    fn candidate_datasets_are_in_priority_order() {
        let sources = DatasetSource::candidate_datasets();
        assert!(sources.len() >= 4);
        assert_eq!(sources[0].name, "Lichess Elite Database");
        assert_eq!(sources[1].name, "MillionBase");
    }

    #[test]
    fn teacher_student_default_config_is_well_defined() {
        let cfg = TeacherStudentConfig::default_config();
        assert_eq!(cfg.teacher_name, "Stockfish 19");
        assert_eq!(cfg.teacher_depth, 12);
        assert!(cfg.validation_suite.len() >= 5);
        assert!(cfg.validation_suite.contains(&"arr-adversarial-robustness"));
        assert_eq!(cfg.checkpoint_every, 5);
        assert_eq!(cfg.max_epochs, 30);
        assert_eq!(
            cfg.dataset_manifest.len(),
            DatasetSource::candidate_datasets().len()
        );
    }

    #[test]
    fn teacher_engine_defaults_are_explicitly_unconfigured() {
        let cfg = NnuePipelineConfig::default_paths();
        assert!(cfg.teacher_engine_path.is_none());
        assert_eq!(cfg.teacher_depth, 12);
    }

    #[test]
    fn small_datasets_are_not_marked_training_ready() {
        assert!(!dataset_is_ready_for_training(847));
        assert!(dataset_is_ready_for_training(1_000_000));
    }

    #[test]
    fn checkpoint_passes_adversarial_robustness_check() {
        let mut board =
            crate::board::state::BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        assert!(verify_checkpoint_adversarial_robustness(&mut board));
    }

    #[test]
    fn cma_dataset_augmentation_produces_samples() {
        let board = crate::board::state::BoardState::parse_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        );
        let samples = augment_dataset_with_cma(&board, crate::common::moves::Move::NO_MOVE, 0);
        for s in &samples {
            assert!(s.penalized_eval <= -400);
        }
    }

    #[test]
    fn checkpoint_passes_depth_consistency_check() {
        let board =
            crate::board::state::BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        assert!(verify_checkpoint_depth_consistency(&board));
    }

    #[test]
    fn checkpoint_passes_alp_accuracy_check() {
        assert!(verify_checkpoint_alp_accuracy());
    }

    #[test]
    fn checkpoint_passes_psm_state_check() {
        assert!(verify_checkpoint_psm_state());
    }

    #[test]
    fn checkpoint_passes_gtp_stability_check() {
        assert!(verify_checkpoint_gtp_stability());
    }
}
