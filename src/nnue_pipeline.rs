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
            teacher_model_path: None,
            student_model_path: PathBuf::from("resources/nnue.bin"),
            dataset_manifest: DatasetSource::candidate_datasets().to_vec(),
            validation_suite: vec![
                "perft",
                "tactical-suite",
                "self-play-benchmark",
                "endgame-regression",
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
    pub dataset_dir: PathBuf,
    pub checkpoints_dir: PathBuf,
    pub validation_dir: PathBuf,
}

impl NnuePipelineConfig {
    pub fn default_paths() -> Self {
        Self {
            baseline_model_path: PathBuf::from("resources/nnue.bin"),
            baseline_meta_path: PathBuf::from("resources/nnue.bin.meta"),
            teacher_model_path: None,
            dataset_dir: PathBuf::from("data"),
            checkpoints_dir: PathBuf::from("checkpoints"),
            validation_dir: PathBuf::from("validation"),
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
    "1. Snapshot current NNUE baseline.\n2. Collect real self-play positions and teacher labels.\n3. Train a student model in Rudim format without replacing the engine architecture.\n4. Validate on benchmark suites and keep only stronger checkpoints.\n5. Promote the best checkpoint after regression checks."
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
        "normalize board states to Rudim feature encoding",
        "extract teacher labels and tactical samples",
        "save a validation split separate from training split",
        "run benchmark validation before promoting the checkpoint",
    ]
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
        assert!(cfg.validation_suite.len() >= 4);
        assert_eq!(cfg.checkpoint_every, 5);
        assert_eq!(cfg.max_epochs, 30);
        assert_eq!(cfg.dataset_manifest.len(), DatasetSource::candidate_datasets().len());
    }
}
