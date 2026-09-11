use std::fs;
use bullet_lib::game::inputs::SparseInputType;
use bulletformat::ChessBoard;
use bullet_lib::nn::optimiser::AdamWOptimiser;
use bullet_lib::value::{NoOutputBuckets, ValueTrainer};
use bullet_lib::{
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{ViriBinpackLoader, viribinpack::Filter},
    },
};
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::eval::nnue::v2::{
    HIDDEN_SIZE, MAX_ACTIVE_FEATURES, NetworkHeader, RAW_INPUT_SIZE, RawBoard,
    TRANSFORMED_SIZE, active_features_raw,
};

#[derive(Clone, Copy, Debug, Default)]
struct RudimV2Input;

fn chessboard_to_rawboard(pos: &ChessBoard) -> RawBoard {
    let mut raw = RawBoard::empty();
    for (piece_u8, sq_bullet) in (*pos).into_iter() {
        let ptype = (piece_u8 & 7) as usize;
        if ptype > 5 {
            continue;
        }
        let side = if piece_u8 & 8 == 0 {
            Side::White
        } else {
            Side::Black
        };
        let piece = Piece::from(ptype);
        let sq_rudim = (sq_bullet ^ 56) as usize;
        if sq_rudim >= 64 {
            continue;
        }
        let bit = 1u64 << sq_rudim;
        raw.pieces[ptype] |= bit;
        match side {
            Side::White => raw.white |= bit,
            Side::Black => raw.black |= bit,
            Side::Both => {}
        }
        raw.mapping[sq_rudim] = piece;
    }
    raw
}

impl SparseInputType for RudimV2Input {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        RAW_INPUT_SIZE
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let raw = chessboard_to_rawboard(pos);
        let stm = active_features_raw(&raw, Side::White);
        let ntm = active_features_raw(&raw, Side::Black);
        let n = if stm.len < ntm.len { stm.len } else { ntm.len };
        for i in 0..n {
            f(stm.indices[i], ntm.indices[i]);
        }
    }

    fn shorthand(&self) -> String {
        "rudim-v2-88944x1024".to_string()
    }

    fn description(&self) -> String {
        "Rudim v2 raw inputs 88944".to_string()
    }
}

pub fn run(custom_dataset_path: Option<&str>) {
    run_with_mode(custom_dataset_path, false);
}

pub fn run_smoke(custom_dataset_path: Option<&str>) {
    run_with_mode(custom_dataset_path, true);
}

fn run_with_mode(custom_dataset_path: Option<&str>, smoke_mode: bool) {
    let dataset_path = custom_dataset_path.unwrap_or(DEFAULT_DATASET_PATH);
    let mut trainer = build_trainer();
    let schedule = if smoke_mode {
        build_smoke_schedule()
    } else {
        build_schedule()
    };
    let settings = if smoke_mode {
        build_smoke_settings()
    } else {
        build_settings()
    };
    let dataloader = build_dataloader(dataset_path);
    println!("Starting bullet training loop{}...", if smoke_mode { " (smoke mode)" } else { "" });
    trainer.run(&schedule, &settings, &dataloader);
    println!("Bullet training completed successfully!");
    if smoke_mode {
        println!("Smoke training finished; keeping production NNUE unchanged.");
    } else {
        copy_trained_weights();
    }
}

const DEFAULT_DATASET_PATH: &str = "data/v1_gen3_1m_d7.binpack";
const OUTPUT_DIRECTORY: &str = "checkpoints";
const TARGET_WEIGHTS_PATH: &str = "resources/nnue-v2-bullet.bin";
const INITIAL_LR: f32 = 0.001;
const FINAL_LR: f32 = 0.00001;
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.7;
const EVAL_SCALE: f32 = 400.0;
const NET_ID: &str = "rudim-256";
const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;
const START_SUPERBATCH: usize = 1;
const END_SUPERBATCH: usize = 40;
const SAVE_RATE: usize = 5;
const THREADS: usize = 4;
const BATCH_QUEUE_SIZE: usize = 4;
const DATALOADER_PER_THREAD_BUFFERS: usize = 512;

fn build_trainer() -> ValueTrainer<AdamWOptimiser, RudimV2Input, NoOutputBuckets> {
    let header = NetworkHeader::current().to_bytes().to_vec();
    ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(RudimV2Input)
        .save_format(&[
            SavedFormat::custom(header),
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i8>(64),
            SavedFormat::id("l1b").round().quantise::<i32>(255 * 64),
            SavedFormat::id("l2w").round().quantise::<i8>(64),
            SavedFormat::id("l2b").round().quantise::<i32>(255 * 64),
            SavedFormat::id("outw").round().quantise::<i8>(64),
            SavedFormat::id("outb").round().quantise::<i32>(255 * 64),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", RAW_INPUT_SIZE, TRANSFORMED_SIZE);
            let l1 = builder.new_affine("l1", 2 * TRANSFORMED_SIZE, HIDDEN_SIZE);
            let l2 = builder.new_affine("l2", 2 * HIDDEN_SIZE, HIDDEN_SIZE);
            let output = builder.new_affine("out", 2 * HIDDEN_SIZE, 1);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let first_hidden = l1.forward(stm_hidden.concat(ntm_hidden)).screlu();
            let second_hidden = l2.forward(first_hidden.concat(first_hidden)).screlu();
            output.forward(second_hidden.concat(second_hidden))
        })
}

fn build_schedule() -> TrainingSchedule<lr::CosineDecayLR, wdl::LinearWDL> {
    TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch: START_SUPERBATCH,
            end_superbatch: END_SUPERBATCH,
        },
        wdl_scheduler: wdl::LinearWDL {
            start: WDL_START,
            end: WDL_END,
        },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: INITIAL_LR,
            final_lr: FINAL_LR,
            final_superbatch: END_SUPERBATCH,
        },
        save_rate: SAVE_RATE,
    }
}

fn build_smoke_schedule() -> TrainingSchedule<lr::CosineDecayLR, wdl::LinearWDL> {
    TrainingSchedule {
        net_id: format!("{}-smoke", NET_ID),
        eval_scale: EVAL_SCALE,
        steps: TrainingSteps {
            batch_size: 256,
            batches_per_superbatch: 4,
            start_superbatch: 1,
            end_superbatch: 1,
        },
        wdl_scheduler: wdl::LinearWDL {
            start: WDL_START,
            end: WDL_END,
        },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.0005,
            final_lr: 0.00005,
            final_superbatch: 1,
        },
        save_rate: 1,
    }
}

fn build_settings() -> LocalSettings<'static> {
    LocalSettings {
        threads: THREADS,
        test_set: None,
        output_directory: OUTPUT_DIRECTORY,
        batch_queue_size: BATCH_QUEUE_SIZE,
    }
}

fn build_smoke_settings() -> LocalSettings<'static> {
    LocalSettings {
        threads: 1,
        test_set: None,
        output_directory: OUTPUT_DIRECTORY,
        batch_queue_size: 1,
    }
}

fn build_dataloader(dataset_path: &str) -> ViriBinpackLoader {
    let filter = Filter::default();
    ViriBinpackLoader::new(dataset_path, DATALOADER_PER_THREAD_BUFFERS, THREADS, filter)
}

fn copy_trained_weights() {
    let cp_dir = format!("{}/{}-{}", OUTPUT_DIRECTORY, NET_ID, END_SUPERBATCH);
    let cp_path = format!("{}/quantised.bin", cp_dir);
    println!(
        "Copying weights from {} to {}",
        cp_path, TARGET_WEIGHTS_PATH
    );
    if let Err(e) = fs::copy(&cp_path, TARGET_WEIGHTS_PATH) {
        eprintln!("Error copying weights: {}", e);
    } else {
        println!("Successfully copied weights to {}!", TARGET_WEIGHTS_PATH);
    }
}
