//! SFNNv16 trainer (single big net) on the old bullet value API.
//!
//! Structural match with Stockfish SFNNv16:
//! - inputs: HalfKAv2_hm (22_528) ++ FullThreats (59_808) ++ PP_3Wide (4_560).
//! - feature emission is paired per element: one `f(white_idx, black_idx)`
//!   call per piece / threat / pawn-pair (the framework fills stm/ntm tensors).
//! - buckets: 8 material buckets `(popcount - 1) / 4`, selected in-graph.
//! - transformer with pairwise products of accumulator halves (/512).
//! - per-bucket fc_0 (1024 -> 32), pair [sqr, clip] (64 -> 32) on fc_1,
//!   fc_2 (128 -> 1) plus the forwarded fc_0[30] - fc_0[31] term.
//! - PSQT as a second output (no bias) with a small auxiliary loss.
//!
//! CONVERTER CONTRACT (GPU session): bullet checkpoints use float-friendly
//! scales (see save_format below). A converter must map them to the exact
//! SF .nnue layout (version + file/transformer/arch hashes, LEB128 sections,
//! raw threat then pair weights, split PSQ/threat/pair columns, i8 threat and
//! pair weights, 8 arch stacks, no trailing bytes) and then verify with a
//! roundtrip test comparing converted-net evals against
//! `ValueTrainer::eval_raw_output` on test FENs.
//! Known calibration points: transformer QA=255, linears QB=64, the fixed
//! /512 product divisor, fwd scale 9600/16384, PSQT save scale 16 with aux
//! target `sigmoid(psqt / 600)`. Bullet save fails hard on out-of-range
//! values, so per-weight clipping ranges must be set before a long run.

use bullet_lib::game::inputs::SparseInputType;
use bullet_lib::game::outputs::OutputBuckets;
use bullet_lib::nn::optimiser::AdamWOptimiser;
use bullet_lib::value::ValueTrainer;
use bullet_lib::{
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{SfBinpackLoader, sfbinpack::TrainingDataEntry},
    },
};
use bulletformat::ChessBoard;

use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::eval::nnue::v16::{
    BIG_INPUT_DIMS, FC0_OUT, FC1_IN, FC1_OUT, L1 as BIG_L1, MAX_ACTIVE as BIG_MAX_ACTIVE,
    N_BUCKETS, PSQ_DIMS, SfnnPosition, THREAT_DIMS, for_each_pair, for_each_threat, halfka_index,
    pair_index_for, threat_index_for,
};

const PSQT_AUX_WEIGHT: f32 = 0.1;
const PSQT_NORM: f32 = 600.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct SfnnBuckets;

impl OutputBuckets<ChessBoard> for SfnnBuckets {
    const BUCKETS: usize = N_BUCKETS;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        // Stockfish bucket formula (NOT bullet's MaterialCount).
        ((pos.occ().count_ones() as usize).saturating_sub(1) / 4).min(N_BUCKETS - 1) as u8
    }
}

fn chessboard_to_sfnn(pos: &ChessBoard) -> SfnnPosition {
    // Bullet squares are A1=0, identical to Stockfish numbering: no conversion.
    let mut pieces = [0u64; 6];
    let mut white = 0u64;
    let mut black = 0u64;
    let mut mapping = [6u8; 64];
    for (piece_u8, sq) in (*pos).into_iter() {
        let ptype = (piece_u8 & 7) as usize;
        if ptype > 5 {
            continue;
        }
        let sq = sq as usize;
        if sq >= 64 {
            continue;
        }
        let bit = 1u64 << sq;
        pieces[ptype] |= bit;
        if piece_u8 & 8 == 0 {
            white |= bit;
        } else {
            black |= bit;
        }
        mapping[sq] = ptype as u8;
    }
    SfnnPosition {
        pieces,
        white,
        black,
        mapping,
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Sfnn16BigInput;

impl SparseInputType for Sfnn16BigInput {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        BIG_INPUT_DIMS
    }

    fn max_active(&self) -> usize {
        BIG_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let sfnn = chessboard_to_sfnn(pos);
        let w_ksq = sfnn.king_square(Side::White);
        let b_ksq = sfnn.king_square(Side::Black);
        // One paired call per piece (framework contract).
        let mut bb = sfnn.white | sfnn.black;
        while bb != 0 {
            let s = bb.trailing_zeros() as usize;
            bb &= bb - 1;
            let side = if (sfnn.white >> s) & 1 == 1 {
                Side::White
            } else {
                Side::Black
            };
            let pt = sfnn.mapping[s] as usize;
            if pt > 5 {
                continue;
            }
            let piece = Piece::ALL[pt];
            if let (Some(w), Some(b)) = (
                halfka_index(Side::White, side, piece, s, w_ksq),
                halfka_index(Side::Black, side, piece, s, b_ksq),
            ) {
                f(w, b);
            }
        }
        // One paired call per threat.
        for_each_threat(&sfnn, |attacker, from, to, attacked| {
            if let (Some(w), Some(b)) = (
                threat_index_for(Side::White, attacker, from, to, attacked, w_ksq),
                threat_index_for(Side::Black, attacker, from, to, attacked, b_ksq),
            ) {
                f(PSQ_DIMS + w, PSQ_DIMS + b);
            }
        });
        // One paired call per pawn-pair.
        for_each_pair(&sfnn, |color, from, to, paired| {
            let w = pair_index_for(Side::White, color, from, to, paired, w_ksq);
            let b = pair_index_for(Side::Black, color, from, to, paired, b_ksq);
            f(PSQ_DIMS + THREAT_DIMS + w, PSQ_DIMS + THREAT_DIMS + b);
        });
    }

    fn shorthand(&self) -> String {
        "sfnn16-big-86896".to_string()
    }

    fn description(&self) -> String {
        "SFNNv16 big inputs 86896 (HalfKA 22528 + threats 59808 + pairs 4560)".to_string()
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
    let settings = if smoke_mode {
        build_smoke_settings()
    } else {
        build_settings()
    };
    println!(
        "Starting bullet training loop (big net){}...",
        if smoke_mode { " (smoke mode)" } else { "" }
    );
    let mut big = build_big_trainer();
    let dataloader = build_dataloader(dataset_path);
    big.run(
        &build_schedule(BIG_NET_ID, smoke_mode),
        &settings,
        &dataloader,
    );
    println!("Bullet training completed successfully!");
    if smoke_mode {
        println!("Smoke training finished; keeping production NNUE unchanged.");
    } else {
        copy_trained_weights();
    }
}

const DEFAULT_DATASET_PATH: &str = "data/v1_gen3_1m_d7.binpack";
const OUTPUT_DIRECTORY: &str = "checkpoints";
const BIG_KEEPER_PATH: &str = "resources/sfnn16-big-checkpoint.bin";
const INITIAL_LR: f32 = 0.001;
const FINAL_LR: f32 = 0.00001;
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.7;
const EVAL_SCALE: f32 = 400.0;
const BIG_NET_ID: &str = "rudim-sfnn16-big";
const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;
const START_SUPERBATCH: usize = 1;
const END_SUPERBATCH: usize = 40;
const SAVE_RATE: usize = 5;
const THREADS: usize = 4;
const BATCH_QUEUE_SIZE: usize = 4;
const DATALOADER_PER_THREAD_BUFFERS: usize = 512;

type BigTrainer = ValueTrainer<AdamWOptimiser, Sfnn16BigInput, SfnnBuckets>;

fn sfnn_save_format(prefix: &str, psqt_scale: i32) -> Vec<SavedFormat> {
    vec![
        SavedFormat::id(&format!("{prefix}l0w"))
            .round()
            .quantise::<i16>(255),
        SavedFormat::id(&format!("{prefix}l0b"))
            .round()
            .quantise::<i16>(255),
        SavedFormat::id(&format!("{prefix}l1w"))
            .round()
            .quantise::<i8>(64),
        SavedFormat::id(&format!("{prefix}l1b"))
            .round()
            .quantise::<i32>(255 * 64),
        SavedFormat::id(&format!("{prefix}l2w"))
            .round()
            .quantise::<i8>(64),
        SavedFormat::id(&format!("{prefix}l2b"))
            .round()
            .quantise::<i32>(255 * 64),
        SavedFormat::id(&format!("{prefix}outw"))
            .round()
            .quantise::<i8>(64),
        SavedFormat::id(&format!("{prefix}outb"))
            .round()
            .quantise::<i32>(255 * 64),
        SavedFormat::id(&format!("{prefix}psqt"))
            .round()
            .quantise::<i32>(psqt_scale),
    ]
}

fn build_big_trainer() -> BigTrainer {
    ValueTrainerBuilder::default()
        .dual_perspective()
        .output_buckets(SfnnBuckets)
        .optimiser(AdamW)
        .inputs(Sfnn16BigInput)
        .save_format(&sfnn_save_format("", 16))
        .build_custom(|builder, (stm_inputs, ntm_inputs, buckets), targets| {
            let l0 = builder.new_affine("l0", BIG_INPUT_DIMS, BIG_L1);
            let l1 = builder.new_affine("l1", BIG_L1, N_BUCKETS * FC0_OUT);
            let l2 = builder.new_affine("l2", FC1_IN, FC1_OUT);
            let out = builder.new_affine("out", FC1_OUT, 1);
            let psqt_w = builder.new_weights(
                "psqt",
                bullet_lib::nn::Shape::new(N_BUCKETS, BIG_INPUT_DIMS),
                bullet_lib::nn::InitSettings::Zeroed,
            );

            // Transformer with pairwise products of accumulator halves.
            let stm_p = l0.forward(stm_inputs).max(0.0).min(1.0).pairwise_mul();
            let ntm_p = l0.forward(ntm_inputs).max(0.0).min(1.0).pairwise_mul();
            let trans = stm_p.concat(ntm_p);

            // Per-bucket fc_0, pair [sqr, clip] over all 32 outputs.
            let fc0 = l1.forward(trans).select(buckets);
            let pair = fc0.abs_pow(2.0).crelu().concat(fc0.crelu());
            let fc1 = l2.forward(pair).crelu();
            let final_out = out.forward(fc1);

            // PSQT second output (stm - ntm over the selected bucket).
            let psqt_stm = psqt_w.matmul(stm_inputs).select(buckets);
            let psqt_ntm = psqt_w.matmul(ntm_inputs).select(buckets);
            let psqt_out = (psqt_stm - psqt_ntm) / 2.0;

            let main = final_out.sigmoid().squared_error(targets);
            let aux = (psqt_out / PSQT_NORM).sigmoid().squared_error(targets);
            (final_out, main + aux * PSQT_AUX_WEIGHT)
        })
}

fn build_schedule(
    net_id: &str,
    smoke: bool,
) -> TrainingSchedule<lr::CosineDecayLR, wdl::LinearWDL> {
    if smoke {
        return TrainingSchedule {
            net_id: format!("{}-smoke", net_id),
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
        };
    }
    TrainingSchedule {
        net_id: net_id.to_string(),
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

fn sf_filter(_: &TrainingDataEntry) -> bool {
    true
}

fn build_dataloader(dataset_path: &str) -> SfBinpackLoader<fn(&TrainingDataEntry) -> bool> {
    SfBinpackLoader::new(
        dataset_path,
        DATALOADER_PER_THREAD_BUFFERS,
        THREADS,
        sf_filter,
    )
}

fn copy_trained_weights() {
    std::fs::create_dir_all("resources").ok();
    let big_cp = format!("{}/{}-{}", OUTPUT_DIRECTORY, BIG_NET_ID, END_SUPERBATCH);
    let big_cp = format!("{}/quantised.bin", big_cp);
    println!("Copying big weights from {} to {}", big_cp, BIG_KEEPER_PATH);
    if let Err(e) = std::fs::copy(&big_cp, BIG_KEEPER_PATH) {
        eprintln!("Error copying big weights: {}", e);
    }
}
