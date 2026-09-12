//! SFNNv10 trainers (big + small) on the old bullet value API.
//!
//! Structural match with Stockfish SFNNv10 (commit 8e5392d):
//! - inputs: big = HalfKAv2_hm (22_528) ++ FullThreats (79_856); small = HalfKA.
//! - feature emission is paired per element: one `f(white_idx, black_idx)`
//!   call per piece / threat (the framework fills stm/nstm tensors from it).
//! - buckets: 8 material buckets `(popcount - 1) / 4`, selected in-graph.
//! - transformer with pairwise products of accumulator halves.
//! - fc_0 per bucket (1024/128 -> 16), pair [sqr, clip] on the first 15,
//!   forwarded fc_0[15] output term, fc_1 (30 -> 32) + clip, fc_2 (32 -> 1).
//! - PSQT as a second output (no bias) with a small auxiliary loss.
//!
//! CONVERTER CONTRACT (GPU session): bullet checkpoints use float-friendly
//! scales (see save_format below). A converter must map them to the exact
//! SF .nnue layout (LEB128, split PSQ/threat columns, i8 threat weights,
//! 8 arch stacks, hashes) and then verify with a roundtrip test comparing
//! converted-net evals against `ValueTrainer::eval_raw_output` on test FENs.
//! Known calibration points: transformer QA=255, linears QB=64, the fixed
//! /512 product divisor, fwd scale 9600/8128, PSQT save scale 16 with aux
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
        loader::{ViriBinpackLoader, viribinpack::Filter},
    },
};
use bulletformat::ChessBoard;

use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::eval::nnue::v16::{
    BIG_INPUT_DIMS, FC0_ACT, FC0_OUT, FC1_IN, FC1_OUT, N_BUCKETS, PSQ_DIMS,
    SMALL_MAX_ACTIVE, SfnnPosition, for_each_threat, halfka_index, threat_index_for,
    L1 as BIG_L1, MAX_ACTIVE as BIG_MAX_ACTIVE,
};

const SMALL_L1: usize = 128;

const PSQT_AUX_WEIGHT: f32 = 0.1;
const PSQT_NORM: f32 = 600.0;
const FWD_SCALE: f32 = 9600.0 / 8128.0;

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
struct Sfnn10BigInput;

impl SparseInputType for Sfnn10BigInput {
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
    }

    fn shorthand(&self) -> String {
        "sfnn16-big-102384".to_string()
    }

    fn description(&self) -> String {
        "SFNNv10 big inputs 102384 (HalfKA 22528 + threats 79856)".to_string()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Sfnn10SmallInput;

impl SparseInputType for Sfnn10SmallInput {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        PSQ_DIMS
    }

    fn max_active(&self) -> usize {
        SMALL_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        let sfnn = chessboard_to_sfnn(pos);
        let w_ksq = sfnn.king_square(Side::White);
        let b_ksq = sfnn.king_square(Side::Black);
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
    }

    fn shorthand(&self) -> String {
        "sfnn16-small-22528".to_string()
    }

    fn description(&self) -> String {
        "SFNNv10 small inputs 22528 (HalfKA only)".to_string()
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
    println!(
        "Starting bullet training loop (small net){}...",
        if smoke_mode { " (smoke mode)" } else { "" }
    );
    let mut small = build_small_trainer();
    let dataloader = build_dataloader(dataset_path);
    small.run(
        &build_schedule(SMALL_NET_ID, smoke_mode),
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
const SMALL_KEEPER_PATH: &str = "resources/sfnn16-small-checkpoint.bin";
const INITIAL_LR: f32 = 0.001;
const FINAL_LR: f32 = 0.00001;
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.7;
const EVAL_SCALE: f32 = 400.0;
const BIG_NET_ID: &str = "rudim-sfnn16-big";
const SMALL_NET_ID: &str = "rudim-sfnn16-small";
const BATCH_SIZE: usize = 16_384;
const BATCHES_PER_SUPERBATCH: usize = 6104;
const START_SUPERBATCH: usize = 1;
const END_SUPERBATCH: usize = 40;
const SAVE_RATE: usize = 5;
const THREADS: usize = 4;
const BATCH_QUEUE_SIZE: usize = 4;
const DATALOADER_PER_THREAD_BUFFERS: usize = 512;

type BigTrainer = ValueTrainer<AdamWOptimiser, Sfnn10BigInput, SfnnBuckets>;
type SmallTrainer = ValueTrainer<AdamWOptimiser, Sfnn10SmallInput, SfnnBuckets>;

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
        .inputs(Sfnn10BigInput)
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

            // Per-bucket fc_0, pair [sqr, clip] on the first 15, forward [15].
            let fc0 = l1.forward(trans).select(buckets);
            let fc0a = fc0.slice_rows(0, FC0_ACT);
            let pair = fc0a.abs_pow(2.0).crelu().concat(fc0a.crelu());
            let fwd = fc0.slice_rows(FC0_ACT, FC0_OUT);

            let fc1 = l2.forward(pair).crelu();
            let positional = out.forward(fc1);
            let final_out = positional + fwd * FWD_SCALE;

            // PSQT second output (stm - ntm over the selected bucket).
            let psqt_stm = psqt_w.matmul(stm_inputs).select(buckets);
            let psqt_ntm = psqt_w.matmul(ntm_inputs).select(buckets);
            let psqt_out = (psqt_stm - psqt_ntm) / 2.0;

            let main = final_out.sigmoid().squared_error(targets);
            let aux = (psqt_out / PSQT_NORM).sigmoid().squared_error(targets);
            (final_out, main + aux * PSQT_AUX_WEIGHT)
        })
}

fn build_small_trainer() -> SmallTrainer {
    ValueTrainerBuilder::default()
        .dual_perspective()
        .output_buckets(SfnnBuckets)
        .optimiser(AdamW)
        .inputs(Sfnn10SmallInput)
        .save_format(&sfnn_save_format("s", 16))
        .build_custom(|builder, (stm_inputs, ntm_inputs, buckets), targets| {
            let l0 = builder.new_affine("sl0", PSQ_DIMS, SMALL_L1);
            let l1 = builder.new_affine("sl1", SMALL_L1, N_BUCKETS * FC0_OUT);
            let l2 = builder.new_affine("sl2", FC1_IN, FC1_OUT);
            let out = builder.new_affine("sout", FC1_OUT, 1);
            let psqt_w = builder.new_weights(
                "spsqt",
                bullet_lib::nn::Shape::new(N_BUCKETS, PSQ_DIMS),
                bullet_lib::nn::InitSettings::Zeroed,
            );

            let stm_p = l0.forward(stm_inputs).max(0.0).min(1.0).pairwise_mul();
            let ntm_p = l0.forward(ntm_inputs).max(0.0).min(1.0).pairwise_mul();
            let trans = stm_p.concat(ntm_p);

            let fc0 = l1.forward(trans).select(buckets);
            let fc0a = fc0.slice_rows(0, FC0_ACT);
            let pair = fc0a.abs_pow(2.0).crelu().concat(fc0a.crelu());
            let fwd = fc0.slice_rows(FC0_ACT, FC0_OUT);

            let fc1 = l2.forward(pair).crelu();
            let positional = out.forward(fc1);
            let final_out = positional + fwd * FWD_SCALE;

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

fn build_dataloader(dataset_path: &str) -> ViriBinpackLoader {
    let filter = Filter::default();
    ViriBinpackLoader::new(dataset_path, DATALOADER_PER_THREAD_BUFFERS, THREADS, filter)
}

fn copy_trained_weights() {
    let big_cp = format!("{}/{}-{}", OUTPUT_DIRECTORY, BIG_NET_ID, END_SUPERBATCH);
    let big_cp = format!("{}/quantised.bin", big_cp);
    println!("Copying big weights from {} to {}", big_cp, BIG_KEEPER_PATH);
    if let Err(e) = std::fs::copy(&big_cp, BIG_KEEPER_PATH) {
        eprintln!("Error copying big weights: {}", e);
    }
    let small_cp = format!("{}/{}-{}", OUTPUT_DIRECTORY, SMALL_NET_ID, END_SUPERBATCH);
    let small_cp = format!("{}/quantised.bin", small_cp);
    println!(
        "Copying small weights from {} to {}",
        small_cp, SMALL_KEEPER_PATH
    );
    if let Err(e) = std::fs::copy(&small_cp, SMALL_KEEPER_PATH) {
        eprintln!("Error copying small weights: {}", e);
    }
}
