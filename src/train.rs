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
        ((pos.occ().count_ones() as usize).saturating_sub(1) / 4).min(N_BUCKETS - 1) as u8
    }
}

fn chessboard_to_sfnn(pos: &ChessBoard) -> SfnnPosition {
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

        for_each_threat(&sfnn, |attacker, from, to, attacked| {
            if let (Some(w), Some(b)) = (
                threat_index_for(Side::White, attacker, from, to, attacked, w_ksq),
                threat_index_for(Side::Black, attacker, from, to, attacked, b_ksq),
            ) {
                f(PSQ_DIMS + w, PSQ_DIMS + b);
            }
        });

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
const BIG_KEEPER_NAME: &str = "sfnn16-big-checkpoint.bin";
const BIG_KEEPER_PATH: &str = "resources/sfnn16-big-checkpoint.bin";
const INITIAL_LR: f32 = 0.001;
const FINAL_LR: f32 = 0.00001;
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.7;
const EVAL_SCALE: f32 = 400.0;
const BIG_NET_ID: &str = "whale-sfnn16-big";
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

fn fc0_activation_pair(fc0: bullet_lib::nn::ModelNode<'_>) -> bullet_lib::nn::ModelNode<'_> {
    let clipped = fc0.crelu();
    (clipped * clipped).concat(clipped)
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

            let stm_p = l0.forward(stm_inputs).max(0.0).min(1.0).pairwise_mul();
            let ntm_p = l0.forward(ntm_inputs).max(0.0).min(1.0).pairwise_mul();
            let trans = stm_p.concat(ntm_p);

            let fc0 = l1.forward(trans).select(buckets);
            let pair = fc0_activation_pair(fc0);
            let fc1 = l2.forward(pair).crelu();
            let final_out = out.forward(fc1);

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
        output_directory: output_directory(),
        batch_queue_size: BATCH_QUEUE_SIZE,
    }
}

fn build_smoke_settings() -> LocalSettings<'static> {
    LocalSettings {
        threads: 1,
        test_set: None,
        output_directory: output_directory(),
        batch_queue_size: 1,
    }
}

fn sf_filter(_: &TrainingDataEntry) -> bool {
    true
}

fn output_directory() -> &'static str {
    let dir = std::env::var("WHALE_OUT_DIR").unwrap_or_else(|_| OUTPUT_DIRECTORY.to_string());
    Box::leak(dir.into_boxed_str())
}

fn keeper_path() -> String {
    format!("{}/{}", output_directory(), BIG_KEEPER_NAME)
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
    let keeper = keeper_path();
    if let Some(parent) = std::path::Path::new(&keeper).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let big_cp = format!("{}/{}-{}", output_directory(), BIG_NET_ID, END_SUPERBATCH);
    let big_cp = format!("{}/quantised.bin", big_cp);
    println!("Copying big weights from {} to {}", big_cp, keeper);
    if let Err(e) = std::fs::copy(&big_cp, &keeper) {
        eprintln!("Error copying big weights: {}", e);
    }
    if keeper != BIG_KEEPER_PATH {
        std::fs::create_dir_all("resources").ok();
        if let Err(e) = std::fs::copy(&keeper, BIG_KEEPER_PATH) {
            eprintln!("Error syncing keeper to {}: {}", BIG_KEEPER_PATH, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;
    use crate::eval::nnue::v16::{append_halfka, append_pairs, append_threats, material_bucket};

    #[test]
    fn regression_fc0_activation_pair_matches_rudi_inference() {
        let cases = [
            -2.0,
            -1.0,
            -0.5,
            -f32::EPSILON,
            0.0,
            0.25,
            0.5,
            1.0,
            1.5,
            2.0,
        ];
        let values: [f32; FC0_OUT] = std::array::from_fn(|i| cases[i % cases.len()]);
        let builder = bullet_lib::nn::ModelBuilder::default();
        let input = builder.new_constant(bullet_lib::nn::Shape::new(FC0_OUT, 1), &values);
        let pair = builder.no_grad(|| fc0_activation_pair(input));
        assert_eq!(pair.shape().rows(), FC1_IN);
        assert_eq!(pair.shape().cols(), 1);
        let output = pair.detach();
        let graph = output.builder().build([output]);
        let evaluated = graph
            .evaluate(std::collections::BTreeMap::new())
            .unwrap()
            .unwrap();
        let actual = evaluated.get(&output.node()).unwrap().f32();
        assert_eq!(actual.len(), FC1_IN);
        for (i, value) in values.into_iter().enumerate() {
            let clipped = value.clamp(0.0, 1.0);
            assert_eq!(
                actual[i],
                clipped * clipped,
                "squared branch, input {value}"
            );
            assert_eq!(
                actual[FC0_OUT + i],
                clipped,
                "clipped branch, input {value}"
            );
        }
    }

    #[test]
    fn regression_training_features_match_inference() {
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1",
            "2k5/1p2p3/2n5/3P4/2B1P3/8/8/6K1 w - - 0 1",
        ] {
            let board = BoardState::parse_fen(fen);
            let pos = SfnnPosition::from_board(&board);
            let mut bbs = [0; 8];
            bbs[0] = pos.white;
            bbs[1] = pos.black;
            bbs[2..].copy_from_slice(&pos.pieces);
            for stm in [Side::White, Side::Black] {
                let data = ChessBoard::from_raw(bbs, stm as usize, 0, 0.5).unwrap();
                let mut actual = [Vec::new(), Vec::new()];
                Sfnn16BigInput.map_features(&data, |us, them| {
                    actual[0].push(us);
                    actual[1].push(them);
                });
                for (slot, perspective) in [stm, stm.other()].into_iter().enumerate() {
                    let mut expected = Vec::new();
                    append_halfka(&pos, perspective, &mut expected);
                    append_threats(&pos, perspective, &mut expected);
                    append_pairs(&pos, perspective, &mut expected);
                    expected.sort_unstable();
                    actual[slot].sort_unstable();
                    assert_eq!(actual[slot], expected, "{fen}, {stm:?}, slot {slot}");
                    assert!(actual[slot].len() <= Sfnn16BigInput.max_active());
                    assert!(
                        actual[slot]
                            .iter()
                            .all(|&f| f < Sfnn16BigInput.num_inputs())
                    );
                }
                assert_eq!(
                    SfnnBuckets.bucket(&data) as usize,
                    material_bucket(pos.piece_count())
                );
            }
        }
    }
}
