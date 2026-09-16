pub mod accumulator;
pub mod features;
pub mod loader;
pub mod v16;

use crate::board::state::BoardState;
use crate::common::side::Side;

use self::loader::Network;

pub const ACC_SIZE: usize = 256;
pub const INPUT_SIZE: usize = 768;

pub const SCALE: i32 = 400;

// Hash-keyed cache for the raw network score (before the fifty-move damp,
// which is not part of the board hash). The score is a pure function of the
// position, so caching it is exact. Thread-local: each search thread gets
// its own table, no locking on the hot path.
const EVAL_CACHE_BITS: u32 = 20;
const EVAL_CACHE_SIZE: usize = 1 << EVAL_CACHE_BITS;
const EVAL_CACHE_MASK: u64 = (EVAL_CACHE_SIZE as u64) - 1;

#[derive(Clone, Copy)]
struct EvalCacheEntry {
    hash: u64,
    score: i16,
}

std::thread_local! {
    static EVAL_CACHE: std::cell::RefCell<Box<[EvalCacheEntry]>> = std::cell::RefCell::new(
        vec![
            EvalCacheEntry {
                hash: u64::MAX,
                score: 0
            };
            EVAL_CACHE_SIZE
        ]
        .into_boxed_slice(),
    );
}

#[inline(always)]
fn probe_eval_cache(hash: u64) -> Option<i16> {
    EVAL_CACHE.with(|cache| {
        let entry = cache.borrow()[(hash & EVAL_CACHE_MASK) as usize];
        (entry.hash == hash).then_some(entry.score)
    })
}

#[inline(always)]
fn store_eval_cache(hash: u64, score: i16) {
    EVAL_CACHE.with(|cache| {
        cache.borrow_mut()[(hash & EVAL_CACHE_MASK) as usize] = EvalCacheEntry { hash, score };
    });
}

/// Drop all cached raw scores. Called when the active network changes so a
/// stale score from the previous net can never be served.
pub fn clear_eval_cache() {
    EVAL_CACHE.with(|cache| {
        cache.borrow_mut().fill(EvalCacheEntry {
            hash: u64::MAX,
            score: 0,
        });
    });
}

#[inline(always)]
pub fn evaluate(board: &mut BoardState) -> i16 {
    evaluate_with_optimism(board, 0)
}

#[inline(always)]
pub fn evaluate_with_optimism(board: &mut BoardState, optimism: i32) -> i16 {
    let score = if let Some(hit) = probe_eval_cache(board.board_hash) {
        hit
    } else {
        board.ensure_accumulators_fresh();
        let raw = if v16::maintenance_active() {
            if let Some(ev) = v16::evaluate_board_detailed(board) {
                let psqt = ev.psqt as i64;
                let positional = ev.positional as i64;
                let mut nnue = psqt + positional;
                let complexity = (psqt - positional).abs();
                let mut opt = optimism as i64;
                opt += opt * complexity / 476;
                nnue -= nnue * complexity / 18236;
                let pawn_cnt = board.pieces[crate::common::piece::Piece::Pawn]
                    .0
                    .count_ones() as i64;
                let knight_cnt = board.pieces[crate::common::piece::Piece::Knight]
                    .0
                    .count_ones() as i64;
                let bishop_cnt = board.pieces[crate::common::piece::Piece::Bishop]
                    .0
                    .count_ones() as i64;
                let rook_cnt = board.pieces[crate::common::piece::Piece::Rook]
                    .0
                    .count_ones() as i64;
                let queen_cnt = board.pieces[crate::common::piece::Piece::Queen]
                    .0
                    .count_ones() as i64;
                let non_pawn_mat =
                    knight_cnt * 300 + bishop_cnt * 300 + rook_cnt * 500 + queen_cnt * 900;
                let material = 534 * pawn_cnt + non_pawn_mat;
                let mut v = nnue + (nnue * material + opt * 7675) / 91000;
                v -= v * board.half_move_clock as i64 / 199;
                v.clamp(-29000, 29000) as i16
            } else {
                v16::evaluate_board(board).unwrap_or(0)
            }
        } else {
            evaluate_fast(board, optimism)
        };
        store_eval_cache(board.board_hash, raw);
        raw
    };

    score.clamp(-29000, 29000)
}

#[inline(always)]
pub fn evaluate_fast(board: &mut BoardState, optimism: i32) -> i16 {
    board.ensure_accumulators_fresh();
    let network = Network::get_embedded();
    let s = evaluate_internal(board, network);
    let pawn_cnt = board.pieces[crate::common::piece::Piece::Pawn]
        .0
        .count_ones() as i64;
    let knight_cnt = board.pieces[crate::common::piece::Piece::Knight]
        .0
        .count_ones() as i64;
    let bishop_cnt = board.pieces[crate::common::piece::Piece::Bishop]
        .0
        .count_ones() as i64;
    let rook_cnt = board.pieces[crate::common::piece::Piece::Rook]
        .0
        .count_ones() as i64;
    let queen_cnt = board.pieces[crate::common::piece::Piece::Queen]
        .0
        .count_ones() as i64;
    let non_pawn_mat =
        knight_cnt * 300 + bishop_cnt * 300 + rook_cnt * 500 + queen_cnt * 900;
    let material = 534 * pawn_cnt + non_pawn_mat;
    let opt = optimism as i64;
    let mut v = s as i64 + (s as i64 * material + opt * 7675) / 91000;
    if board.half_move_clock > 0 {
        v -= v * board.half_move_clock as i64 / 199;
    }
    v.clamp(-29000, 29000) as i16
}

#[inline(always)]
pub fn evaluate_internal(board: &BoardState, network: &Network) -> i16 {
    let side_to_move = board.side_to_move;
    let (acc_active, acc_passive) = if side_to_move == Side::White {
        (
            &board.history.accumulators[board.history.index].white,
            &board.history.accumulators[board.history.index].black,
        )
    } else {
        (
            &board.history.accumulators[board.history.index].black,
            &board.history.accumulators[board.history.index].white,
        )
    };

    let mut output: i64 = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                output +=
                    evaluate_side_avx2(&acc_active.state, &network.output_weights[0..ACC_SIZE]);
                output += evaluate_side_avx2(
                    &acc_passive.state,
                    &network.output_weights[ACC_SIZE..2 * ACC_SIZE],
                );
            }
        } else {
            for (&input, &weight) in acc_active
                .state
                .iter()
                .zip(&network.output_weights[0..ACC_SIZE])
            {
                let val = i64::from(input).clamp(0, 255);
                let screlu = val * val;
                output += screlu * i64::from(weight);
            }

            for (&input, &weight) in acc_passive
                .state
                .iter()
                .zip(&network.output_weights[ACC_SIZE..2 * ACC_SIZE])
            {
                let val = i64::from(input).clamp(0, 255);
                let screlu = val * val;
                output += screlu * i64::from(weight);
            }
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        for (&input, &weight) in acc_active
            .state
            .iter()
            .zip(&network.output_weights[0..ACC_SIZE])
        {
            let val = i64::from(input).clamp(0, 255);
            let screlu = val * val;
            output += screlu * i64::from(weight);
        }

        for (&input, &weight) in acc_passive
            .state
            .iter()
            .zip(&network.output_weights[ACC_SIZE..2 * ACC_SIZE])
        {
            let val = i64::from(input).clamp(0, 255);
            let screlu = val * val;
            output += screlu * i64::from(weight);
        }
    }

    output /= 255;
    output += i64::from(network.output_bias);
    output *= i64::from(SCALE);
    output /= 255 * 64;

    output.clamp(-29000, 29000) as i16
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn evaluate_side_avx2(state: &[i16; ACC_SIZE], weights: &[i16]) -> i64 {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi16(255);
        let mut total: i64 = 0;

        let s_ptr = state.as_ptr() as *const __m256i;
        let w_ptr = weights.as_ptr() as *const __m256i;

        let mut sum_lo = _mm256_setzero_si256();
        let mut sum_hi = _mm256_setzero_si256();

        for i in 0..16 {
            let s = _mm256_load_si256(s_ptr.add(i));
            let w = _mm256_loadu_si256(w_ptr.add(i));
            let clamped = _mm256_min_epi16(_mm256_max_epi16(s, zero), max_val);
            let screlu = _mm256_mullo_epi16(clamped, clamped);

            let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
            let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));
            let sc_lo = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(screlu));
            let sc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(screlu, 1));

            sum_lo = _mm256_add_epi32(sum_lo, _mm256_mullo_epi32(w_lo, sc_lo));
            sum_hi = _mm256_add_epi32(sum_hi, _mm256_mullo_epi32(w_hi, sc_hi));
        }
        let sum = _mm256_add_epi32(sum_lo, sum_hi);
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum);
        for v in tmp {
            total += v as i64;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;

    #[test]
    fn test_nnue_forward_pass_mathematical_correctness() {
        let mut network = Network::new_boxed();
        network.output_bias = 10;
        network.output_weights[..ACC_SIZE].fill(2);
        network.output_weights[ACC_SIZE..2 * ACC_SIZE].fill(3);

        let mut board = BoardState::new();

        let idx = board.history.index;
        board.history.accumulators[idx].white.state.fill(10);
        board.history.accumulators[idx].black.state.fill(20);

        board.side_to_move = Side::White;
        let score = evaluate_internal(&board, &network);

        // Active state value: 10.clamp(0, 255) = 10. screlu = 10 * 10 = 100.
        // Passive state value: 20.clamp(0, 255) = 20. screlu = 20 * 20 = 400.
        // sum = 256 * (100 * 2) + 256 * (400 * 3) = 51200 + 307200 = 358400
        // Dequantize:
        // output = 358400 / 255 = 1405
        // output += 10 (bias) = 1415
        // output *= 400 (SCALE) = 566000
        // output /= 16320 (QA * QB) = 34
        assert_eq!(score, 34);
    }

    #[test]
    fn test_avx2_matches_scalar_evaluation() {
        let mut network = Network::new_boxed();
        network.output_bias = 42;
        for i in 0..ACC_SIZE {
            network.output_weights[i] = (i as i16 % 29) - 14;
            network.output_weights[ACC_SIZE + i] = (i as i16 % 31) - 15;
        }

        let mut board = BoardState::new();
        let idx = board.history.index;
        for i in 0..ACC_SIZE {
            board.history.accumulators[idx].white.state[i] = ((i as i16 * 17) % 350) - 50;
            board.history.accumulators[idx].black.state[i] = ((i as i16 * 23) % 400) - 70;
        }

        let score = evaluate_internal(&board, &network);

        let mut scalar_output: i64 = 0;
        for (&input, &weight) in board.history.accumulators[idx]
            .white
            .state
            .iter()
            .zip(&network.output_weights[0..ACC_SIZE])
        {
            let val = i64::from(input).clamp(0, 255);
            scalar_output += val * val * i64::from(weight);
        }
        for (&input, &weight) in board.history.accumulators[idx]
            .black
            .state
            .iter()
            .zip(&network.output_weights[ACC_SIZE..2 * ACC_SIZE])
        {
            let val = i64::from(input).clamp(0, 255);
            scalar_output += val * val * i64::from(weight);
        }
        scalar_output /= 255;
        scalar_output += i64::from(network.output_bias);
        scalar_output *= i64::from(SCALE);
        scalar_output /= 255 * 64;
        let scalar_score = scalar_output.clamp(-29000, 29000) as i16;

        assert_eq!(score, scalar_score);
    }
}
