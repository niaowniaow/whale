pub mod accumulator;
pub mod dcn;
pub mod features;
pub mod loader;
pub mod v16;

use crate::board::state::BoardState;
use crate::common::side::Side;

use self::loader::Network;

pub const ACC_SIZE: usize = 256;
pub const INPUT_SIZE: usize = 768;

pub const SCALE: i32 = 400;

const EVAL_CACHE_BITS: u32 = 18;
const EVAL_CACHE_SIZE: usize = 1 << EVAL_CACHE_BITS;
const EVAL_CACHE_MASK: u64 = (EVAL_CACHE_SIZE as u64) - 1;

#[derive(Clone, Copy)]
struct EvalCacheEntry {
    hash: u64,
    optimism: i32,
    halfmove: u8,
    score: i16,
}

std::thread_local! {
    static EVAL_CACHE: std::cell::RefCell<Box<[EvalCacheEntry]>> = std::cell::RefCell::new(
        vec![
            EvalCacheEntry {
                hash: u64::MAX,
                optimism: 0,
                halfmove: 0,
                score: 0
            };
            EVAL_CACHE_SIZE
        ]
        .into_boxed_slice(),
    );
}

#[inline(always)]
fn probe_eval_cache(hash: u64, optimism: i32, halfmove: u8) -> Option<i16> {
    EVAL_CACHE.with(|cache| {
        let entry = cache.borrow()[(hash & EVAL_CACHE_MASK) as usize];
        (entry.hash == hash && entry.optimism == optimism && entry.halfmove == halfmove)
            .then_some(entry.score)
    })
}

#[inline(always)]
fn store_eval_cache(hash: u64, optimism: i32, halfmove: u8, score: i16) {
    EVAL_CACHE.with(|cache| {
        cache.borrow_mut()[(hash & EVAL_CACHE_MASK) as usize] = EvalCacheEntry {
            hash,
            optimism,
            halfmove,
            score,
        };
    });
}

pub fn clear_eval_cache() {
    EVAL_CACHE.with(|cache| {
        cache.borrow_mut().fill(EvalCacheEntry {
            hash: u64::MAX,
            optimism: 0,
            halfmove: 0,
            score: 0,
        });
    });
}

static DUAL_NET_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

#[inline(always)]
pub fn set_dual_net(enabled: bool) {
    DUAL_NET_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[inline(always)]
pub fn is_dual_net_enabled() -> bool {
    DUAL_NET_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn evaluate_qsearch(board: &mut BoardState, optimism: i32, alpha: i16, beta: i16) -> i16 {
    let board_hash = board.board_hash;
    let halfmove = board.half_move_clock;
    if let Some(hit) = probe_eval_cache(board_hash, optimism, halfmove) {
        return hit;
    }
    if !v16::maintenance_active() || !is_dual_net_enabled() {
        return evaluate_with_optimism(board, optimism);
    }
    let fast = evaluate_fast(board, optimism);
    if fast >= beta + 120 {
        store_eval_cache(board_hash, optimism, halfmove, fast);
        return fast;
    }
    if fast <= alpha - 426 {
        store_eval_cache(board_hash, optimism, halfmove, fast);
        return fast;
    }
    if fast.abs() >= 380 {
        store_eval_cache(board_hash, optimism, halfmove, fast);
        return fast;
    }
    evaluate_with_optimism(board, optimism)
}

#[inline(always)]
pub fn evaluate(board: &mut BoardState) -> i16 {
    evaluate_with_optimism(board, 0)
}

#[inline(always)]
pub fn evaluate_with_depth(board: &mut BoardState, optimism: i32, depth: u8) -> i16 {
    let base_score = evaluate_with_optimism(board, optimism);
    dcn::DcnModel::condition_evaluation(base_score, depth, board, dcn::DcnConfig::default())
}

#[inline(always)]
pub fn evaluate_with_depth_cached(
    board: &mut BoardState,
    optimism: i32,
    depth: u8,
    nt: &crate::board::node_threats::NodeThreats,
) -> i16 {
    let base_score = evaluate_with_optimism(board, optimism);
    dcn::DcnModel::condition_evaluation_cached(
        base_score,
        depth,
        board,
        nt.checks(),
        nt.queen_threat_us(board),
        nt.queen_threat_them(board),
        dcn::DcnConfig::default(),
    )
}

#[inline(always)]
pub fn evaluate_with_optimism(board: &mut BoardState, optimism: i32) -> i16 {
    let board_hash = board.board_hash;
    let halfmove = board.half_move_clock;
    let score = if let Some(hit) = probe_eval_cache(board_hash, optimism, halfmove) {
        hit
    } else {
        let raw = if v16::maintenance_active() {
            let fast = if is_dual_net_enabled() {
                evaluate_fast(board, optimism)
            } else {
                0
            };
            if is_dual_net_enabled() && fast.abs() >= 380 {
                fast
            } else if let Some(ev) = v16::evaluate_board_detailed(board) {
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
                let non_pawn_mat = knight_cnt * v16::KNIGHT_VALUE as i64
                    + bishop_cnt * v16::BISHOP_VALUE as i64
                    + rook_cnt * v16::ROOK_VALUE as i64
                    + queen_cnt * v16::QUEEN_VALUE as i64;
                let material = 534 * pawn_cnt + non_pawn_mat;
                let mut v = nnue + (nnue * material + opt * 7675) / 91000;
                v -= v * board.half_move_clock as i64 / 199;
                v.clamp(-29000, 29000) as i16
            } else {
                v16::evaluate_board(board).unwrap_or(0)
            }
        } else {
            board.ensure_accumulators_fresh();
            evaluate_fast(board, optimism)
        };
        store_eval_cache(board_hash, optimism, halfmove, raw);
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
    let non_pawn_mat = knight_cnt * v16::KNIGHT_VALUE as i64
        + bishop_cnt * v16::BISHOP_VALUE as i64
        + rook_cnt * v16::ROOK_VALUE as i64
        + queen_cnt * v16::QUEEN_VALUE as i64;
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

    #[test]
    fn test_eval_cache_hit_miss() {
        clear_eval_cache();
        assert_eq!(probe_eval_cache(123, 0, 0), None);
        store_eval_cache(123, 0, 0, 77);
        assert_eq!(probe_eval_cache(123, 0, 0), Some(77));
        assert_eq!(probe_eval_cache(123, 5, 0), None);
        assert_eq!(probe_eval_cache(123, 0, 1), None);
        assert_eq!(probe_eval_cache(124, 0, 0), None);
        clear_eval_cache();
        assert_eq!(probe_eval_cache(123, 0, 0), None);
    }

    #[test]
    fn test_evaluate_is_cached_and_deterministic() {
        use crate::common::helpers::STARTING_FEN;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let first = evaluate(&mut board);
        let second = evaluate(&mut board);
        assert_eq!(first, second);
        assert!(first.abs() < 5000);

        let _ = evaluate_with_optimism(&mut board, 5);
    }

    #[test]
    fn test_evaluate_fast_halfmove_damping() {
        use crate::common::helpers::STARTING_FEN;
        use crate::common::piece::Piece;
        use crate::common::square::Square;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);
        board.add_piece(Square::E4, Side::White, Piece::Queen, true);
        board.flush_pending_updates(board.history.index);
        board.ensure_accumulators_fresh();
        let fresh = evaluate_fast(&mut board, 0);
        assert_ne!(fresh, 0);
        board.half_move_clock = 60;
        let damped = evaluate_fast(&mut board, 0);
        assert_ne!(fresh, damped);
        assert!(damped.abs() < fresh.abs());
    }

    #[test]
    fn test_evaluate_with_depth_runs() {
        use crate::common::helpers::STARTING_FEN;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let score = evaluate_with_depth(&mut board, 0, 1);
        assert!(score.abs() <= 29000);
    }

    #[test]
    fn test_evaluate_internal_black_to_move_swaps_sides() {
        let mut network = Network::new_boxed();
        network.output_bias = 0;
        network.output_weights[..ACC_SIZE].fill(1);
        network.output_weights[ACC_SIZE..2 * ACC_SIZE].fill(10);

        let mut board = BoardState::new();
        let idx = board.history.index;
        board.history.accumulators[idx].white.state.fill(30);
        board.history.accumulators[idx].black.state.fill(5);

        board.side_to_move = Side::White;
        let white_stm = evaluate_internal(&board, &network);
        board.side_to_move = Side::Black;
        let black_stm = evaluate_internal(&board, &network);

        assert_ne!(white_stm, black_stm);
    }

    #[test]
    fn test_evaluate_fast_optimism_shifts_score() {
        use crate::common::helpers::STARTING_FEN;
        use crate::common::piece::Piece;
        use crate::common::square::Square;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);
        board.add_piece(Square::E4, Side::White, Piece::Queen, true);
        board.flush_pending_updates(board.history.index);
        board.ensure_accumulators_fresh();
        let base = evaluate_fast(&mut board, 0);
        let optimistic = evaluate_fast(&mut board, 1000);
        assert_ne!(base, optimistic);
        assert!(optimistic > base);
    }

    #[test]
    fn test_evaluate_with_optimism_keys_cache_separately() {
        use crate::common::helpers::STARTING_FEN;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let hash = board.board_hash;
        let halfmove = board.half_move_clock;
        let a = evaluate_with_optimism(&mut board, 0);

        assert_eq!(probe_eval_cache(hash, 0, halfmove), Some(a));
        assert_eq!(probe_eval_cache(hash, 7, halfmove), None);
        let b = evaluate_with_optimism(&mut board, 7);
        assert_eq!(probe_eval_cache(hash, 7, halfmove), Some(b));

        board.half_move_clock = halfmove.wrapping_add(1);
        assert_eq!(
            probe_eval_cache(hash, 0, board.half_move_clock),
            if halfmove.wrapping_add(1) == halfmove {
                Some(a)
            } else {
                None
            }
        );
    }

    #[test]
    fn test_evaluate_internal_clamps_extreme_scores() {
        let mut high = Network::new_boxed();
        high.output_bias = 0;
        high.output_weights.fill(100);
        let mut low = Network::new_boxed();
        low.output_bias = 0;
        low.output_weights.fill(-100);

        let mut board = BoardState::new();
        let idx = board.history.index;
        board.history.accumulators[idx].white.state.fill(300);
        board.history.accumulators[idx].black.state.fill(300);
        board.side_to_move = Side::White;

        assert_eq!(evaluate_internal(&board, &high), 29000);
        assert_eq!(evaluate_internal(&board, &low), -29000);
    }

    #[test]
    fn test_evaluate_internal_negative_inputs_clamp_to_zero() {
        let mut network = Network::new_boxed();
        network.output_bias = 7;
        network.output_weights.fill(5);

        let mut board = BoardState::new();
        let idx = board.history.index;
        board.history.accumulators[idx].white.state.fill(-100);
        board.history.accumulators[idx].black.state.fill(-100);
        board.side_to_move = Side::White;

        assert_eq!(evaluate_internal(&board, &network), 0);
    }

    #[test]
    fn test_evaluate_fast_negative_optimism_lowers_score() {
        use crate::common::helpers::STARTING_FEN;
        use crate::common::piece::Piece;
        use crate::common::square::Square;

        let mut board = BoardState::parse_fen(STARTING_FEN);
        board.add_piece(Square::E4, Side::White, Piece::Queen, true);
        board.flush_pending_updates(board.history.index);
        board.ensure_accumulators_fresh();
        let base = evaluate_fast(&mut board, 0);
        let pessimistic = evaluate_fast(&mut board, -1000);
        assert_ne!(base, pessimistic);
        assert!(pessimistic < base);
    }

    #[test]
    fn test_evaluate_fast_material_zero_matches_internal() {
        use crate::common::piece::Piece;
        use crate::common::square::Square;

        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, true);
        board.add_piece(Square::E8, Side::Black, Piece::King, true);
        board.flush_pending_updates(board.history.index);
        board.ensure_accumulators_fresh();
        board.half_move_clock = 0;
        let network = Network::get_embedded();

        assert_eq!(
            evaluate_fast(&mut board, 0),
            evaluate_internal(&board, network)
        );
    }

    #[test]
    fn test_evaluate_with_depth_covers_all_regimes() {
        use crate::common::helpers::STARTING_FEN;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);

        for depth in [0u8, 1, 5, 10, 15, 16, 64] {
            let score = evaluate_with_depth(&mut board, 0, depth);
            assert!(score.abs() <= 29000, "depth {depth} out of range");
        }
    }

    #[test]
    fn test_eval_cache_evicted_on_index_collision() {
        clear_eval_cache();
        store_eval_cache(123, 0, 0, 11);
        assert_eq!(probe_eval_cache(123, 0, 0), Some(11));

        let colliding = 123u64 + EVAL_CACHE_SIZE as u64;
        store_eval_cache(colliding, 0, 0, 22);
        assert_eq!(probe_eval_cache(colliding, 0, 0), Some(22));
        assert_eq!(probe_eval_cache(123, 0, 0), None);
        clear_eval_cache();
    }

    #[test]
    fn test_evaluate_stays_within_mate_range() {
        use crate::common::helpers::STARTING_FEN;
        use crate::common::piece::Piece;
        use crate::common::square::Square;

        clear_eval_cache();
        let mut board = BoardState::parse_fen(STARTING_FEN);
        for sq in [Square::E4, Square::D5] {
            board.add_piece(sq, Side::White, Piece::Queen, true);
            board.flush_pending_updates(board.history.index);
        }
        for optimism in [0, 500, -500] {
            let s = evaluate_with_optimism(&mut board, optimism);
            assert!(s.abs() <= 29000);
        }
        board.half_move_clock = 100;
        let damped = evaluate(&mut board);
        assert!(damped.abs() <= 29000);
    }
}
