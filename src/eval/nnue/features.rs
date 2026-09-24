use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

pub const PIECE_FEATURES: usize = 768;

pub const HALFMOVE_BUCKETS: usize = 4;

pub const REPETITION_BUCKETS: usize = 3;

pub const CASTLING_PROGRESS_FEATURES: usize = 2;

pub const TOTAL_FEATURES: usize =
    PIECE_FEATURES + HALFMOVE_BUCKETS + REPETITION_BUCKETS + CASTLING_PROGRESS_FEATURES;

pub const HALFMOVE_BASE: usize = PIECE_FEATURES;

pub const REPETITION_BASE: usize = PIECE_FEATURES + HALFMOVE_BUCKETS;

pub const CASTLING_BASE: usize = PIECE_FEATURES + HALFMOVE_BUCKETS + REPETITION_BUCKETS;

pub const NO_PROGRESS_HALFMOVE: u8 = 50;

#[inline(always)]
pub fn get_feature_index(piece: Piece, relative_side: Side, square: Square) -> Option<usize> {
    if piece == Piece::None {
        return None;
    }

    let side_offset = (relative_side as usize) * 384;
    let piece_offset = (piece as usize) * 64;

    let sq = (square as usize) ^ 56;

    Some(side_offset + piece_offset + sq)
}

#[inline(always)]
pub fn halfmove_bucket(halfmove: u8) -> usize {
    let b = (halfmove as usize) / 25;
    if b > HALFMOVE_BUCKETS - 1 {
        HALFMOVE_BUCKETS - 1
    } else {
        b
    }
}

#[inline(always)]
pub fn halfmove_feature_index(halfmove: u8) -> usize {
    HALFMOVE_BASE + halfmove_bucket(halfmove)
}

#[inline(always)]
pub fn repetition_bucket(count: u8) -> usize {
    let c = count as usize;
    if c > REPETITION_BUCKETS - 1 {
        REPETITION_BUCKETS - 1
    } else {
        c
    }
}

#[inline(always)]
pub fn repetition_feature_index(count: u8) -> usize {
    REPETITION_BASE + repetition_bucket(count)
}

#[inline(always)]
pub fn castling_feature_index() -> usize {
    CASTLING_BASE
}

#[inline(always)]
pub fn no_progress_feature_index() -> usize {
    CASTLING_BASE + 1
}

#[inline(always)]
pub fn castling_active(has_rights: bool) -> Option<usize> {
    if has_rights {
        Some(castling_feature_index())
    } else {
        None
    }
}

#[inline(always)]
pub fn no_progress_active(halfmove: u8) -> Option<usize> {
    if halfmove >= NO_PROGRESS_HALFMOVE {
        Some(no_progress_feature_index())
    } else {
        None
    }
}

#[inline(always)]
pub fn aux_feature_indices(
    halfmove: u8,
    repetitions: u8,
    has_castling: bool,
    out: &mut [usize; 4],
) -> usize {
    out[0] = halfmove_feature_index(halfmove);
    out[1] = repetition_feature_index(repetitions);
    let mut n = 2;
    if let Some(idx) = castling_active(has_castling) {
        out[n] = idx;
        n += 1;
    }
    if let Some(idx) = no_progress_active(halfmove) {
        out[n] = idx;
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_returns_none() {
        assert_eq!(
            get_feature_index(Piece::None, Side::White, Square::A1),
            None
        );
    }

    #[test]
    fn test_bounds_and_limits() {
        let min_idx = get_feature_index(Piece::Pawn, Side::White, Square::A1).unwrap();
        assert_eq!(min_idx, 0);

        let max_idx = get_feature_index(Piece::King, Side::Black, Square::H8).unwrap();

        assert_eq!(max_idx, 767);
    }
}
