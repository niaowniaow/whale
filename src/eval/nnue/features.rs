use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

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
