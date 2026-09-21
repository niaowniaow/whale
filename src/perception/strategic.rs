use crate::bitboard::lookups::pawn_attacks;
use crate::board::state::BoardState;
use crate::common::piece::Piece;
use crate::common::side::Side;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StrategicSnapshot {
    pub white_pawn_islands: u8,
    pub black_pawn_islands: u8,
    pub white_passed_pawns: u8,
    pub black_passed_pawns: u8,
    pub white_isolated_pawns: u8,
    pub black_isolated_pawns: u8,
    pub white_backward_pawns: u8,
    pub black_backward_pawns: u8,
    pub white_bishop_pair: bool,
    pub black_bishop_pair: bool,
    pub open_files: u8,
    pub white_semi_open_files: u8,
    pub black_semi_open_files: u8,
    pub white_king_shelter: u8,
    pub black_king_shelter: u8,
}

#[inline(always)]
fn file_mask(f: usize) -> u64 {
    0x0101010101010101u64 << f
}

pub fn compute_strategic_snapshot(board: &BoardState) -> StrategicSnapshot {
    let mut snap = StrategicSnapshot::default();

    let wp = board.get_pieces(Side::White, Piece::Pawn).0;
    let bp = board.get_pieces(Side::Black, Piece::Pawn).0;

    snap.white_pawn_islands = count_pawn_islands(wp);
    snap.black_pawn_islands = count_pawn_islands(bp);

    snap.white_isolated_pawns = count_isolated_pawns(wp);
    snap.black_isolated_pawns = count_isolated_pawns(bp);

    snap.white_passed_pawns = count_passed_pawns(wp, bp, Side::White);
    snap.black_passed_pawns = count_passed_pawns(bp, wp, Side::Black);

    snap.white_backward_pawns = count_backward_pawns(wp, bp, Side::White);
    snap.black_backward_pawns = count_backward_pawns(bp, wp, Side::Black);

    snap.white_bishop_pair = board.get_pieces(Side::White, Piece::Bishop).0.count_ones() >= 2;
    snap.black_bishop_pair = board.get_pieces(Side::Black, Piece::Bishop).0.count_ones() >= 2;

    for f in 0..8 {
        let mask = file_mask(f);
        let has_w = (wp & mask) != 0;
        let has_b = (bp & mask) != 0;
        if !has_w && !has_b {
            snap.open_files += 1;
        } else if !has_w && has_b {
            snap.white_semi_open_files += 1;
        } else if has_w && !has_b {
            snap.black_semi_open_files += 1;
        }
    }

    snap.white_king_shelter = king_shelter_score(board, Side::White, wp);
    snap.black_king_shelter = king_shelter_score(board, Side::Black, bp);

    snap
}

fn count_pawn_islands(pawns: u64) -> u8 {
    let mut islands = 0u8;
    let mut in_island = false;
    for f in 0..8 {
        let has_pawn = (pawns & file_mask(f)) != 0;
        if has_pawn && !in_island {
            islands += 1;
            in_island = true;
        } else if !has_pawn {
            in_island = false;
        }
    }
    islands
}

fn count_isolated_pawns(pawns: u64) -> u8 {
    let mut count = 0u8;
    for f in 0..8 {
        let file_pawns = pawns & file_mask(f);
        if file_pawns == 0 {
            continue;
        }
        let left = if f > 0 { file_mask(f - 1) } else { 0 };
        let right = if f < 7 { file_mask(f + 1) } else { 0 };
        if (pawns & (left | right)) == 0 {
            count += file_pawns.count_ones() as u8;
        }
    }
    count
}

fn count_passed_pawns(our_pawns: u64, their_pawns: u64, side: Side) -> u8 {
    let mut count = 0u8;
    let mut pawns = our_pawns;
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as usize;
        pawns &= pawns - 1;

        let file = sq % 8;
        let rank = sq / 8;

        let left = if file > 0 { file_mask(file - 1) } else { 0 };
        let right = if file < 7 { file_mask(file + 1) } else { 0 };
        let mid = file_mask(file);
        let span_files = left | mid | right;

        let span_ahead = if side == Side::White {
            let mut ahead = 0u64;
            for r in 0..rank {
                ahead |= 0xFFu64 << (r * 8);
            }
            ahead
        } else {
            let mut ahead = 0u64;
            for r in (rank + 1)..8 {
                ahead |= 0xFFu64 << (r * 8);
            }
            ahead
        };

        if (their_pawns & span_files & span_ahead) == 0 {
            count += 1;
        }
    }
    count
}

fn count_backward_pawns(our_pawns: u64, their_pawns: u64, side: Side) -> u8 {
    let mut count = 0u8;
    let mut pawns = our_pawns;
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as usize;
        pawns &= pawns - 1;

        let file = sq % 8;
        let rank = sq / 8;

        let adjacent_files = (if file > 0 { file_mask(file - 1) } else { 0 })
            | (if file < 7 { file_mask(file + 1) } else { 0 });

        let behind_or_equal = if side == Side::White {
            let mut mask = 0u64;
            for r in rank..8 {
                mask |= 0xFFu64 << (r * 8);
            }
            mask
        } else {
            let mut mask = 0u64;
            for r in 0..=rank {
                mask |= 0xFFu64 << (r * 8);
            }
            mask
        };

        let cannot_be_supported = (our_pawns & adjacent_files & behind_or_equal) == 0;
        if cannot_be_supported {
            let stop_sq = if side == Side::White {
                sq.saturating_sub(8)
            } else {
                sq + 8
            };
            if stop_sq < 64 {
                let attacks = pawn_attacks()[side.other() as usize][stop_sq];
                if (their_pawns & attacks) != 0 {
                    count += 1;
                }
            }
        }
    }
    count
}

fn king_shelter_score(board: &BoardState, side: Side, our_pawns: u64) -> u8 {
    let king_bb = board.get_pieces(side, Piece::King);
    if king_bb.is_empty() {
        return 0;
    }
    let king_sq = king_bb.get_lsb() as usize;
    let k_file = king_sq % 8;
    let k_rank = king_sq / 8;

    let mut score = 0u8;
    let start_file = if k_file > 0 { k_file - 1 } else { 0 };
    let end_file = if k_file < 7 { k_file + 1 } else { 7 };

    for f in start_file..=end_file {
        let pawns_on_file = our_pawns & file_mask(f);
        if pawns_on_file != 0 {
            score += 2;
            let closest_rank = if side == Side::White {
                63usize.saturating_sub(pawns_on_file.leading_zeros() as usize) / 8
            } else {
                pawns_on_file.trailing_zeros() as usize / 8
            };
            let dist = closest_rank.abs_diff(k_rank);
            if dist <= 1 {
                score += 1;
            }
        }
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn starting_position_strategic_snapshot() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let snap = compute_strategic_snapshot(&board);

        assert_eq!(snap.white_pawn_islands, 1);
        assert_eq!(snap.black_pawn_islands, 1);
        assert_eq!(snap.white_isolated_pawns, 0);
        assert_eq!(snap.black_isolated_pawns, 0);
        assert_eq!(snap.white_passed_pawns, 0);
        assert_eq!(snap.black_passed_pawns, 0);
        assert_eq!(snap.open_files, 0);
        assert!(snap.white_bishop_pair);
        assert!(snap.black_bishop_pair);
    }
}
