use super::*;

#[inline(always)]
pub(super) fn to_sf(sq: usize) -> usize {
    sq ^ 56
}

const KING_BUCKETS: [u32; 64] = [
    28, 29, 30, 31, 31, 30, 29, 28, 24, 25, 26, 27, 27, 26, 25, 24, 20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16, 12, 13, 14, 15, 15, 14, 13, 12, 8, 9, 10, 11, 11, 10, 9, 8, 4,
    5, 6, 7, 7, 6, 5, 4, 0, 1, 2, 3, 3, 2, 1, 0,
];

#[inline(always)]
pub(super) fn halfka_orient(ksq_sf: usize) -> usize {
    if ksq_sf & 7 < 4 { 7 } else { 0 }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn sf_piece_code(side: Side, piece: Piece) -> usize {
    let base = piece as usize + 1;
    if side == Side::Black { base + 8 } else { base }
}
pub struct SfnnPosition {
    pub pieces: [u64; 6],
    pub white: u64,
    pub black: u64,

    pub mapping: [u8; 64],
}

impl SfnnPosition {
    pub fn from_board(board: &BoardState) -> Self {
        let mut pieces = [0u64; 6];
        for &p in &Piece::ALL {
            pieces[p as usize] = board.pieces[p].0.swap_bytes();
        }
        let mut mapping = [6u8; 64];
        for (i, slot) in mapping.iter_mut().enumerate() {
            let rank = 7 - (i >> 3);
            *slot = board.piece_mapping[(rank << 3) | (i & 7)] as u8;
        }
        Self {
            pieces,
            white: board.occupancies[Side::White].0.swap_bytes(),
            black: board.occupancies[Side::Black].0.swap_bytes(),
            mapping,
        }
    }

    #[inline(always)]
    pub fn occupied(&self) -> u64 {
        self.white | self.black
    }

    pub fn king_square(&self, perspective: Side) -> usize {
        let bb = self.pieces[Piece::King as usize]
            & if perspective == Side::White {
                self.white
            } else {
                self.black
            };
        bb.trailing_zeros() as usize
    }

    pub fn piece_count(&self) -> usize {
        self.occupied().count_ones() as usize
    }
}
pub fn halfka_index(
    perspective: Side,
    side: Side,
    piece: Piece,
    sq_sf: usize,
    ksq_sf: usize,
) -> Option<usize> {
    if piece == Piece::None {
        return None;
    }
    let flip = if perspective == Side::Black { 56 } else { 0 };
    let oriented = sq_sf ^ halfka_orient(ksq_sf) ^ flip;
    let ps = if piece == Piece::King {
        640
    } else {
        (piece as usize) * 128 + if side == perspective { 0 } else { 64 }
    };
    let bucket = KING_BUCKETS[ksq_sf ^ flip] as usize;
    Some(oriented + ps + bucket * PS_PLANES)
}

pub fn append_halfka(pos: &SfnnPosition, perspective: Side, out: &mut Vec<usize>) {
    let own_king = pos.pieces[Piece::King as usize]
        & if perspective == Side::White {
            pos.white
        } else {
            pos.black
        };
    if own_king == 0 {
        return;
    }
    let ksq = pos.king_square(perspective);
    let mut bb = pos.occupied();
    while bb != 0 {
        let s = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let is_white = (pos.white >> s) & 1 == 1;
        let side = if is_white { Side::White } else { Side::Black };
        let pt = pos.mapping[s] as usize;
        if pt > 5 {
            continue;
        }
        let piece = Piece::ALL[pt];
        if let Some(idx) = halfka_index(perspective, side, piece, s, ksq) {
            out.push(idx);
        }
    }
}
pub(super) fn kings_present(pos: &SfnnPosition) -> bool {
    (pos.pieces[Piece::King as usize] & pos.white) != 0
        && (pos.pieces[Piece::King as usize] & pos.black) != 0
}
