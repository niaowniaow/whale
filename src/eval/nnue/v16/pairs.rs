use super::*;

#[inline(always)]
pub(super) fn make_pawn_id(color: Side, square_sf: usize) -> Option<usize> {
    square_sf.checked_sub(8).map(|s| (color as usize) * 48 + s)
}

const PAWN_PAIR_BB_TABLE: [u64; 64] = {
    let mut table = [0u64; 64];
    let mut s = 0;
    while s < 64 {
        let file = s & 7;
        let file_bb = 0x0101_0101_0101_0101u64 << file;
        let east = (file_bb << 1) & !0x0101_0101_0101_0101u64;
        let west = (file_bb >> 1) & !0x8080_8080_8080_8080u64;
        let files = file_bb | east | west;
        let rank18 = 0xFF00_0000_0000_00FFu64;
        table[s] = files & !rank18 & !(1u64 << s);
        s += 1;
    }
    table
};

#[inline(always)]
fn pawn_pair_bb(s: usize) -> u64 {
    PAWN_PAIR_BB_TABLE[s]
}

pub fn pair_make_index(
    perspective: Side,
    color: Side,
    from_sf: usize,
    to_sf: usize,
    paired_color: Side,
    ksq_sf: usize,
) -> Option<usize> {
    let flip = if perspective == Side::Black { 56 } else { 0 };
    let left_files = (ksq_sf & 7) < 4;
    let orient = flip ^ if left_files { 0 } else { 7 };

    let from_oriented = from_sf ^ orient;
    let to_oriented = to_sf ^ orient;

    let color_oriented = if perspective == color {
        Side::White
    } else {
        Side::Black
    };
    let paired_color_oriented = if perspective == paired_color {
        Side::White
    } else {
        Side::Black
    };

    let id_a = make_pawn_id(color_oriented, from_oriented)?;
    let id_b = make_pawn_id(paired_color_oriented, to_oriented)?;

    let hi = id_a.max(id_b);
    let lo = id_a.min(id_b);

    Some(hi * (hi - 1) / 2 + lo + PSQ_DIMS + THREAT_DIMS)
}

pub fn pair_index_for(
    perspective: Side,
    color: Side,
    from_sf: usize,
    to_sf: usize,
    paired_color: Side,
    ksq_sf: usize,
) -> Option<usize> {
    pair_make_index(perspective, color, from_sf, to_sf, paired_color, ksq_sf)
        .map(|abs| abs - (PSQ_DIMS + THREAT_DIMS))
}

pub fn for_each_pair(pos: &SfnnPosition, mut emit: impl FnMut(Side, usize, usize, Side)) {
    let white = pos.pieces[Piece::Pawn as usize] & pos.white;
    let black = pos.pieces[Piece::Pawn as usize] & pos.black;

    let mut bb = white;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);

        let mut ww = band & bb;
        while ww != 0 {
            let to = ww.trailing_zeros() as usize;
            ww &= ww - 1;
            emit(Side::White, from, to, Side::White);
        }

        let mut wb = band & black;
        while wb != 0 {
            let to = wb.trailing_zeros() as usize;
            wb &= wb - 1;
            emit(Side::White, from, to, Side::Black);
        }
    }

    let mut bb = black;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);

        let mut bbk = band & bb;
        while bbk != 0 {
            let to = bbk.trailing_zeros() as usize;
            bbk &= bbk - 1;
            emit(Side::Black, from, to, Side::Black);
        }
    }
}

pub fn append_pairs(pos: &SfnnPosition, perspective: Side, out: &mut Vec<usize>) {
    let ksq = pos.king_square(perspective);
    for_each_pair(pos, |color, from, to, paired| {
        if let Some(idx) = pair_make_index(perspective, color, from, to, paired, ksq) {
            out.push(idx);
        }
    });
}
pub fn collect_pairs(
    pos: &SfnnPosition,
    perspective: Side,
    out: &mut [usize; MAX_PAIR_ACTIVE],
) -> usize {
    let ksq = pos.king_square(perspective);
    let white = pos.pieces[Piece::Pawn as usize] & pos.white;
    let black = pos.pieces[Piece::Pawn as usize] & pos.black;
    let mut len = 0;

    let mut bb = white;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);

        let mut ww = band & bb;
        while ww != 0 {
            let to = ww.trailing_zeros() as usize;
            ww &= ww - 1;
            if let Some(idx) = pair_make_index(perspective, Side::White, from, to, Side::White, ksq)
                && len < MAX_PAIR_ACTIVE
            {
                out[len] = idx;
                len += 1;
            }
        }

        let mut wb = band & black;
        while wb != 0 {
            let to = wb.trailing_zeros() as usize;
            wb &= wb - 1;
            if let Some(idx) = pair_make_index(perspective, Side::White, from, to, Side::Black, ksq)
                && len < MAX_PAIR_ACTIVE
            {
                out[len] = idx;
                len += 1;
            }
        }
    }

    let mut bb = black;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);

        let mut bbk = band & bb;
        while bbk != 0 {
            let to = bbk.trailing_zeros() as usize;
            bbk &= bbk - 1;
            if let Some(idx) = pair_make_index(perspective, Side::Black, from, to, Side::Black, ksq)
                && len < MAX_PAIR_ACTIVE
            {
                out[len] = idx;
                len += 1;
            }
        }
    }
    len
}
