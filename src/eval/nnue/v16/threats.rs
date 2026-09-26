use super::*;

const FILE_A_BB: u64 = 0x0101_0101_0101_0101;
const FILE_H_BB: u64 = 0x8080_8080_8080_8080;

const KNIGHT_ATTACKS_TABLE: [u64; 64] = {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        let mut attacks = 0u64;
        attacks |= (b << 17) & !FILE_A_BB;
        attacks |= (b << 15) & !FILE_H_BB;
        attacks |= (b << 10) & !(FILE_A_BB | (FILE_A_BB << 1));
        attacks |= (b << 6) & !(FILE_H_BB | (FILE_H_BB >> 1));
        attacks |= (b >> 17) & !FILE_H_BB;
        attacks |= (b >> 15) & !FILE_A_BB;
        attacks |= (b >> 10) & !(FILE_H_BB | (FILE_H_BB >> 1));
        attacks |= (b >> 6) & !(FILE_A_BB | (FILE_A_BB << 1));
        table[sq] = attacks;
        sq += 1;
    }
    table
};

const KING_ATTACKS_TABLE: [u64; 64] = {
    let mut table = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        let mut attacks = 0u64;
        attacks |= (b << 8) | (b >> 8);
        attacks |= ((b << 1) & !FILE_A_BB) | ((b >> 1) & !FILE_H_BB);
        attacks |= ((b << 9) & !FILE_A_BB) | ((b << 7) & !FILE_H_BB);
        attacks |= ((b >> 7) & !FILE_A_BB) | ((b >> 9) & !FILE_H_BB);
        table[sq] = attacks;
        sq += 1;
    }
    table
};

const PAWN_ATTACKS_TABLE: [[u64; 64]; 2] = {
    let mut table = [[0u64; 64]; 2];
    let mut sq = 0;
    while sq < 64 {
        let b = 1u64 << sq;
        table[0][sq] = ((b << 9) & !FILE_A_BB) | ((b << 7) & !FILE_H_BB);
        table[1][sq] = ((b >> 7) & !FILE_A_BB) | ((b >> 9) & !FILE_H_BB);
        sq += 1;
    }
    table
};

#[inline(always)]
pub(super) fn knight_attacks_sf(sq: usize) -> u64 {
    KNIGHT_ATTACKS_TABLE[sq]
}

#[inline(always)]
fn king_attacks_sf(sq: usize) -> u64 {
    KING_ATTACKS_TABLE[sq]
}

#[inline(always)]
fn pawn_attacks_sf(side: Side, sq: usize) -> u64 {
    PAWN_ATTACKS_TABLE[side as usize][sq]
}

#[inline(always)]
fn bishop_attacks_sf(sq: usize, whale_occ: crate::bitboard::Bitboard) -> u64 {
    let whale_sq = crate::common::square::Square::from(sq ^ 56);
    crate::bitboard::lookups::get_bishop_attacks_from_table(whale_sq, whale_occ)
        .0
        .swap_bytes()
}

#[inline(always)]
fn rook_attacks_sf(sq: usize, whale_occ: crate::bitboard::Bitboard) -> u64 {
    let whale_sq = crate::common::square::Square::from(sq ^ 56);
    crate::bitboard::lookups::get_rook_attacks_from_table(whale_sq, whale_occ)
        .0
        .swap_bytes()
}

#[inline(always)]
fn queen_attacks_sf(sq: usize, whale_occ: crate::bitboard::Bitboard) -> u64 {
    let whale_sq = crate::common::square::Square::from(sq ^ 56);
    (crate::bitboard::lookups::get_bishop_attacks_from_table(whale_sq, whale_occ).0
        | crate::bitboard::lookups::get_rook_attacks_from_table(whale_sq, whale_occ).0)
        .swap_bytes()
}

#[inline(always)]
fn slider_attacks_sf(sf_piece_type: usize, sq: usize, occ: u64) -> u64 {
    let whale_occ = crate::bitboard::Bitboard(occ.swap_bytes());
    match sf_piece_type {
        3 => bishop_attacks_sf(sq, whale_occ),
        4 => rook_attacks_sf(sq, whale_occ),
        _ => queen_attacks_sf(sq, whale_occ),
    }
}

pub(super) fn pseudo_attacks_sf(piece_code: usize, sq: usize) -> u64 {
    let pt = piece_code & 7;
    match pt {
        1 => {
            let f = (sq & 7) as i32;
            let r = (sq >> 3) as i32;
            let dr = if piece_code < 8 { 1 } else { -1 };
            let mut attacks = 0u64;
            for df in [-1, 1] {
                let (cf, cr) = (f + df, r + dr);
                if (0..8).contains(&cf) && (0..8).contains(&cr) {
                    attacks |= 1u64 << ((cr * 8 + cf) as usize);
                }
            }
            attacks
        }
        2 => knight_attacks_sf(sq),
        3..=5 => slider_attacks_sf(pt, sq, 0),
        6 => king_attacks_sf(sq),
        _ => 0,
    }
}

#[inline(always)]
pub(super) fn sf_piece_type(pt_whale: usize) -> usize {
    pt_whale + 1
}

const NUM_VALID_TARGETS: [usize; 16] = [0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0];

const THREAT_MAP: [[i32; 6]; 6] = [
    [-1, 0, -1, 1, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

#[inline(always)]
pub(super) fn threat_orient(perspective: Side, sq_sf: usize) -> usize {
    let left_files = sq_sf & 7 < 4;
    match (perspective, left_files) {
        (Side::White, true) => 0,
        (Side::White, false) => 7,
        (Side::Black, true) => 56,
        (Side::Black, false) => 63,
        (Side::Both, _) => 0,
    }
}
pub(super) struct ThreatLuts {
    pub(super) offsets: [[u32; 66]; 16],
    pub(super) lut1: [[u32; 16]; 16],
    pub(super) lut2: [[[u8; 64]; 64]; 16],
}

fn build_threat_luts() -> ThreatLuts {
    let codes = [1usize, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14];
    let mut offsets = [[0u32; 66]; 16];
    let mut cumulative = 0u32;
    for &code in &codes {
        let pt = code & 7;
        let mut piece_offset = 0u32;
        for (from, slot) in offsets[code].iter_mut().enumerate().take(64) {
            *slot = piece_offset;
            if pt != 1 || (8..56).contains(&from) {
                piece_offset += pseudo_attacks_sf(code, from).count_ones();
            }
        }
        offsets[code][64] = piece_offset;
        offsets[code][65] = cumulative;
        cumulative += NUM_VALID_TARGETS[code] as u32 * piece_offset;
    }
    debug_assert_eq!(cumulative as usize, THREAT_DIMS);

    let mut lut1 = [[0u32; 16]; 16];
    for &attacker in &codes {
        for &attacked in &codes {
            let enemy = (attacker ^ attacked) == 8;
            let at = (attacker & 7) - 1;
            let vt = (attacked & 7) - 1;
            let map = THREAT_MAP[at][vt];
            let semi = (attacker & 7) == (attacked & 7) && (enemy || (attacker & 7) != 1);
            let excluded = map < 0;

            let color = (attacked >> 3) as i32;
            let per = NUM_VALID_TARGETS[attacker] as i32 / 2;
            let feature = offsets[attacker][65]
                .wrapping_add(((color * per + map) as u32).wrapping_mul(offsets[attacker][64]));
            let info = ((excluded as u32) << 1) | ((semi && !excluded) as u32);
            lut1[attacker][attacked] = (feature << 8) | info;
        }
    }

    let mut lut2 = [[[0u8; 64]; 64]; 16];
    for &attacker in &codes {
        for (from, row) in lut2[attacker].iter_mut().enumerate() {
            let attacks = pseudo_attacks_sf(attacker, from);
            for (to, cell) in row.iter_mut().enumerate() {
                let mask = (1u64 << to).wrapping_sub(1);
                *cell = (attacks & mask).count_ones() as u8;
            }
        }
    }

    ThreatLuts {
        offsets,
        lut1,
        lut2,
    }
}

pub(super) fn threat_luts() -> &'static ThreatLuts {
    static LUTS: OnceLock<ThreatLuts> = OnceLock::new();
    LUTS.get_or_init(build_threat_luts)
}

pub(super) fn threat_make_index(
    perspective: Side,
    mut attacker: usize,
    from_sf: usize,
    to_sf: usize,
    mut attacked: usize,
    ksq_sf: usize,
) -> Option<usize> {
    let luts = threat_luts();
    let orient = threat_orient(perspective, ksq_sf);
    let from = from_sf ^ orient;
    let to = to_sf ^ orient;
    if perspective == Side::Black {
        attacker ^= 8;
        attacked ^= 8;
    }
    let word = luts.lut1[attacker][attacked];
    let info = (word & 0xFF) as u8;
    let less_than = (from < to) as u8;
    if (info + less_than) & 2 != 0 {
        return None;
    }
    let index = (word >> 8) + luts.offsets[attacker][from] + luts.lut2[attacker][from][to] as u32;
    if index as usize >= THREAT_DIMS {
        return None;
    }
    Some(index as usize)
}
pub fn for_each_threat(pos: &SfnnPosition, mut emit: impl FnMut(usize, usize, usize, usize)) {
    let occ = pos.occupied();
    let whale_occ = crate::bitboard::Bitboard(occ.swap_bytes());
    for color_idx in 0..2 {
        let c = if color_idx == 0 {
            Side::White
        } else {
            Side::Black
        };
        let side_bb = if c == Side::White {
            pos.white
        } else {
            pos.black
        };
        for pt in 0..6 {
            let mut bb = pos.pieces[pt] & side_bb;
            let sf_pt = sf_piece_type(pt);
            let attacker = if c == Side::White { sf_pt } else { sf_pt + 8 };
            if pt == 0 {
                while bb != 0 {
                    let from = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let mut targets = pawn_attacks_sf(c, from) & occ;
                    while targets != 0 {
                        let to = targets.trailing_zeros() as usize;
                        targets &= targets - 1;
                        let victim = pos.mapping[to] as usize;
                        if victim > 5 {
                            continue;
                        }
                        let attacked = victim + 1 + (1 - ((pos.white >> to) as usize & 1)) * 8;
                        emit(attacker, from, to, attacked);
                    }
                }
            } else {
                while bb != 0 {
                    let from = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let attacks = match pt {
                        1 => knight_attacks_sf(from),
                        2 => bishop_attacks_sf(from, whale_occ),
                        3 => rook_attacks_sf(from, whale_occ),
                        4 => queen_attacks_sf(from, whale_occ),
                        _ => king_attacks_sf(from),
                    } & occ;
                    let mut targets = attacks;
                    while targets != 0 {
                        let to = targets.trailing_zeros() as usize;
                        targets &= targets - 1;
                        let victim = pos.mapping[to] as usize;
                        if victim > 5 {
                            continue;
                        }
                        let attacked = victim + 1 + (1 - ((pos.white >> to) as usize & 1)) * 8;
                        emit(attacker, from, to, attacked);
                    }
                }
            }
        }
    }
}

pub fn threat_index_for(
    perspective: Side,
    attacker: usize,
    from_sf: usize,
    to_sf: usize,
    attacked: usize,
    ksq_sf: usize,
) -> Option<usize> {
    threat_make_index(perspective, attacker, from_sf, to_sf, attacked, ksq_sf)
}

pub fn append_threats(pos: &SfnnPosition, perspective: Side, out: &mut Vec<usize>) {
    let ksq = pos.king_square(perspective);
    for_each_threat(pos, |attacker, from, to, attacked| {
        if let Some(idx) = threat_make_index(perspective, attacker, from, to, attacked, ksq) {
            out.push(PSQ_DIMS + idx);
        }
    });
}
pub fn collect_threats(
    pos: &SfnnPosition,
    perspective: Side,
    out: &mut [usize; MAX_THREAT_ACTIVE],
) -> usize {
    let ksq = pos.king_square(perspective);
    let mut len = 0;
    for_each_threat(pos, |attacker, from, to, attacked| {
        if let Some(idx) = threat_make_index(perspective, attacker, from, to, attacked, ksq)
            && len < MAX_THREAT_ACTIVE
        {
            out[len] = PSQ_DIMS + idx;
            len += 1;
        }
    });
    len
}
