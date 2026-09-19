// Stockfish SFNNv16 inference path (HalfKAv2_hm + FullThreats + PP_3Wide).
//
// Feature sets: HalfKAv2_hm (22_528) + FullThreats (59_808) + PP_3Wide (4_560).
// Big net: L1 1024, FC0 32, FC1 32, 8 material buckets sharing one transformer.
// Transformer uses pairwise products (/512), per-feature PSQT (8 buckets),
// paired Sqr/Clip activations on fc_0 and fc_1, plus the forwarded
// fc_0[30] - fc_0[31] output term. Integer math follows the Stockfish paths,
// so official SFNNv16 .nnue files load and evaluate through the same model.
// Outer UCI/search score scaling remains Whale-specific.
//
// What is intentionally Whale-specific:
// - HalfKA/PSQT are incrementally maintained; threats/pairs recompute per eval,
// - optional custom "RUDI" checkpoint layout,
// - file loading only (no embedding of Stockfish weights in this repo).
// - the engine only uses a v16 net after an explicit `setoption EvalFile`;
//   there is no implicit auto-load during evaluation.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::board::state::BoardState;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

pub const VERSION: u8 = 16;
pub const N_BUCKETS: usize = 8;

// ---- Feature dimensions (Stockfish HalfKAv2_hm + FullThreats) ----
pub const PSQ_DIMS: usize = 22_528;
pub const THREAT_DIMS: usize = 59_808;
pub const PAIR_DIMS: usize = 4_560;
pub const BIG_INPUT_DIMS: usize = PSQ_DIMS + THREAT_DIMS + PAIR_DIMS; // 86_896
pub const PS_PLANES: usize = 704; // 11 * 64
pub const KING_BUCKET_COUNT: usize = 32;

// ---- Layer sizes ----
pub const L1: usize = 1024;

pub const FC0_OUT: usize = 32;
pub const FC0_ACT: usize = 32;
pub const FC1_IN: usize = 64; // 32 sqr + 32 clip
pub const FC1_OUT: usize = 32; // L3

// ---- Quantization / scales (nnue_common.h) ----
pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: u32 = 6;
pub const SF_FILE_VERSION: u32 = 0x7AF32F20;
pub const SF17_FILE_VERSION: u32 = 0x6A448AFA;
pub const PSQ_HASH: u32 = 0x7f234cb8;
pub const THREAT_HASH: u32 = 0x2e6b9d04;
pub const PAIR_HASH: u32 = 0x86f2b1dd;
const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

// ---- Gate / combine (evaluate.cpp) ----
pub const PAWN_VALUE: i32 = 208;
pub const KNIGHT_VALUE: i32 = 781;
pub const BISHOP_VALUE: i32 = 825;
pub const ROOK_VALUE: i32 = 1276;
pub const QUEEN_VALUE: i32 = 2538;
pub const SMALL_GATE: i32 = 962;
pub const SMALL_FALLBACK: i32 = 236;

// max active features: HalfKA pieces (<=32) + official threat/pair lists.
pub const MAX_THREAT_ACTIVE: usize = 256;
pub const MAX_PAIR_ACTIVE: usize = 256;
pub const MAX_ACTIVE: usize = 32 + MAX_THREAT_ACTIVE + MAX_PAIR_ACTIVE;

// ---------------------------------------------------------------------------
// Square helpers. Internal feature math uses Stockfish numbering (A1 = 0);
// Whale squares are vertically flipped (A8 = 0), converted with `^ 56`.
// ---------------------------------------------------------------------------

#[inline(always)]
fn to_sf(sq: usize) -> usize {
    sq ^ 56
}

// KingBuckets table (Stockfish half_ka_v2_hm.h), bucket ids in A1=0 order.
const KING_BUCKETS: [u32; 64] = [
    28, 29, 30, 31, 31, 30, 29, 28, 24, 25, 26, 27, 27, 26, 25, 24, 20, 21, 22, 23, 23, 22, 21, 20,
    16, 17, 18, 19, 19, 18, 17, 16, 12, 13, 14, 15, 15, 14, 13, 12, 8, 9, 10, 11, 11, 10, 9, 8, 4,
    5, 6, 7, 7, 6, 5, 4, 0, 1, 2, 3, 3, 2, 1, 0,
];

// HalfKA OrientTBL: mirror files when the king stands on a-d files.
#[inline(always)]
fn halfka_orient(ksq_sf: usize) -> usize {
    if ksq_sf & 7 < 4 { 7 } else { 0 }
}

// Stockfish piece codes: W_PAWN..W_KING = 1..6, B_* = 9..14.
#[inline(always)]
fn sf_piece_code(side: Side, piece: Piece) -> usize {
    let base = piece as usize + 1; // Pawn=0..King=5 -> 1..6
    if side == Side::Black { base + 8 } else { base }
}

// ---------------------------------------------------------------------------
// Position view in Stockfish numbering shared by features and trainer.
// ---------------------------------------------------------------------------

pub struct SfnnPosition {
    pub pieces: [u64; 6], // per Whale Piece discriminant, A1=0 bitboards
    pub white: u64,
    pub black: u64,
    // 0..5 = piece type, 6 = empty, in A1=0 order
    pub mapping: [u8; 64],
}

impl SfnnPosition {
    pub fn from_board(board: &BoardState) -> Self {
        let mut pieces = [0u64; 6];
        for &p in &Piece::ALL {
            pieces[p as usize] = board.pieces[p].0.swap_bytes();
        }
        let mut mapping = [6u8; 64];
        for (i, mapped_piece) in mapping.iter_mut().enumerate() {
            let rp = board.piece_mapping[i ^ 56];
            if rp != Piece::None {
                *mapped_piece = rp as u8;
            }
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

// ---------------------------------------------------------------------------
// HalfKAv2_hm, exact port of make_index.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// SF-numbered attack helpers (threat geometry only).
// ---------------------------------------------------------------------------

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
fn knight_attacks_sf(sq: usize) -> u64 {
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
fn slider_attacks_sf(sf_piece_type: usize, sq: usize, occ: u64) -> u64 {
    use crate::bitboard::Bitboard;
    use crate::bitboard::lookups::{get_bishop_attacks_from_table, get_rook_attacks_from_table};
    use crate::common::square::Square;

    let whale_sq = Square::from(sq ^ 56);
    let whale_occ = Bitboard(occ.swap_bytes());
    let whale_attacks = match sf_piece_type {
        3 => get_bishop_attacks_from_table(whale_sq, whale_occ).0,
        4 => get_rook_attacks_from_table(whale_sq, whale_occ).0,
        _ => {
            get_bishop_attacks_from_table(whale_sq, whale_occ).0
                | get_rook_attacks_from_table(whale_sq, whale_occ).0
        }
    };
    whale_attacks.swap_bytes()
}

fn pseudo_attacks_sf(piece_code: usize, sq: usize) -> u64 {
    // Empty-board attacks for LUT construction. piece_code: SF codes.
    let pt = piece_code & 7;
    match pt {
        1 => {
            // Pawns: both diagonals on the board (rank edges handled by caller).
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

// Whale Piece discriminant -> SF piece type (1..6)
#[inline(always)]
fn sf_piece_type(pt_whale: usize) -> usize {
    pt_whale + 1
}

// ---------------------------------------------------------------------------
// FullThreats tables (v10 values) + exact make_index.
// ---------------------------------------------------------------------------

const NUM_VALID_TARGETS: [usize; 16] = [0, 4, 10, 8, 8, 10, 0, 0, 0, 4, 10, 8, 8, 10, 0, 0];

const THREAT_MAP: [[i32; 6]; 6] = [
    [-1, 0, -1, 1, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

// OrientTBL[perspective][square]: file-based mirror anchors (A1=0 numbering).
#[inline(always)]
fn threat_orient(perspective: Side, sq_sf: usize) -> usize {
    let left_files = sq_sf & 7 < 4;
    match (perspective, left_files) {
        (Side::White, true) => 0,   // SQ_A1
        (Side::White, false) => 7,  // SQ_H1
        (Side::Black, true) => 56,  // SQ_A8
        (Side::Black, false) => 63, // SQ_H8
        (Side::Both, _) => 0,
    }
}

struct ThreatLuts {
    offsets: [[u32; 66]; 16],
    lut1: [[u32; 16]; 16], // packed: bit1 = always excluded, bit0 = semi, base >> 8
    lut2: [[[u8; 64]; 64]; 16],
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
            if pt != 1 {
                piece_offset += pseudo_attacks_sf(code, from).count_ones();
            } else if (8..56).contains(&from) {
                // Pawns on ranks 2..7 (A1=0 numbering).
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
            // C++ does this sum in unsigned 32-bit (wraps for excluded pairs,
            // whose entries are never used); replicate with wrapping ops.
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

fn threat_luts() -> &'static ThreatLuts {
    static LUTS: OnceLock<ThreatLuts> = OnceLock::new();
    LUTS.get_or_init(build_threat_luts)
}

fn threat_make_index(
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

/// Enumerates every attack edge once, in a fixed order, as
/// `(attacker_code, from_sf, to_sf, attacked_code)` with SF piece codes.
pub fn for_each_threat(pos: &SfnnPosition, mut emit: impl FnMut(usize, usize, usize, usize)) {
    let occ = pos.occupied();
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
                // Pawns: diagonal captures onto occupied squares.
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
                        let victim_side = if (pos.white >> to) & 1 == 1 {
                            Side::White
                        } else {
                            Side::Black
                        };
                        emit(
                            attacker,
                            from,
                            to,
                            sf_piece_code(victim_side, Piece::ALL[victim]),
                        );
                    }
                }
            } else {
                while bb != 0 {
                    let from = bb.trailing_zeros() as usize;
                    bb &= bb - 1;
                    let attacks = match pt {
                        1 => knight_attacks_sf(from),
                        2..=4 => slider_attacks_sf(sf_pt, from, occ),
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
                        let victim_side = if (pos.white >> to) & 1 == 1 {
                            Side::White
                        } else {
                            Side::Black
                        };
                        emit(
                            attacker,
                            from,
                            to,
                            sf_piece_code(victim_side, Piece::ALL[victim]),
                        );
                    }
                }
            }
        }
    }
}

/// Threat index for one perspective (needs that side's king square).
/// Returns None for excluded pairs (mirrors the Dimensions sentinel).
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

// ---------------------------------------------------------------------------
// PP_3Wide pairs
// ---------------------------------------------------------------------------

#[inline(always)]
fn make_pawn_id(color: Side, square_sf: usize) -> usize {
    (color as usize) * 48 + square_sf - 8
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
) -> usize {
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

    let id_a = make_pawn_id(color_oriented, from_oriented);
    let id_b = make_pawn_id(paired_color_oriented, to_oriented);

    let hi = id_a.max(id_b);
    let lo = id_a.min(id_b);

    hi * (hi - 1) / 2 + lo + PSQ_DIMS + THREAT_DIMS
}

pub fn pair_index_for(
    perspective: Side,
    color: Side,
    from_sf: usize,
    to_sf: usize,
    paired_color: Side,
    ksq_sf: usize,
) -> usize {
    pair_make_index(perspective, color, from_sf, to_sf, paired_color, ksq_sf)
        - (PSQ_DIMS + THREAT_DIMS)
}

/// Enumerates every pawn-pair relationship once as
/// `(color, from_sf, to_sf, paired_color)`.
pub fn for_each_pair(pos: &SfnnPosition, mut emit: impl FnMut(Side, usize, usize, Side)) {
    let white = pos.pieces[Piece::Pawn as usize] & pos.white;
    let black = pos.pieces[Piece::Pawn as usize] & pos.black;

    let mut bb = white;
    while bb != 0 {
        let from = bb.trailing_zeros() as usize;
        bb &= bb - 1;
        let band = pawn_pair_bb(from);

        let mut ww = band & bb; // Only pair with remaining white pawns
        while ww != 0 {
            let to = ww.trailing_zeros() as usize;
            ww &= ww - 1;
            emit(Side::White, from, to, Side::White);
        }

        let mut wb = band & black; // Pair with all black pawns
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

        let mut bbk = band & bb; // Only pair with remaining black pawns
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
        out.push(pair_make_index(perspective, color, from, to, paired, ksq));
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
            if len < MAX_PAIR_ACTIVE {
                out[len] = pair_make_index(perspective, Side::White, from, to, Side::White, ksq);
                len += 1;
            }
        }

        let mut wb = band & black;
        while wb != 0 {
            let to = wb.trailing_zeros() as usize;
            wb &= wb - 1;
            if len < MAX_PAIR_ACTIVE {
                out[len] = pair_make_index(perspective, Side::White, from, to, Side::Black, ksq);
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
            if len < MAX_PAIR_ACTIVE {
                out[len] = pair_make_index(perspective, Side::Black, from, to, Side::Black, ksq);
                len += 1;
            }
        }
    }
    len
}

// ---------------------------------------------------------------------------
// Buckets + simple_eval gate (evaluate.cpp).
// ---------------------------------------------------------------------------

pub fn material_bucket(piece_count: usize) -> usize {
    ((piece_count.saturating_sub(1)) / 4).min(N_BUCKETS - 1)
}

pub fn simple_eval(pos: &SfnnPosition, stm: Side) -> i32 {
    let stm_bb = if stm == Side::White {
        pos.white
    } else {
        pos.black
    };
    let ntm_bb = if stm == Side::White {
        pos.black
    } else {
        pos.white
    };
    let mut score = 0i32;
    let table = [
        (Piece::Pawn, PAWN_VALUE),
        (Piece::Knight, KNIGHT_VALUE),
        (Piece::Bishop, BISHOP_VALUE),
        (Piece::Rook, ROOK_VALUE),
        (Piece::Queen, QUEEN_VALUE),
    ];
    for (piece, value) in table {
        let bb = pos.pieces[piece as usize];
        score += value * ((bb & stm_bb).count_ones() as i32 - (bb & ntm_bb).count_ones() as i32);
    }
    score
}

pub fn affine_hash(out_dims: u32, prev: u32) -> u32 {
    let mut h = 0xCC03DAE4u32.wrapping_add(out_dims);
    h ^= prev >> 1;
    h ^= prev << 31;
    h
}

fn relu_hash(prev: u32) -> u32 {
    0x538D24C7u32.wrapping_add(prev)
}

fn rotl1(x: u32) -> u32 {
    x.rotate_left(1)
}

fn combine_hash(hashes: &[u32]) -> u32 {
    let mut h = 0u32;
    for &c in hashes {
        h = rotl1(h) ^ c;
    }
    h
}

pub fn arch_hash(l1: u32) -> u32 {
    // fc_0 -> ac_0 -> fc_1 -> ac_1 -> fc_2 (ac_sqr omitted, like Stockfish).
    let mut h = 0xEC42E90Du32 ^ (l1 * 2);
    h = affine_hash(FC0_OUT as u32, h);
    h = relu_hash(h);
    h = affine_hash(FC1_OUT as u32, h);
    h = relu_hash(h);
    h = affine_hash(1, h);
    h
}

pub fn transformer_hash(use_threats: bool, l1: u32) -> u32 {
    if use_threats {
        combine_hash(&[THREAT_HASH, PAIR_HASH, PSQ_HASH]) ^ (l1 * 2)
    } else {
        PSQ_HASH ^ (l1 * 2)
    }
}

pub fn network_hash(use_threats: bool, l1: u32) -> u32 {
    transformer_hash(use_threats, l1) ^ arch_hash(l1)
}

// De-scramble table for SSSE3ChunkSize=4 fully-connected weights.
#[allow(dead_code)]
fn unscramble_table(out_dims: usize, padded_in: usize) -> Vec<usize> {
    let n = out_dims * padded_in;
    let mut inv = vec![0usize; n];
    for i in 0..n {
        let m = (i / 4) % (padded_in / 4) * out_dims * 4 + (i / padded_in) * 4 + i % 4;
        inv[m] = i;
    }
    inv
}

fn read_u32_le(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated u32");
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    Ok(v)
}

fn read_i32_le(data: &[u8], pos: &mut usize) -> Result<i32, &'static str> {
    Ok(read_u32_le(data, pos)? as i32)
}

fn read_leb128_section<T, F>(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    decode: F,
) -> Result<Vec<T>, &'static str>
where
    F: Fn(&[u8], &mut usize) -> Result<T, &'static str>,
{
    if *pos + LEB128_MAGIC.len() > data.len() {
        return Err("truncated leb128 magic");
    }
    if &data[*pos..*pos + LEB128_MAGIC.len()] != LEB128_MAGIC {
        return Err("bad leb128 magic");
    }
    *pos += LEB128_MAGIC.len();
    let _bytes = read_u32_le(data, pos)?;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(decode(data, pos)?);
    }
    Ok(out)
}

fn decode_leb128_i64(data: &[u8], pos: &mut usize, bits: u32) -> Result<i64, &'static str> {
    let mut result: i64 = 0;
    let mut shift = 0u32;
    loop {
        if *pos >= data.len() {
            return Err("truncated leb128 payload");
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as i64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            if shift < bits && (byte & 0x40) != 0 {
                result |= !0i64 << shift;
            }
            break;
        }
        if shift >= bits + 7 {
            return Err("leb128 overflow");
        }
    }
    Ok(result)
}

fn decode_leb128_i16(data: &[u8], pos: &mut usize) -> Result<i16, &'static str> {
    Ok(decode_leb128_i64(data, pos, 16)? as i16)
}

fn decode_leb128_i32(data: &[u8], pos: &mut usize) -> Result<i32, &'static str> {
    Ok(decode_leb128_i64(data, pos, 32)? as i32)
}

#[derive(Clone, Debug)]
pub struct SfnnArch {
    pub fc0_bias: [i32; FC0_OUT],
    pub fc0_w: Vec<i8>,
    pub fc1_bias: [i32; FC1_OUT],
    pub fc1_w: Vec<i8>,
    pub fc2_bias: i32,
    pub fc2_w: [i8; FC0_OUT * 2 + FC1_OUT * 2],
    pub is_rudi: bool,
}

impl SfnnArch {
    fn load(
        data: &[u8],
        pos: &mut usize,
        fc0_in: usize,
        expected_hash: u32,
    ) -> Result<Self, &'static str> {
        let hash = read_u32_le(data, pos)?;
        if hash != expected_hash {
            return Err("bad arch hash");
        }
        let mut fc0_bias = [0i32; FC0_OUT];
        for b in fc0_bias.iter_mut() {
            *b = read_i32_le(data, pos)?;
        }
        let padded0 = fc0_in.next_multiple_of(32);
        let mut raw0 = vec![0i8; FC0_OUT * padded0];
        for v in raw0.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc0 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }
        let fc0_w = raw0;

        let mut fc1_bias = [0i32; FC1_OUT];
        for b in fc1_bias.iter_mut() {
            *b = read_i32_le(data, pos)?;
        }
        let mut raw1 = vec![0i8; FC1_OUT * 64];
        for v in raw1.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc1 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }
        let fc1_w = raw1;

        let fc2_bias = read_i32_le(data, pos)?;
        let mut fc2_w = [0i8; FC1_IN + FC1_OUT * 2];
        for v in fc2_w.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc2 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }

        Ok(Self {
            fc0_bias,
            fc0_w,
            fc1_bias,
            fc1_w,
            fc2_bias,
            fc2_w,
            is_rudi: false,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct SfnnTransformer {
    pub bias: Vec<i16>,
    pub weights: Vec<i16>,       // PSQ_DIMS x l1, [feat * l1 + j]
    pub threat_w: Vec<i8>,       // THREAT_DIMS x l1 (empty for small net)
    pub threat_w_i16: Vec<i16>,  // For Rudi net
    pub psqt_w: Vec<i32>,        // PSQ_DIMS x 8, [feat * 8 + b]
    pub threat_psqt_w: Vec<i32>, // THREAT_DIMS x 8 (empty for small net)
    pub pair_w: Vec<i8>,         // PAIR_DIMS x l1 (empty for small net)
    pub pair_w_i16: Vec<i16>,    // For Rudi net
    pub pair_psqt_w: Vec<i32>,   // PAIR_DIMS x 8 (empty for small net)
}

#[derive(Clone, Debug)]
pub struct Sfnn16Net {
    pub l1: usize,
    pub use_threats: bool,
    pub is_rudi: bool,
    pub transformer: SfnnTransformer,
    pub stacks: Vec<SfnnArch>, // N_BUCKETS entries
}

impl Sfnn16Net {
    pub fn load_bytes(data: &[u8], use_threats: bool, l1: usize) -> Result<Self, &'static str> {
        let mut pos = 0;
        let version = read_u32_le(data, &mut pos)?;
        if version != SF_FILE_VERSION && version != SF17_FILE_VERSION {
            return Err("bad SFNN file version");
        }
        let file_hash = read_u32_le(data, &mut pos)?;
        if use_threats && file_hash != network_hash(true, l1 as u32) {
            return Err("bad SFNN file hash");
        }
        let desc_len = read_u32_le(data, &mut pos)? as usize;
        if pos + desc_len > data.len() {
            return Err("truncated description");
        }
        pos += desc_len;

        let thash = read_u32_le(data, &mut pos)?;
        if use_threats && thash != transformer_hash(true, l1 as u32) {
            return Err("bad transformer hash");
        }
        let bias = read_leb128_section(data, &mut pos, l1, decode_leb128_i16)?;

        let psq_inputs = PSQ_DIMS;
        let thr_inputs = if use_threats { THREAT_DIMS } else { 0 };
        let pair_inputs = if use_threats { PAIR_DIMS } else { 0 };

        let (weights, psqt_w, threat_w, threat_psqt_w, pair_w, pair_psqt_w) = if use_threats {
            let thr_bytes = thr_inputs * l1;
            let is_sf17 = pos + thr_bytes + LEB128_MAGIC.len() <= data.len()
                && &data[pos + thr_bytes..pos + thr_bytes + LEB128_MAGIC.len()] == LEB128_MAGIC;

            if is_sf17 {
                let tw = data[pos..pos + thr_bytes]
                    .iter()
                    .map(|&b| b as i8)
                    .collect::<Vec<i8>>();
                pos += thr_bytes;

                let tp =
                    read_leb128_section(data, &mut pos, thr_inputs * N_BUCKETS, decode_leb128_i32)?;

                let pair_bytes = pair_inputs * l1;
                if pos + pair_bytes > data.len() {
                    return Err("truncated pair weights");
                }
                let pw = data[pos..pos + pair_bytes]
                    .iter()
                    .map(|&b| b as i8)
                    .collect::<Vec<i8>>();
                pos += pair_bytes;

                let pp = read_leb128_section(
                    data,
                    &mut pos,
                    pair_inputs * N_BUCKETS,
                    decode_leb128_i32,
                )?;

                let w = read_leb128_section(data, &mut pos, psq_inputs * l1, decode_leb128_i16)?;
                let pw_psqt =
                    read_leb128_section(data, &mut pos, psq_inputs * N_BUCKETS, decode_leb128_i32)?;

                (w, pw_psqt, tw, tp, pw, pp)
            } else {
                let combined = read_leb128_section(
                    data,
                    &mut pos,
                    (thr_inputs + psq_inputs) * l1,
                    decode_leb128_i16,
                )?;
                let (t, m) = combined.split_at(thr_inputs * l1);
                let w = m.to_vec();
                let tw = t.iter().map(|&v| v as i8).collect::<Vec<i8>>();

                let combined_psqt = read_leb128_section(
                    data,
                    &mut pos,
                    (thr_inputs + psq_inputs) * N_BUCKETS,
                    decode_leb128_i32,
                )?;
                let (tp_slice, pw_slice) = combined_psqt.split_at(thr_inputs * N_BUCKETS);
                let pw_psqt = pw_slice.to_vec();
                let tp = tp_slice.to_vec();

                (w, pw_psqt, tw, tp, Vec::new(), Vec::new())
            }
        } else {
            let w = read_leb128_section(data, &mut pos, psq_inputs * l1, decode_leb128_i16)?;
            let pw_psqt =
                read_leb128_section(data, &mut pos, psq_inputs * N_BUCKETS, decode_leb128_i32)?;
            (w, pw_psqt, Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        let ahash = arch_hash(l1 as u32);
        let mut stacks = Vec::with_capacity(N_BUCKETS);
        for _ in 0..N_BUCKETS {
            stacks.push(SfnnArch::load(data, &mut pos, l1, ahash)?);
        }
        if pos != data.len() {
            return Err("trailing data after network");
        }

        Ok(Self {
            l1,
            use_threats,
            is_rudi: false,
            transformer: SfnnTransformer {
                bias,
                weights,
                threat_w,
                threat_w_i16: Vec::new(),
                psqt_w,
                threat_psqt_w,
                pair_w,
                pair_w_i16: Vec::new(),
                pair_psqt_w,
            },
            stacks,
        })
    }

    pub fn load_rudi(data: &[u8]) -> Result<Self, &'static str> {
        let raw_payload = if data.starts_with(b"RUDI") {
            let mut header_offset = 4;
            let payload_len = read_u32_le(data, &mut header_offset)? as usize;
            if !matches!(payload_len, 181_011_108 | 181_011_136)
                || data.len() != header_offset + payload_len
            {
                return Err("unsupported RUDI payload layout");
            }
            &data[header_offset..]
        } else if data.len() == 181_011_136 || data.len() == 181_011_108 {
            data
        } else {
            return Err("not a whale net");
        };

        let mut offset = 0;
        let l0w_slice = read_rudi_i16(raw_payload, &mut offset, BIG_INPUT_DIMS * L1)?;

        let l0b_slice = read_rudi_i16(raw_payload, &mut offset, L1)?;

        let l1w_slice = read_rudi_i8(raw_payload, &mut offset, L1 * N_BUCKETS * FC0_OUT)?;

        let l1b_slice = read_rudi_i32(raw_payload, &mut offset, N_BUCKETS * FC0_OUT)?;

        let l2w_slice = read_rudi_i8(raw_payload, &mut offset, FC1_IN * FC1_OUT)?;

        let l2b_slice = read_rudi_i32(raw_payload, &mut offset, FC1_OUT)?;

        let outw_slice = read_rudi_i8(raw_payload, &mut offset, FC1_OUT)?;

        let outb_slice = read_rudi_i32(raw_payload, &mut offset, 1)?;

        let psqt_slice = read_rudi_i32(raw_payload, &mut offset, N_BUCKETS * BIG_INPUT_DIMS)?;

        let weights = l0w_slice[..PSQ_DIMS * L1].to_vec();
        let threat_w_i16 = l0w_slice[PSQ_DIMS * L1..(PSQ_DIMS + THREAT_DIMS) * L1].to_vec();
        let pair_w_i16 = l0w_slice[(PSQ_DIMS + THREAT_DIMS) * L1..BIG_INPUT_DIMS * L1].to_vec();

        let bias = l0b_slice.to_vec();
        let psqt_w = psqt_slice[..PSQ_DIMS * N_BUCKETS].to_vec();
        let threat_psqt_w =
            psqt_slice[PSQ_DIMS * N_BUCKETS..(PSQ_DIMS + THREAT_DIMS) * N_BUCKETS].to_vec();
        let pair_psqt_w =
            psqt_slice[(PSQ_DIMS + THREAT_DIMS) * N_BUCKETS..BIG_INPUT_DIMS * N_BUCKETS].to_vec();

        let mut stacks = Vec::with_capacity(N_BUCKETS);
        for b in 0..N_BUCKETS {
            let mut fc0_bias = [0i32; FC0_OUT];
            fc0_bias.copy_from_slice(&l1b_slice[b * FC0_OUT..(b + 1) * FC0_OUT]);

            let mut fc0_w = vec![0i8; FC0_OUT * L1];
            for o in 0..FC0_OUT {
                for j in 0..L1 {
                    fc0_w[o * L1 + j] = l1w_slice[j * (N_BUCKETS * FC0_OUT) + b * FC0_OUT + o];
                }
            }

            let mut fc1_bias = [0i32; FC1_OUT];
            fc1_bias.copy_from_slice(&l2b_slice);

            let mut fc1_w = vec![0i8; FC1_OUT * FC1_IN];
            for o in 0..FC1_OUT {
                for j in 0..FC1_IN {
                    fc1_w[o * FC1_IN + j] = l2w_slice[j * FC1_OUT + o];
                }
            }

            let fc2_bias = outb_slice[0];
            let mut fc2_w = [0i8; FC0_OUT * 2 + FC1_OUT * 2];
            fc2_w[..FC1_OUT].copy_from_slice(&outw_slice);

            stacks.push(SfnnArch {
                fc0_bias,
                fc0_w,
                fc1_bias,
                fc1_w,
                fc2_bias,
                fc2_w,
                is_rudi: true,
            });
        }

        Ok(Self {
            l1: L1,
            use_threats: true,
            is_rudi: true,
            transformer: SfnnTransformer {
                bias,
                weights,
                threat_w: Vec::new(),
                threat_w_i16,
                psqt_w,
                threat_psqt_w,
                pair_w: Vec::new(),
                pair_w_i16,
                pair_psqt_w,
            },
            stacks,
        })
    }

    pub fn load_file(path: &str, use_threats: bool, l1: usize) -> Result<Self, &'static str> {
        let bytes = std::fs::read(path).map_err(|_| "cannot read network file")?;
        if (bytes.len() >= 4 && &bytes[0..4] == b"RUDI")
            || bytes.len() == 181_011_136
            || bytes.len() == 181_011_108
        {
            return Self::load_rudi(&bytes);
        }
        Self::load_bytes(&bytes, use_threats, l1)
    }
}

fn read_rudi_i8(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i8>, &'static str> {
    if *offset + count > data.len() {
        return Err("truncated whale payload");
    }
    let out = data[*offset..*offset + count]
        .iter()
        .map(|&b| b as i8)
        .collect();
    *offset += count;
    Ok(out)
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_rudi_i16(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i16>, &'static str> {
    let bytes = count.checked_mul(2).ok_or("bad whale length")?;
    if *offset + bytes > data.len() {
        return Err("truncated whale payload");
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[*offset..*offset + bytes].chunks_exact(2) {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    *offset += bytes;
    Ok(out)
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_rudi_i32(data: &[u8], offset: &mut usize, count: usize) -> Result<Vec<i32>, &'static str> {
    let bytes = count.checked_mul(4).ok_or("bad whale length")?;
    if *offset + bytes > data.len() {
        return Err("truncated whale payload");
    }
    let mut out = Vec::with_capacity(count);
    for chunk in data[*offset..*offset + bytes].chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    *offset += bytes;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Inference (scalar + AVX2 paths).
// ---------------------------------------------------------------------------

// SAFETY invariants for all AVX2 routines below:
//  - Callers verify AVX2 availability via `is_x86_feature_detected!("avx2")`
//    at runtime before invoking any `*_avx2` function.
//  - Buffers and slices passed to these functions have fixed dimensions
//    proportional to SIMD vector widths (e.g. `FC0_OUT = 32`, `FC1_OUT = 32`,
//    `L1 = 1024`, `N_BUCKETS = 8`). Loop indices are strictly bounded by
//    these compile-time constants.
//  - Memory reads use `_mm256_loadu_si256` for potentially unaligned slices,
//    preventing alignment fault exceptions.

#[inline(always)]
fn div_trunc(a: i32, b: i32) -> i32 {
    a / b
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_ps_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    unsafe {
        use std::arch::x86_64::*;
        let hi128 = _mm256_extracti128_si256(v, 1);
        let lo128 = _mm256_castsi256_si128(v);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let shuf = _mm_shuffle_epi32(sum128, 0x4E);
        let sum64 = _mm_add_epi32(sum128, shuf);
        let shuf2 = _mm_shuffle_epi32(sum64, 0x05);
        _mm_cvtsi128_si32(_mm_add_epi32(sum64, shuf2))
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fc0_avx2(arch: &SfnnArch, in_row: &[u8], fc0: &mut [i32; FC0_OUT]) {
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let mask = _mm256_set1_epi8(0x7f);
        let in_ptr = in_row.as_ptr() as *const __m256i;
        let chunks = in_row.len() / 32;
        let l1 = in_row.len();

        let mut o = 0;
        while o < FC0_OUT {
            let mut sum0 = _mm256_setzero_si256();
            let mut sum1 = _mm256_setzero_si256();
            let mut sum2 = _mm256_setzero_si256();
            let mut sum3 = _mm256_setzero_si256();

            let w0_ptr = arch.fc0_w.as_ptr().add(o * l1) as *const __m256i;
            let w1_ptr = arch.fc0_w.as_ptr().add((o + 1) * l1) as *const __m256i;
            let w2_ptr = arch.fc0_w.as_ptr().add((o + 2) * l1) as *const __m256i;
            let w3_ptr = arch.fc0_w.as_ptr().add((o + 3) * l1) as *const __m256i;

            for j in 0..chunks {
                let in_vec = _mm256_loadu_si256(in_ptr.add(j));
                let low = _mm256_and_si256(in_vec, mask);
                let high = _mm256_andnot_si256(mask, in_vec);

                let w0 = _mm256_loadu_si256(w0_ptr.add(j));
                let w1 = _mm256_loadu_si256(w1_ptr.add(j));
                let w2 = _mm256_loadu_si256(w2_ptr.add(j));
                let w3 = _mm256_loadu_si256(w3_ptr.add(j));

                let low32_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w0), ones);
                let high32_0 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w0), ones);
                sum0 = _mm256_add_epi32(sum0, _mm256_add_epi32(low32_0, high32_0));

                let low32_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w1), ones);
                let high32_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w1), ones);
                sum1 = _mm256_add_epi32(sum1, _mm256_add_epi32(low32_1, high32_1));

                let low32_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w2), ones);
                let high32_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w2), ones);
                sum2 = _mm256_add_epi32(sum2, _mm256_add_epi32(low32_2, high32_2));

                let low32_3 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w3), ones);
                let high32_3 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w3), ones);
                sum3 = _mm256_add_epi32(sum3, _mm256_add_epi32(low32_3, high32_3));
            }

            fc0[o] = arch.fc0_bias[o] + hsum256_ps_avx2(sum0);
            fc0[o + 1] = arch.fc0_bias[o + 1] + hsum256_ps_avx2(sum1);
            fc0[o + 2] = arch.fc0_bias[o + 2] + hsum256_ps_avx2(sum2);
            fc0[o + 3] = arch.fc0_bias[o + 3] + hsum256_ps_avx2(sum3);

            o += 4;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fc1_avx2(arch: &SfnnArch, concat_fc0: &[u8], fc1: &mut [i32; FC1_OUT]) {
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let in_ptr = concat_fc0.as_ptr() as *const __m256i;
        let in0 = _mm256_loadu_si256(in_ptr);
        let in1 = _mm256_loadu_si256(in_ptr.add(1));

        for (o, output) in fc1.iter_mut().enumerate() {
            let w_ptr = arch.fc1_w.as_ptr().add(o * 64) as *const __m256i;
            let w0 = _mm256_loadu_si256(w_ptr);
            let w1 = _mm256_loadu_si256(w_ptr.add(1));

            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, w0), ones);
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, w1), ones);
            let sum = _mm256_add_epi32(p0, p1);

            *output = arch.fc1_bias[o] + hsum256_ps_avx2(sum);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fc2_avx2(arch: &SfnnArch, concat: &[u8]) -> i32 {
    unsafe {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi16(1);
        let in_ptr = concat.as_ptr() as *const __m256i;
        let w_ptr = arch.fc2_w.as_ptr() as *const __m256i;

        let in0 = _mm256_loadu_si256(in_ptr);
        let in1 = _mm256_loadu_si256(in_ptr.add(1));
        let in2 = _mm256_loadu_si256(in_ptr.add(2));
        let in3 = _mm256_loadu_si256(in_ptr.add(3));

        let w0 = _mm256_loadu_si256(w_ptr);
        let w1 = _mm256_loadu_si256(w_ptr.add(1));
        let w2 = _mm256_loadu_si256(w_ptr.add(2));
        let w3 = _mm256_loadu_si256(w_ptr.add(3));

        let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, w0), ones);
        let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, w1), ones);
        let p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(in2, w2), ones);
        let p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(in3, w3), ones);

        let sum = _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
        hsum256_ps_avx2(sum)
    }
}

fn propagate(arch: &SfnnArch, l1: usize, input: &[u8]) -> i32 {
    if arch.is_rudi {
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = has_avx2() && l1.is_multiple_of(32);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        let mut fc0 = [0.0f32; FC0_OUT];
        if use_avx2 {
            let mut fc0_i32 = [0i32; FC0_OUT];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                fc0_avx2(arch, &input[..l1], &mut fc0_i32);
            }
            for (output, &v) in fc0.iter_mut().zip(fc0_i32.iter()) {
                *output = (v as f32 / 16320.0).clamp(0.0, 1.0);
            }
        } else {
            for (o, output) in fc0.iter_mut().enumerate() {
                let mut sum = arch.fc0_bias[o];
                let row = o * l1;
                for (&input_value, &weight) in input.iter().zip(&arch.fc0_w[row..row + l1]) {
                    sum += i32::from(input_value) * i32::from(weight);
                }
                *output = (sum as f32 / 16320.0).clamp(0.0, 1.0);
            }
        }

        let mut pair = [0.0f32; FC1_IN];
        for (o, &c) in fc0.iter().enumerate() {
            pair[o] = c * c;
            pair[FC0_OUT + o] = c;
        }

        let mut fc1 = [0.0f32; FC1_OUT];
        for (o, output) in fc1.iter_mut().enumerate() {
            let mut sum = arch.fc1_bias[o] as f32 / 16320.0;
            let row = o * FC1_IN;
            for (&pair_value, &weight) in pair.iter().zip(&arch.fc1_w[row..row + FC1_IN]) {
                sum += pair_value * (weight as f32 / 64.0);
            }
            *output = sum.clamp(0.0, 1.0);
        }

        let mut out = arch.fc2_bias as f32 / 16320.0;
        for (&fc1_value, &weight) in fc1.iter().zip(&arch.fc2_w[..FC1_OUT]) {
            out += fc1_value * (weight as f32 / 64.0);
        }

        let score_cp = ((out - 2.80) * 100.0).clamp(-29000.0, 29000.0) as i32;
        return score_cp * OUTPUT_SCALE;
    }

    let mut fc0 = [0i32; FC0_OUT];
    let in_row = &input[..l1];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2() && l1 == 1024;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc0_avx2(arch, in_row, &mut fc0);
        }
    } else {
        for (o, output) in fc0.iter_mut().enumerate() {
            let mut value = arch.fc0_bias[o];
            let row = o * l1;
            for (&input_value, &weight) in in_row.iter().zip(&arch.fc0_w[row..row + l1]) {
                value += i32::from(input_value) * i32::from(weight);
            }
            *output = value;
        }
    }

    let mut concat = [0u8; FC0_OUT * 2 + FC1_OUT * 2];
    for (i, &v) in fc0.iter().enumerate() {
        let sqr = (((v as i64 * v as i64) >> 21).min(127)) as u8;
        let clip = (v >> 7).clamp(0, 127) as u8;
        concat[i] = sqr;
        concat[FC0_OUT + i] = clip;
    }

    let mut fc1 = [0i32; FC1_OUT];
    let concat_fc0 = &concat[0..FC0_OUT * 2];

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc1_avx2(arch, concat_fc0, &mut fc1);
        }
    } else {
        for (o, output) in fc1.iter_mut().enumerate() {
            let mut value = arch.fc1_bias[o];
            let row = o * (FC0_OUT * 2);
            for (&input_value, &weight) in
                concat_fc0.iter().zip(&arch.fc1_w[row..row + FC0_OUT * 2])
            {
                value += i32::from(input_value) * i32::from(weight);
            }
            *output = value;
        }
    }

    for (i, &v) in fc1.iter().enumerate() {
        let sqr = (((v as i64 * v as i64) >> 19).min(127)) as u8;
        let clip = (v >> 6).clamp(0, 127) as u8;
        concat[FC0_OUT * 2 + i] = sqr;
        concat[FC0_OUT * 2 + FC1_OUT + i] = clip;
    }

    let mut out = arch.fc2_bias;
    let dot = if use_avx2 && arch.fc2_w.len() == 128 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc2_avx2(arch, &concat)
        }
        #[cfg(not(target_arch = "x86_64"))]
        0
    } else {
        concat
            .iter()
            .zip(arch.fc2_w.iter())
            .map(|(&input_value, &weight)| i32::from(input_value) * i32::from(weight))
            .sum()
    };
    out += dot;
    let skip = fc0[FC0_OUT - 2] - fc0[FC0_OUT - 1];
    out += skip;

    let multiplier = 600 * OUTPUT_SCALE as i64;
    let denominator = 128 * 64 * 2;
    ((out as i64 * multiplier) / denominator) as i32
}

// ---------------------------------------------------------------------------
// Incremental accumulators (HalfKA + PSQT; threats are per-eval).
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Sfnn16Accs {
    pub halfka: [[i16; L1]; 2],
    pub psqt: [[i32; N_BUCKETS]; 2],
    pub threat_psqt: [[i32; N_BUCKETS]; 2],
    pub generation: u64,
}

impl Sfnn16Accs {
    pub fn empty() -> Self {
        Self {
            halfka: [[0i16; L1]; 2],
            psqt: [[0i32; N_BUCKETS]; 2],
            threat_psqt: [[0i32; N_BUCKETS]; 2],
            generation: 0,
        }
    }
}

pub struct LoadedNets {
    pub net: Sfnn16Net,
}

static NETS: RwLock<Option<Arc<LoadedNets>>> = RwLock::new(None);
static ACTIVE_NET: std::sync::atomic::AtomicPtr<LoadedNets> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static NETS_GEN: AtomicU64 = AtomicU64::new(0);
static PENDING_PATH: RwLock<Option<String>> = RwLock::new(None);

#[cfg(target_arch = "x86_64")]
static HAS_AVX2: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| is_x86_feature_detected!("avx2"));

#[inline(always)]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        *HAS_AVX2
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
pub fn active_loaded_net() -> Option<&'static LoadedNets> {
    let ptr = ACTIVE_NET.load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(&*ptr) }
    }
}

#[allow(dead_code)]
fn loaded_nets() -> Option<Arc<LoadedNets>> {
    NETS.read().ok().and_then(|guard| guard.clone())
}

pub fn try_load_default_path() -> Option<&'static str> {
    if maintenance_active() {
        return Some("active");
    }
    let candidates = [
        "models/whale_big.nnue",
        "models/whale_medium.nnue",
        "models/whale_small.nnue",
        "whale_big.nnue",
        "whale_medium.nnue",
        "whale_small.nnue",
        "whale_farseer_final.nnue",
        "v16/whale_farseer_final.nnue",
        "whale_farseerT76.nnue",
        "v16/whale_farseerT76.nnue",
        "v16/nn-1a298aa575a0.nnue",
        "v16/whale.nnue",
        "v16/quantised.bin",
        "whale.nnue",
        "quantised.bin",
        // Legacy Rudim names (backward compat)
        "rudim_farseer_final.nnue",
        "v16/rudim_farseer_final.nnue",
        "rudim_farseerT76.nnue",
        "v16/rudim_farseerT76.nnue",
        "v16/rudim.nnue",
        "rudim.nnue",
        "resources/sfnn16-big-checkpoint.bin",
    ];
    candidates
        .into_iter()
        .find(|&path| std::path::Path::new(path).exists() && load_net(path).is_ok())
}

pub fn try_load_default() -> bool {
    try_load_default_path().is_some()
}

#[inline(always)]
pub fn maintenance_active() -> bool {
    !ACTIVE_NET.load(Ordering::Relaxed).is_null()
}

fn current_gen() -> u64 {
    NETS_GEN.load(Ordering::SeqCst)
}

/// Load an SFNNv16 file. On success the network activates (bumps the
/// generation so all accumulator entries refresh lazily); on failure the
/// previous network (if any) is kept.
pub fn load_net(path: &str) -> Result<(), &'static str> {
    let net =
        Sfnn16Net::load_file(path, true, L1).or_else(|_| Sfnn16Net::load_file(path, false, L1))?;
    let loaded = Arc::new(LoadedNets { net });
    ACTIVE_NET.store(Arc::as_ptr(&loaded) as *mut LoadedNets, Ordering::Release);
    if let Ok(mut guard) = NETS.write() {
        *guard = Some(loaded);
    }
    if let Ok(mut guard) = PENDING_PATH.write() {
        *guard = Some(path.to_string());
    }
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    crate::eval::nnue::clear_eval_cache();
    Ok(())
}

pub fn unload_nets() {
    ACTIVE_NET.store(std::ptr::null_mut(), Ordering::Release);
    if let Ok(mut guard) = NETS.write() {
        *guard = None;
    }
    if let Ok(mut guard) = PENDING_PATH.write() {
        *guard = None;
    }
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    crate::eval::nnue::clear_eval_cache();
}

pub fn resolve_model_path(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "<empty>" || trimmed.eq_ignore_ascii_case("embedded") {
        return None;
    }
    if std::path::Path::new(trimmed).exists() {
        return Some(trimmed.to_string());
    }
    let candidates = [
        format!("models/{trimmed}"),
        format!("models/{trimmed}.nnue"),
        format!("models/whale_{trimmed}.nnue"),
    ];
    for candidate in candidates {
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    Some(trimmed.to_string())
}

pub fn active_model_name() -> String {
    if let Ok(guard) = PENDING_PATH.read()
        && let Some(ref p) = *guard
    {
        let path = std::path::Path::new(p);
        return path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(p)
            .to_string();
    }
    "embedded".to_string()
}

pub fn set_eval_file(which: &str, path: &str) -> Result<&'static str, &'static str> {
    if which.eq_ignore_ascii_case("EvalFileSmall") {
        return Ok("EvalFileSmall is deprecated and ignored (SFNNv16 uses a single network)");
    }
    if which.eq_ignore_ascii_case("EvalFile") || which.eq_ignore_ascii_case("Model") {
        let resolved = resolve_model_path(path);
        if let Ok(mut guard) = PENDING_PATH.write() {
            *guard = resolved;
        }
    } else {
        return Err("unknown eval file option");
    }
    try_activate()
}

fn try_activate() -> Result<&'static str, &'static str> {
    let path = PENDING_PATH.read().unwrap().clone();
    match path {
        Some(p) => match load_net(&p) {
            Ok(()) => Ok("SFNNv16 network activated"),
            Err(e) => Err(e),
        },
        None => {
            unload_nets();
            Ok("SFNNv16 network deactivated")
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scatter_halfka_add_avx2(acc: &mut [i16], w: &[i16]) {
    unsafe {
        use std::arch::x86_64::*;
        let a_ptr = acc.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m256i;
        let n = acc.len() / 64;
        for i in 0..n {
            let idx = i * 4;
            let va0 = _mm256_loadu_si256(a_ptr.add(idx));
            let vw0 = _mm256_loadu_si256(w_ptr.add(idx));
            let va1 = _mm256_loadu_si256(a_ptr.add(idx + 1));
            let vw1 = _mm256_loadu_si256(w_ptr.add(idx + 1));
            let va2 = _mm256_loadu_si256(a_ptr.add(idx + 2));
            let vw2 = _mm256_loadu_si256(w_ptr.add(idx + 2));
            let va3 = _mm256_loadu_si256(a_ptr.add(idx + 3));
            let vw3 = _mm256_loadu_si256(w_ptr.add(idx + 3));

            _mm256_storeu_si256(a_ptr.add(idx), _mm256_add_epi16(va0, vw0));
            _mm256_storeu_si256(a_ptr.add(idx + 1), _mm256_add_epi16(va1, vw1));
            _mm256_storeu_si256(a_ptr.add(idx + 2), _mm256_add_epi16(va2, vw2));
            _mm256_storeu_si256(a_ptr.add(idx + 3), _mm256_add_epi16(va3, vw3));
        }
        for i in (n * 4)..(acc.len() / 16) {
            let va = _mm256_loadu_si256(a_ptr.add(i));
            let vw = _mm256_loadu_si256(w_ptr.add(i));
            _mm256_storeu_si256(a_ptr.add(i), _mm256_add_epi16(va, vw));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scatter_halfka_sub_avx2(acc: &mut [i16], w: &[i16]) {
    unsafe {
        use std::arch::x86_64::*;
        let a_ptr = acc.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m256i;
        let n = acc.len() / 64;
        for i in 0..n {
            let idx = i * 4;
            let va0 = _mm256_loadu_si256(a_ptr.add(idx));
            let vw0 = _mm256_loadu_si256(w_ptr.add(idx));
            let va1 = _mm256_loadu_si256(a_ptr.add(idx + 1));
            let vw1 = _mm256_loadu_si256(w_ptr.add(idx + 1));
            let va2 = _mm256_loadu_si256(a_ptr.add(idx + 2));
            let vw2 = _mm256_loadu_si256(w_ptr.add(idx + 2));
            let va3 = _mm256_loadu_si256(a_ptr.add(idx + 3));
            let vw3 = _mm256_loadu_si256(w_ptr.add(idx + 3));

            _mm256_storeu_si256(a_ptr.add(idx), _mm256_sub_epi16(va0, vw0));
            _mm256_storeu_si256(a_ptr.add(idx + 1), _mm256_sub_epi16(va1, vw1));
            _mm256_storeu_si256(a_ptr.add(idx + 2), _mm256_sub_epi16(va2, vw2));
            _mm256_storeu_si256(a_ptr.add(idx + 3), _mm256_sub_epi16(va3, vw3));
        }
        for i in (n * 4)..(acc.len() / 16) {
            let va = _mm256_loadu_si256(a_ptr.add(i));
            let vw = _mm256_loadu_si256(w_ptr.add(i));
            _mm256_storeu_si256(a_ptr.add(i), _mm256_sub_epi16(va, vw));
        }
    }
}

fn scatter_halfka(
    tr: &SfnnTransformer,
    l1: usize,
    feats: &[usize],
    acc: &mut [i16],
    psqt: &mut [i32; N_BUCKETS],
    sign: i16,
) {
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2() && l1.is_multiple_of(16);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for &f in feats {
        debug_assert!(f < PSQ_DIMS);
        let base = f * l1;
        let w_slice = &tr.weights[base..base + l1];
        let acc_slice = &mut acc[..l1];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if sign == 1 {
                    scatter_halfka_add_avx2(acc_slice, w_slice);
                } else {
                    scatter_halfka_sub_avx2(acc_slice, w_slice);
                }
            }
        } else {
            if sign == 1 {
                for (a, &w) in acc_slice.iter_mut().zip(w_slice) {
                    *a = a.wrapping_add(w);
                }
            } else {
                for (a, &w) in acc_slice.iter_mut().zip(w_slice) {
                    *a = a.wrapping_sub(w);
                }
            }
        }
        let pbase = f * N_BUCKETS;
        let pw_slice = &tr.psqt_w[pbase..pbase + N_BUCKETS];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                let p_ptr = psqt.as_mut_ptr() as *mut __m256i;
                let pw_ptr = pw_slice.as_ptr() as *const __m256i;
                let vp = _mm256_loadu_si256(p_ptr);
                let vpw = _mm256_loadu_si256(pw_ptr);
                let res = if sign == 1 {
                    _mm256_add_epi32(vp, vpw)
                } else {
                    _mm256_sub_epi32(vp, vpw)
                };
                _mm256_storeu_si256(p_ptr, res);
            }
        } else {
            if sign == 1 {
                for b in 0..N_BUCKETS {
                    psqt[b] = psqt[b].wrapping_add(pw_slice[b]);
                }
            } else {
                for b in 0..N_BUCKETS {
                    psqt[b] = psqt[b].wrapping_sub(pw_slice[b]);
                }
            }
        }
    }
}

pub fn refresh_perspective(
    pos: &SfnnPosition,
    nets: &LoadedNets,
    perspective: Side,
    accs: &mut Sfnn16Accs,
) {
    let p = perspective as usize;
    let mut feats = Vec::new();
    append_halfka(pos, perspective, &mut feats);
    let slot = &mut accs.halfka[p];
    slot.copy_from_slice(&nets.net.transformer.bias);
    let mut ps = [0i32; N_BUCKETS];
    scatter_halfka(&nets.net.transformer, L1, &feats, slot, &mut ps, 1);
    accs.psqt[p] = ps;
}

#[derive(Clone)]
pub struct FinnyEntry {
    pub halfka: [i16; L1],
    pub psqt: [i32; N_BUCKETS],
    pub occupied: u64,
    pub white: u64,
    pub mapping: [u8; 64],
    pub generation: u64,
    pub valid: bool,
}

impl Default for FinnyEntry {
    fn default() -> Self {
        Self {
            halfka: [0; L1],
            psqt: [0; N_BUCKETS],
            occupied: 0,
            white: 0,
            mapping: [6; 64],
            generation: 0,
            valid: false,
        }
    }
}

std::thread_local! {
    static FINNY_CACHE: RefCell<Option<Box<[[FinnyEntry; 2]; 64]>>> = const { RefCell::new(None) };
}

pub fn update_perspective_finny_or_refresh(
    pos: &SfnnPosition,
    nets: &LoadedNets,
    perspective: Side,
    accs: &mut Sfnn16Accs,
) {
    let pi = perspective as usize;
    let ksq = pos.king_square(perspective);
    let current_generation = current_gen();

    FINNY_CACHE.with(|cache| {
        let mut cache_borrow = cache.borrow_mut();
        let cache_box = cache_borrow.get_or_insert_with(|| {
            let mut v = Vec::with_capacity(64);
            for _ in 0..64 {
                v.push([FinnyEntry::default(), FinnyEntry::default()]);
            }
            v.into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!())
        });
        let entry = &mut cache_box[ksq][pi];

        if entry.valid && entry.generation == current_generation {
            let cur_occ = pos.occupied();
            let entry_occ = entry.occupied;
            let removed_bb = entry_occ & !cur_occ;
            let added_bb = cur_occ & !entry_occ;
            let common_bb = cur_occ & entry_occ;

            let mut diff_count = (removed_bb | added_bb).count_ones() as usize;
            if diff_count <= 8 {
                let mut changed_common = 0u64;
                let mut temp = common_bb;
                while temp != 0 {
                    let s = temp.trailing_zeros() as usize;
                    temp &= temp - 1;
                    let cur_w = (pos.white >> s) & 1;
                    let entry_w = (entry.white >> s) & 1;
                    if cur_w != entry_w || pos.mapping[s] != entry.mapping[s] {
                        changed_common |= 1u64 << s;
                        diff_count += 1;
                        if diff_count > 8 {
                            break;
                        }
                    }
                }

                if diff_count <= 8 {
                    accs.halfka[pi] = entry.halfka;
                    accs.psqt[pi] = entry.psqt;
                    let slot = &mut accs.halfka[pi];
                    let psqt = &mut accs.psqt[pi];

                    let mut del_bb = removed_bb | changed_common;
                    while del_bb != 0 {
                        let s = del_bb.trailing_zeros() as usize;
                        del_bb &= del_bb - 1;
                        let is_w = (entry.white >> s) & 1 == 1;
                        let side = if is_w { Side::White } else { Side::Black };
                        let pt = entry.mapping[s] as usize;
                        if pt <= 5 {
                            let piece = Piece::ALL[pt];
                            if let Some(f) = halfka_index(perspective, side, piece, s, ksq) {
                                scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                            }
                        }
                    }

                    let mut add_bb = added_bb | changed_common;
                    while add_bb != 0 {
                        let s = add_bb.trailing_zeros() as usize;
                        add_bb &= add_bb - 1;
                        let is_w = (pos.white >> s) & 1 == 1;
                        let side = if is_w { Side::White } else { Side::Black };
                        let pt = pos.mapping[s] as usize;
                        if pt <= 5 {
                            let piece = Piece::ALL[pt];
                            if let Some(f) = halfka_index(perspective, side, piece, s, ksq) {
                                scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, 1);
                            }
                        }
                    }

                    entry.halfka = accs.halfka[pi];
                    entry.psqt = accs.psqt[pi];
                    entry.occupied = cur_occ;
                    entry.white = pos.white;
                    entry.mapping = pos.mapping;
                    return;
                }
            }
        }

        refresh_perspective(pos, nets, perspective, accs);
        entry.halfka = accs.halfka[pi];
        entry.psqt = accs.psqt[pi];
        entry.occupied = pos.occupied();
        entry.white = pos.white;
        entry.mapping = pos.mapping;
        entry.generation = current_generation;
        entry.valid = true;
    });
}

fn kings_present(pos: &SfnnPosition) -> bool {
    (pos.pieces[Piece::King as usize] & pos.white) != 0
        && (pos.pieces[Piece::King as usize] & pos.black) != 0
}

pub fn refresh_all(pos: &SfnnPosition, accs: &mut Sfnn16Accs) {
    // Mid-parse boards may miss a king: leave the entry stale so it
    // refreshes once the position is complete.
    if !kings_present(pos) {
        return;
    }
    let nets = active_loaded_net();
    if let Some(nets) = nets {
        refresh_perspective(pos, nets, Side::White, accs);
        refresh_perspective(pos, nets, Side::Black, accs);
        accs.threat_psqt = [[0i32; N_BUCKETS]; 2];
        accs.generation = current_gen();
    } else {
        accs.generation = current_gen();
    }
}

pub fn ensure_fresh(pos: &SfnnPosition, accs: &mut Sfnn16Accs) {
    if accs.generation != current_gen() {
        refresh_all(pos, accs);
    }
}

pub fn apply_queued(
    pos: &SfnnPosition,
    accs: &mut Sfnn16Accs,
    adds: &[Option<SfnnEvent>],
    dels: &[Option<SfnnEvent>],
    king_moved: &[bool; 2],
) {
    let nets = active_loaded_net();
    if let Some(nets) = nets {
        for (pi, perspective) in [Side::White, Side::Black].iter().enumerate() {
            if king_moved[pi] {
                update_perspective_finny_or_refresh(pos, nets, *perspective, accs);
                continue;
            }
            let ksq = pos.king_square(*perspective);
            let slot = &mut accs.halfka[pi];
            let psqt = &mut accs.psqt[pi];
            for opt in dels {
                if let Some((sq_sf, side, piece)) = *opt
                    && let Some(f) = halfka_index(*perspective, side, piece, sq_sf, ksq)
                {
                    scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                }
            }
            for opt in adds {
                if let Some((sq_sf, side, piece)) = *opt
                    && let Some(f) = halfka_index(*perspective, side, piece, sq_sf, ksq)
                {
                    scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, 1);
                }
            }
        }
        accs.generation = current_gen();
    }
}

// ---------------------------------------------------------------------------
// Full evaluation.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_threat_w_i16_avx2(buf: &mut [i32; L1], w: &[i16]) {
    unsafe {
        use std::arch::x86_64::*;
        let b_ptr = buf.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m128i;
        let chunks = buf.len() / 8;
        for i in 0..chunks {
            let w_128 = _mm_loadu_si128(w_ptr.add(i));
            let w_256 = _mm256_cvtepi16_epi32(w_128);
            let b = _mm256_loadu_si256(b_ptr.add(i));
            _mm256_storeu_si256(b_ptr.add(i), _mm256_add_epi32(b, w_256));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_transform_rudi_avx2(
    base_acc: &[i16],
    threat_buf: &[i32; L1],
    dst: &mut [u8],
    half: usize,
    use_threats: bool,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(255);
        let mult257 = _mm256_set1_epi32(257);
        let base_ptr = base_acc.as_ptr();
        let threat_ptr = threat_buf.as_ptr() as *const __m256i;

        let n_chunks = half / 16;
        for chunk in 0..n_chunks {
            let j = chunk * 16;
            let b0_16 = _mm256_loadu_si256(base_ptr.add(j) as *const __m256i);
            let mut s0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_16));
            let mut s0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_16, 1));

            let b1_16 = _mm256_loadu_si256(base_ptr.add(j + half) as *const __m256i);
            let mut s1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_16));
            let mut s1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_16, 1));

            if use_threats {
                let t0_lo = _mm256_loadu_si256(threat_ptr.add(chunk * 2));
                let t0_hi = _mm256_loadu_si256(threat_ptr.add(chunk * 2 + 1));
                let t1_lo = _mm256_loadu_si256(threat_ptr.add((half / 8) + chunk * 2));
                let t1_hi = _mm256_loadu_si256(threat_ptr.add((half / 8) + chunk * 2 + 1));

                s0_lo = _mm256_add_epi32(s0_lo, t0_lo);
                s0_hi = _mm256_add_epi32(s0_hi, t0_hi);
                s1_lo = _mm256_add_epi32(s1_lo, t1_lo);
                s1_hi = _mm256_add_epi32(s1_hi, t1_hi);
            }

            let c0_lo = _mm256_min_epi32(_mm256_max_epi32(s0_lo, zero), max_val);
            let c0_hi = _mm256_min_epi32(_mm256_max_epi32(s0_hi, zero), max_val);
            let c1_lo = _mm256_min_epi32(_mm256_max_epi32(s1_lo, zero), max_val);
            let c1_hi = _mm256_min_epi32(_mm256_max_epi32(s1_hi, zero), max_val);

            let prod_lo = _mm256_mullo_epi32(c0_lo, c1_lo);
            let prod_hi = _mm256_mullo_epi32(c0_hi, c1_hi);

            let div_lo = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(prod_lo, mult257), mult257),
                16,
            );
            let div_hi = _mm256_srli_epi32(
                _mm256_add_epi32(_mm256_mullo_epi32(prod_hi, mult257), mult257),
                16,
            );

            let d_lo_0 = _mm256_castsi256_si128(div_lo);
            let d_lo_1 = _mm256_extracti128_si256(div_lo, 1);
            let d_hi_0 = _mm256_castsi256_si128(div_hi);
            let d_hi_1 = _mm256_extracti128_si256(div_hi, 1);

            let p0 = _mm_packs_epi32(d_lo_0, d_lo_1);
            let p1 = _mm_packs_epi32(d_hi_0, d_hi_1);

            let bytes = _mm_packus_epi16(p0, p1);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, bytes);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn pairwise_transform_threats_tiled_avx2(
    base_acc: &[i16],
    threat_w: &[i8],
    pair_w: &[i8],
    threat_features: &[usize],
    pair_features: &[usize],
    dst: &mut [u8],
    half: usize,
    clip_max: i32,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi16(clip_max as i16);
        let base_ptr = base_acc.as_ptr();
        let threat_ptr = threat_w.as_ptr();
        let pair_ptr = pair_w.as_ptr();

        for chunk in 0..(half / 16) {
            let offset0 = chunk * 16;
            let offset1 = offset0 + half;

            let mut sum0 = _mm256_setzero_si256();
            let mut sum1 = _mm256_setzero_si256();

            for &f in threat_features {
                let t = f - PSQ_DIMS;
                let w_base = threat_ptr.add(t * 1024);

                let raw0 = _mm_loadu_si128(w_base.add(offset0) as *const __m128i);
                let w0 = _mm256_cvtepi8_epi16(raw0);
                sum0 = _mm256_add_epi16(sum0, w0);

                let raw1 = _mm_loadu_si128(w_base.add(offset1) as *const __m128i);
                let w1 = _mm256_cvtepi8_epi16(raw1);
                sum1 = _mm256_add_epi16(sum1, w1);
            }

            if !pair_w.is_empty() {
                for &f in pair_features {
                    let t = f - PSQ_DIMS - THREAT_DIMS;
                    let w_base = pair_ptr.add(t * 1024);

                    let raw0 = _mm_loadu_si128(w_base.add(offset0) as *const __m128i);
                    let w0 = _mm256_cvtepi8_epi16(raw0);
                    sum0 = _mm256_add_epi16(sum0, w0);

                    let raw1 = _mm_loadu_si128(w_base.add(offset1) as *const __m128i);
                    let w1 = _mm256_cvtepi8_epi16(raw1);
                    sum1 = _mm256_add_epi16(sum1, w1);
                }
            }

            let b0 = _mm256_loadu_si256(base_ptr.add(offset0) as *const __m256i);
            let b1 = _mm256_loadu_si256(base_ptr.add(offset1) as *const __m256i);

            let s0 = _mm256_adds_epi16(b0, sum0);
            let s1 = _mm256_adds_epi16(b1, sum1);

            let s0_clamp = _mm256_min_epi16(_mm256_max_epi16(s0, zero), max_val);
            let s0_shl = _mm256_slli_epi16(s0_clamp, 7);
            let s1_min = _mm256_min_epi16(s1, max_val);

            let mul = _mm256_mulhi_epi16(s0_shl, s1_min);

            let lo = _mm256_castsi256_si128(mul);
            let hi = _mm256_extracti128_si256(mul, 1);
            let bytes = _mm_packus_epi16(lo, hi);
            _mm_storeu_si128(dst.as_mut_ptr().add(offset0) as *mut __m128i, bytes);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_psqt_avx2(
    dst: &mut [i32; N_BUCKETS],
    threat_psqt_w: &[i32],
    pair_psqt_w: &[i32],
    threat_features: &[usize],
    pair_features: &[usize],
) {
    unsafe {
        use std::arch::x86_64::*;
        let mut acc = _mm256_setzero_si256();
        let t_ptr = threat_psqt_w.as_ptr();
        for &f in threat_features {
            let t = f - PSQ_DIMS;
            let p = _mm256_loadu_si256(t_ptr.add(t * N_BUCKETS) as *const __m256i);
            acc = _mm256_add_epi32(acc, p);
        }
        if !pair_psqt_w.is_empty() {
            let p_ptr = pair_psqt_w.as_ptr();
            for &f in pair_features {
                let t = f - PSQ_DIMS - THREAT_DIMS;
                let p = _mm256_loadu_si256(p_ptr.add(t * N_BUCKETS) as *const __m256i);
                acc = _mm256_add_epi32(acc, p);
            }
        }
        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, acc);
    }
}

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_transform_threats_avx2(
    base_acc: &[i16],
    threat_buf: &[i32; L1],
    dst: &mut [u8],
    half: usize,
    clip_max: i32,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(clip_max);
        let base_ptr = base_acc.as_ptr();
        let threat_ptr = threat_buf.as_ptr() as *const __m256i;

        for chunk in 0..(half / 16) {
            let j = chunk * 16;
            let b0_256 = _mm256_loadu_si256(base_ptr.add(j) as *const __m256i);
            let b0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_256));
            let b0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_256, 1));

            let b1_256 = _mm256_loadu_si256(base_ptr.add(j + half) as *const __m256i);
            let b1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_256));
            let b1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_256, 1));

            let t0_lo = _mm256_loadu_si256(threat_ptr.add(chunk * 2));
            let t0_hi = _mm256_loadu_si256(threat_ptr.add(chunk * 2 + 1));
            let t1_lo = _mm256_loadu_si256(threat_ptr.add(64 + chunk * 2));
            let t1_hi = _mm256_loadu_si256(threat_ptr.add(64 + chunk * 2 + 1));

            let s0_lo = _mm256_add_epi32(b0_lo, t0_lo);
            let s0_hi = _mm256_add_epi32(b0_hi, t0_hi);
            let s1_lo = _mm256_add_epi32(b1_lo, t1_lo);
            let s1_hi = _mm256_add_epi32(b1_hi, t1_hi);

            let c0_lo = _mm256_min_epi32(_mm256_max_epi32(s0_lo, zero), max_val);
            let c0_hi = _mm256_min_epi32(_mm256_max_epi32(s0_hi, zero), max_val);
            let c1_lo = _mm256_min_epi32(_mm256_max_epi32(s1_lo, zero), max_val);
            let c1_hi = _mm256_min_epi32(_mm256_max_epi32(s1_hi, zero), max_val);

            let prod_lo = _mm256_mullo_epi32(c0_lo, c1_lo);
            let prod_hi = _mm256_mullo_epi32(c0_hi, c1_hi);

            let div_lo = _mm256_srli_epi32(prod_lo, 9);
            let div_hi = _mm256_srli_epi32(prod_hi, 9);

            let d_lo_0 = _mm256_castsi256_si128(div_lo);
            let d_lo_1 = _mm256_extracti128_si256(div_lo, 1);
            let d_hi_0 = _mm256_castsi256_si128(div_hi);
            let d_hi_1 = _mm256_extracti128_si256(div_hi, 1);

            let p0 = _mm_packs_epi32(d_lo_0, d_lo_1);
            let p1 = _mm_packs_epi32(d_hi_0, d_hi_1);

            let bytes = _mm_packus_epi16(p0, p1);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, bytes);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pairwise_transform_no_threats_avx2(
    base_acc: &[i16],
    dst: &mut [u8],
    half: usize,
    clip_max: i32,
) {
    unsafe {
        use std::arch::x86_64::*;
        let zero = _mm256_setzero_si256();
        let max_val = _mm256_set1_epi32(clip_max);
        let base_ptr = base_acc.as_ptr();

        for chunk in 0..(half / 16) {
            let j = chunk * 16;
            let b0_256 = _mm256_loadu_si256(base_ptr.add(j) as *const __m256i);
            let b0_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_256));
            let b0_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_256, 1));

            let b1_256 = _mm256_loadu_si256(base_ptr.add(j + half) as *const __m256i);
            let b1_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_256));
            let b1_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_256, 1));

            let c0_lo = _mm256_min_epi32(_mm256_max_epi32(b0_lo, zero), max_val);
            let c0_hi = _mm256_min_epi32(_mm256_max_epi32(b0_hi, zero), max_val);
            let c1_lo = _mm256_min_epi32(_mm256_max_epi32(b1_lo, zero), max_val);
            let c1_hi = _mm256_min_epi32(_mm256_max_epi32(b1_hi, zero), max_val);

            let prod_lo = _mm256_mullo_epi32(c0_lo, c1_lo);
            let prod_hi = _mm256_mullo_epi32(c0_hi, c1_hi);

            let div_lo = _mm256_srli_epi32(prod_lo, 9);
            let div_hi = _mm256_srli_epi32(prod_hi, 9);

            let d_lo_0 = _mm256_castsi256_si128(div_lo);
            let d_lo_1 = _mm256_extracti128_si256(div_lo, 1);
            let d_hi_0 = _mm256_castsi256_si128(div_hi);
            let d_hi_1 = _mm256_extracti128_si256(div_hi, 1);

            let p0 = _mm_packs_epi32(d_lo_0, d_lo_1);
            let p1 = _mm_packs_epi32(d_hi_0, d_hi_1);

            let bytes = _mm_packus_epi16(p0, p1);
            _mm_storeu_si128(dst.as_mut_ptr().add(j) as *mut __m128i, bytes);
        }
    }
}

fn eval_with_net(
    pos: &SfnnPosition,
    net: &Sfnn16Net,
    halfka_accs: [&[i16]; 2],
    psqt_accs: [&[i32; N_BUCKETS]; 2],
    stm: Side,
    bucket: usize,
) -> (i32, i32) {
    let l1 = net.l1;
    let half = l1 / 2;
    let clip_max: i32 = 255;
    let perspectives = [stm, stm.other()];

    let mut threat_lists = [[0usize; MAX_THREAT_ACTIVE]; 2];
    let mut threat_lens = [0usize; 2];
    let mut pair_lists = [[0usize; MAX_PAIR_ACTIVE]; 2];
    let mut pair_lens = [0usize; 2];

    if net.use_threats {
        let mut raw_threats = [(0u8, 0u8, 0u8, 0u8); 128];
        let mut n_raw_threats = 0usize;
        for_each_threat(pos, |attacker, from, to, attacked| {
            if n_raw_threats < 128 {
                raw_threats[n_raw_threats] = (attacker as u8, from as u8, to as u8, attacked as u8);
                n_raw_threats += 1;
            }
        });

        for (slot, &persp) in perspectives.iter().enumerate() {
            let ksq = pos.king_square(persp);
            let mut len = 0;
            for &(attacker, from, to, attacked) in &raw_threats[..n_raw_threats] {
                if let Some(idx) = threat_make_index(
                    persp,
                    attacker as usize,
                    from as usize,
                    to as usize,
                    attacked as usize,
                    ksq,
                ) && len < MAX_THREAT_ACTIVE
                {
                    threat_lists[slot][len] = PSQ_DIMS + idx;
                    len += 1;
                }
            }
            threat_lens[slot] = len;
        }

        if !net.transformer.pair_w.is_empty() || !net.transformer.pair_w_i16.is_empty() {
            let mut raw_pairs = [(Side::White, 0u8, 0u8, Side::White); 64];
            let mut n_raw_pairs = 0usize;
            for_each_pair(pos, |color, from, to, paired| {
                if n_raw_pairs < 64 {
                    raw_pairs[n_raw_pairs] = (color, from as u8, to as u8, paired);
                    n_raw_pairs += 1;
                }
            });

            for (slot, &persp) in perspectives.iter().enumerate() {
                let ksq = pos.king_square(persp);
                let mut len = 0;
                for &(color, from, to, paired) in &raw_pairs[..n_raw_pairs] {
                    if len < MAX_PAIR_ACTIVE {
                        pair_lists[slot][len] =
                            pair_make_index(persp, color, from as usize, to as usize, paired, ksq);
                        len += 1;
                    }
                }
                pair_lens[slot] = len;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2() && l1 == 1024 && !net.is_rudi;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let mut feats = [0u8; L1];
    let mut per_psqt = [[0i32; N_BUCKETS]; 2];

    if use_avx2 && half == 512 {
        for (slot, &persp) in perspectives.iter().enumerate() {
            let pi = persp as usize;
            let base_acc = halfka_accs[pi];
            let dst = &mut feats[slot * half..(slot + 1) * half];

            if net.use_threats {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    pairwise_transform_threats_tiled_avx2(
                        base_acc,
                        &net.transformer.threat_w,
                        &net.transformer.pair_w,
                        &threat_lists[slot][..threat_lens[slot]],
                        &pair_lists[slot][..pair_lens[slot]],
                        dst,
                        half,
                        clip_max,
                    );
                    accumulate_psqt_avx2(
                        &mut per_psqt[slot],
                        &net.transformer.threat_psqt_w,
                        &net.transformer.pair_psqt_w,
                        &threat_lists[slot][..threat_lens[slot]],
                        &pair_lists[slot][..pair_lens[slot]],
                    );
                }
            } else {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    pairwise_transform_no_threats_avx2(base_acc, dst, half, clip_max);
                }
            }
        }
    } else {
        for (slot, &persp) in perspectives.iter().enumerate() {
            let pi = persp as usize;
            let base_acc = halfka_accs[pi];
            let mut threat_buf = [0i32; L1];
            if net.use_threats {
                if net.is_rudi {
                    #[cfg(target_arch = "x86_64")]
                    let use_rudi_avx2 = has_avx2() && l1 == L1;
                    #[cfg(not(target_arch = "x86_64"))]
                    let use_rudi_avx2 = false;

                    for &f in &threat_lists[slot][..threat_lens[slot]] {
                        let t = f - PSQ_DIMS;
                        let base = t * l1;
                        let w_slice = &net.transformer.threat_w_i16[base..base + l1];
                        if use_rudi_avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                add_threat_w_i16_avx2(&mut threat_buf, w_slice);
                            }
                        } else {
                            for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                                *acc += i32::from(w);
                            }
                        }
                    }
                    for &f in &pair_lists[slot][..pair_lens[slot]] {
                        let base = (f - PSQ_DIMS - THREAT_DIMS) * l1;
                        let w_slice = &net.transformer.pair_w_i16[base..base + l1];
                        if use_rudi_avx2 {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                add_threat_w_i16_avx2(&mut threat_buf, w_slice);
                            }
                        } else {
                            for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                                *acc += i32::from(w);
                            }
                        }
                    }
                } else {
                    for &f in &threat_lists[slot][..threat_lens[slot]] {
                        let t = f - PSQ_DIMS;
                        let base = t * l1;
                        let w_slice = &net.transformer.threat_w[base..base + l1];
                        for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                            *acc += i32::from(w);
                        }
                        let base_psqt = t * N_BUCKETS;
                        let p_slice =
                            &net.transformer.threat_psqt_w[base_psqt..base_psqt + N_BUCKETS];
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    }
                    for &f in &pair_lists[slot][..pair_lens[slot]] {
                        let t = f - PSQ_DIMS - THREAT_DIMS;
                        let base = t * l1;
                        let w_slice = &net.transformer.pair_w[base..base + l1];
                        for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                            *acc += i32::from(w);
                        }
                        let base_psqt = t * N_BUCKETS;
                        let p_slice =
                            &net.transformer.pair_psqt_w[base_psqt..base_psqt + N_BUCKETS];
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    }
                }
            }
            let dst = &mut feats[slot * half..(slot + 1) * half];
            if net.is_rudi {
                #[cfg(target_arch = "x86_64")]
                let use_rudi_avx2 = has_avx2() && half.is_multiple_of(16);
                #[cfg(not(target_arch = "x86_64"))]
                let use_rudi_avx2 = false;

                if use_rudi_avx2 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        pairwise_transform_rudi_avx2(
                            base_acc,
                            &threat_buf,
                            dst,
                            half,
                            net.use_threats,
                        );
                    }
                } else {
                    for j in 0..half {
                        let mut s0 = i32::from(base_acc[j]);
                        let mut s1 = i32::from(base_acc[j + half]);
                        if net.use_threats {
                            s0 += threat_buf[j];
                            s1 += threat_buf[j + half];
                        }
                        let c0 = s0.clamp(0, clip_max);
                        let c1 = s1.clamp(0, clip_max);
                        dst[j] = ((c0 * c1) / 255) as u8;
                    }
                }
            } else {
                for j in 0..half {
                    let mut s0 = i32::from(base_acc[j]);
                    let mut s1 = i32::from(base_acc[j + half]);
                    if net.use_threats {
                        s0 += threat_buf[j];
                        s1 += threat_buf[j + half];
                    }
                    let c0 = s0.clamp(0, clip_max);
                    let c1 = s1.clamp(0, clip_max);
                    dst[j] = ((c0 * c1) / 512) as u8;
                }
            }
        }
    }
    let mut psqt = psqt_accs[stm as usize][bucket] - psqt_accs[stm.other() as usize][bucket];
    if net.use_threats {
        psqt = div_trunc(psqt + per_psqt[0][bucket] - per_psqt[1][bucket], 2);
    } else {
        psqt = div_trunc(psqt, 2);
    }
    let stack = &net.stacks[bucket];
    let positional = propagate(stack, l1, &feats[..l1]);
    if net.is_rudi {
        (0, div_trunc(positional, OUTPUT_SCALE))
    } else {
        (
            div_trunc(psqt, OUTPUT_SCALE),
            div_trunc(positional, OUTPUT_SCALE),
        )
    }
}

pub struct SfnnEval {
    pub psqt: i32,
    pub positional: i32,
    pub combined: i32,
    pub used_small: bool,
}

pub fn evaluate_nets(pos: &SfnnPosition, accs: &mut Sfnn16Accs, stm: Side) -> Option<SfnnEval> {
    // Kingless mid-parse boards would yield trailing_zeros() == 64 below
    // (king-square-indexed tables); bail out instead of indexing OOB.
    if !kings_present(pos) {
        return None;
    }
    let nets = active_loaded_net()?;
    ensure_fresh(pos, accs);

    let bucket = material_bucket(pos.piece_count());
    let refs: [&[i16]; 2] = [&accs.halfka[0], &accs.halfka[1]];
    let psqt_refs: [&[i32; N_BUCKETS]; 2] = [&accs.psqt[0], &accs.psqt[1]];

    let (psqt, positional) = eval_with_net(pos, &nets.net, refs, psqt_refs, stm, bucket);
    let combined = if nets.net.is_rudi {
        positional
    } else {
        psqt + positional
    };

    Some(SfnnEval {
        psqt,
        positional,
        combined,
        used_small: false,
    })
}

pub fn ensure_sfnn16_fresh(board: &mut BoardState, pos: &SfnnPosition) {
    let idx = board.history.index;
    let current_generation = current_gen();
    let nets = match active_loaded_net() {
        Some(n) => n,
        None => return,
    };

    if !kings_present(pos) {
        return;
    }

    for (p, &perspective) in [Side::White, Side::Black].iter().enumerate() {
        if board.history.sfnn16_computed[idx][p]
            && board.history.sfnn16[idx].generation == current_generation
        {
            continue;
        }

        let mut ancestor = None;
        for anc in (0..idx).rev() {
            if board.history.sfnn16_pending[anc + 1].king_moved[p]
                || board.history.sfnn16_pending[anc + 1].overflowed
            {
                break;
            }
            if board.history.sfnn16_computed[anc][p]
                && board.history.sfnn16[anc].generation == current_generation
            {
                ancestor = Some(anc);
                break;
            }
        }

        if let Some(anc) = ancestor {
            let ksq = pos.king_square(perspective);
            for k in (anc + 1)..=idx {
                if !board.history.sfnn16_computed[k][p]
                    || board.history.sfnn16[k].generation != current_generation
                {
                    board.history.sfnn16[k].halfka[p] = board.history.sfnn16[k - 1].halfka[p];
                    board.history.sfnn16[k].psqt[p] = board.history.sfnn16[k - 1].psqt[p];

                    let pending = board.history.sfnn16_pending[k];
                    let slot = &mut board.history.sfnn16[k].halfka[p];
                    let psqt = &mut board.history.sfnn16[k].psqt[p];

                    for &ev in &pending.dels[..pending.n_dels] {
                        if let Some((sq_sf, side, piece)) = ev
                            && let Some(f) = halfka_index(perspective, side, piece, sq_sf, ksq)
                        {
                            scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                        }
                    }

                    for &ev in &pending.adds[..pending.n_adds] {
                        if let Some((sq_sf, side, piece)) = ev
                            && let Some(f) = halfka_index(perspective, side, piece, sq_sf, ksq)
                        {
                            scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, 1);
                        }
                    }

                    board.history.sfnn16_computed[k][p] = true;
                    board.history.sfnn16[k].generation = current_generation;
                }
            }
        } else {
            let accs = &mut board.history.sfnn16[idx];
            update_perspective_finny_or_refresh(pos, nets, perspective, accs);
            board.history.sfnn16_computed[idx][p] = true;
            board.history.sfnn16[idx].generation = current_generation;
        }
    }
}

pub fn evaluate_board(board: &mut BoardState) -> Option<i16> {
    if !maintenance_active() {
        return None;
    }
    let pos = SfnnPosition::from_board(board);
    ensure_sfnn16_fresh(board, &pos);
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];
    // Internal units, same as evaluate_board_detailed: the single caller
    // (nnue::evaluate_with_optimism fallback) applies gate/material/damping
    // itself. No *100/256 rescale here (that scale is only for UCI display).
    evaluate_nets(&pos, accs, board.side_to_move).map(|e| e.combined.clamp(-29000, 29000) as i16)
}

pub fn evaluate_board_detailed(board: &mut BoardState) -> Option<SfnnEval> {
    if !maintenance_active() {
        return None;
    }
    let pos = SfnnPosition::from_board(board);
    ensure_sfnn16_fresh(board, &pos);
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];
    evaluate_nets(&pos, accs, board.side_to_move)
}

// ---------------------------------------------------------------------------
// Board integration helpers (wired from board/nnue.rs).
// ---------------------------------------------------------------------------

/// Queued v10 feature event in SF numbering: (sq_sf, side, piece).
pub type SfnnEvent = (usize, Side, Piece);

#[derive(Clone, Copy, Debug, Default)]
pub struct SfnnPending {
    pub adds: [Option<SfnnEvent>; 8],
    pub dels: [Option<SfnnEvent>; 8],
    pub n_adds: usize,
    pub n_dels: usize,
    pub king_moved: [bool; 2],
    pub overflowed: bool,
}

impl SfnnPending {
    pub fn push_add(&mut self, ev: SfnnEvent) {
        if self.n_adds < self.adds.len() {
            self.adds[self.n_adds] = Some(ev);
            self.n_adds += 1;
        } else {
            self.overflowed = true;
        }
    }

    pub fn push_del(&mut self, ev: SfnnEvent) {
        if self.n_dels < self.dels.len() {
            self.dels[self.n_dels] = Some(ev);
            self.n_dels += 1;
        } else {
            self.overflowed = true;
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

pub fn note_add(pending: &mut SfnnPending, square: Square, side: Side, piece: Piece) {
    if piece == Piece::King {
        pending.king_moved[side as usize] = true;
    }
    pending.push_add((to_sf(square as usize), side, piece));
}

pub fn note_remove(pending: &mut SfnnPending, square: Square, side: Side, piece: Piece) {
    if piece == Piece::King {
        pending.king_moved[side as usize] = true;
    }
    pending.push_del((to_sf(square as usize), side, piece));
}

/// Flush queued events into the history entry (incremental HalfKA + PSQT).
/// Falls back to full refresh on overflow or king moves per perspective.
pub fn flush_pending(pos: &SfnnPosition, accs: &mut Sfnn16Accs, pending: &mut SfnnPending) {
    if !maintenance_active() {
        pending.clear();
        return;
    }
    if !kings_present(pos) {
        pending.clear();
        return;
    }
    if accs.generation != current_gen() {
        // Stale entry (new position or new nets): full refresh covers all
        // queued events, so drop them.
        refresh_all(pos, accs);
        pending.clear();
        return;
    }
    if pending.overflowed {
        refresh_all(pos, accs);
        pending.clear();
        return;
    }
    apply_queued(
        pos,
        accs,
        &pending.adds[..pending.n_adds],
        &pending.dels[..pending.n_dels],
        &pending.king_moved,
    );
    pending.clear();
}

// ---------------------------------------------------------------------------
// Trainer-facing helpers (feature enumeration in SF numbering).
// ---------------------------------------------------------------------------

/// Fill White/Black-perspective feature lists from SF-numbered bitboards
/// (trainer path). Threat indices are relative to PSQ_DIMS (caller adds the
/// offset when the full index space is needed).
pub fn trainer_features(
    pieces: &[u64; 6],
    white: u64,
    black: u64,
    mapping: &[u8; 64],
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let pos = SfnnPosition {
        pieces: *pieces,
        white,
        black,
        mapping: *mapping,
    };
    let mut w_h = Vec::new();
    let mut b_h = Vec::new();
    append_halfka(&pos, Side::White, &mut w_h);
    append_halfka(&pos, Side::Black, &mut b_h);
    let mut w_t = Vec::new();
    let mut b_t = Vec::new();
    append_threats(&pos, Side::White, &mut w_t);
    append_threats(&pos, Side::Black, &mut b_t);
    append_pairs(&pos, Side::White, &mut w_t);
    append_pairs(&pos, Side::Black, &mut b_t);
    for v in w_t.iter_mut().chain(b_t.iter_mut()) {
        *v -= PSQ_DIMS;
    }
    (w_h, b_h, w_t, b_t)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn regression_fc0_full_range_dot() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut arch = SfnnArch {
            fc0_bias: [0; FC0_OUT],
            fc0_w: vec![0; 32 * FC0_OUT],
            fc1_bias: [0; FC1_OUT],
            fc1_w: vec![0; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
            is_rudi: true,
        };
        arch.fc0_bias[0] = 3;
        arch.fc0_bias[1] = -5;
        arch.fc0_w[..2].fill(127);
        arch.fc0_w[32..34].fill(-128);
        let mut input = [0u8; 32];
        input[..2].fill(255);
        let mut actual = [0; FC0_OUT];
        unsafe { fc0_avx2(&arch, &input, &mut actual) };
        assert_eq!(actual[0], 3 + 255 * 127 + 255 * 127);
        assert_eq!(actual[1], -5 - 255 * 128 - 255 * 128);
        assert!(actual[2..].iter().all(|&value| value == 0));
    }

    #[test]
    fn regression_rudi_division_identity() {
        for product in 0..=255 * 255 {
            assert_eq!((product * 257 + 257) >> 16, product / 255);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn regression_pairwise_clipping_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let half = L1 / 2;
        let mut base = [0i16; L1];
        let mut threats = [0i32; L1];
        let mut actual = [0u8; L1 / 2];
        let values = [i16::MIN, -256, -1, 0, 1, 127, 254, 255, 256, i16::MAX];
        for a in 0..256 {
            for j in 0..half {
                base[j] = a;
                base[j + half] = (j % 256) as i16;
            }
            unsafe { pairwise_transform_rudi_avx2(&base, &threats, &mut actual, half, false) };
            for (j, &value) in actual.iter().enumerate() {
                assert_eq!(value, (i32::from(a) * (j % 256) as i32 / 255) as u8);
            }
        }
        for j in 0..L1 {
            base[j] = values[j % values.len()];
            threats[j] = ((j * 37) % 1024) as i32 - 512;
        }
        for with_threats in [false, true] {
            for rudi in [false, true] {
                unsafe {
                    if rudi {
                        pairwise_transform_rudi_avx2(
                            &base,
                            &threats,
                            &mut actual,
                            half,
                            with_threats,
                        );
                    } else if with_threats {
                        pairwise_transform_threats_avx2(&base, &threats, &mut actual, half, 255);
                    } else {
                        pairwise_transform_no_threats_avx2(&base, &mut actual, half, 255);
                    }
                }
                for j in 0..half {
                    let extra = |i| if with_threats { threats[i] } else { 0 };
                    let a = (i32::from(base[j]) + extra(j)).clamp(0, 255);
                    let b = (i32::from(base[j + half]) + extra(j + half)).clamp(0, 255);
                    assert_eq!(actual[j], (a * b / if rudi { 255 } else { 512 }) as u8);
                }
            }
        }
    }

    #[test]
    fn regression_rudi_converter_header() {
        let mut bytes = Vec::with_capacity(181_011_116);
        bytes.extend_from_slice(b"RUDI");
        bytes.extend_from_slice(&181_011_108u32.to_le_bytes());
        bytes.resize(181_011_116, 0);
        let net = Sfnn16Net::load_rudi(&bytes).unwrap();
        assert!(net.use_threats);
        assert_eq!(net.transformer.threat_w_i16.len(), THREAT_DIMS * L1);
        assert_eq!(net.transformer.pair_w_i16.len(), PAIR_DIMS * L1);
        bytes.push(0);
        assert!(Sfnn16Net::load_rudi(&bytes).is_err());
    }

    #[test]
    fn regression_rudi_rejects_unknown_header() {
        let mut bytes = vec![0; 64 + 181_011_108];
        bytes[..4].copy_from_slice(b"RUDI");
        assert!(Sfnn16Net::load_rudi(&bytes).is_err());
    }

    #[test]
    fn regression_rudi_dynamic_features() {
        let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let mut arch = SfnnArch {
            fc0_bias: [0; FC0_OUT],
            fc0_w: vec![0; 2 * FC0_OUT],
            fc1_bias: [0; FC1_OUT],
            fc1_w: vec![0; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
            is_rudi: true,
        };
        arch.fc0_w[0] = 64;
        arch.fc1_w[FC0_OUT] = 64;
        arch.fc2_w[0] = 64;
        let mut net = Sfnn16Net {
            l1: 2,
            use_threats: true,
            is_rudi: true,
            transformer: SfnnTransformer {
                threat_w_i16: vec![0; THREAT_DIMS * 2],
                pair_w_i16: vec![0; PAIR_DIMS * 2],
                ..Default::default()
            },
            stacks: vec![arch],
        };
        let mut threats = Vec::new();
        let mut pairs = Vec::new();
        append_threats(&pos, Side::White, &mut threats);
        append_pairs(&pos, Side::White, &mut pairs);
        assert!(!threats.is_empty());
        assert!(!pairs.is_empty());
        let base = [0, 255];
        let psqt = [0; N_BUCKETS];
        let eval =
            |net: &Sfnn16Net| eval_with_net(&pos, net, [&base; 2], [&psqt; 2], Side::White, 0);
        let baseline = eval(&net);
        let t = (threats[0] - PSQ_DIMS) * 2;
        net.transformer.threat_w_i16[t] = 255;
        assert_ne!(eval(&net), baseline);
        net.transformer.threat_w_i16[t] = 0;
        let p = (pairs[0] - PSQ_DIMS - THREAT_DIMS) * 2;
        net.transformer.pair_w_i16[p] = 255;
        assert_ne!(eval(&net), baseline);
        net.use_threats = false;
        assert_eq!(eval(&net), baseline);
    }

    #[test]
    fn test_eval_breakdown() {
        let mut loaded_any = false;
        for net_path in ["v16/nn-1a298aa575a0.nnue", "v16/whale.nnue"] {
            println!("=== Testing net: {} ===", net_path);
            if !std::path::Path::new(net_path).exists() {
                println!("Missing {net_path}, skipping (network files are git-ignored fixtures)");
                continue;
            }
            if let Err(e) = load_net(net_path) {
                println!("Failed to load {net_path}: {e}");
                continue;
            }
            loaded_any = true;
            let is_official = net_path.contains("nn-1a298aa575a0");
            let fens = [
                (
                    "startpos",
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                ),
                (
                    "1. e4 e5 2. Ke2 (B)",
                    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2",
                ),
                (
                    "1. e4 e5 2. Nf3 (B)",
                    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
                ),
                (
                    "1. e4 (B)",
                    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                ),
            ];

            for (name, fen) in fens {
                let board = BoardState::parse_fen(fen);
                let pos = SfnnPosition::from_board(&board);
                let mut accs = Sfnn16Accs::empty();
                refresh_all(&pos, &mut accs);
                if let Some(eval) = evaluate_nets(&pos, &mut accs, board.side_to_move) {
                    // Sanity anchor: the official net evaluates startpos near zero.
                    if is_official && name == "startpos" {
                        assert_eq!(eval.combined, -1);
                    }
                    let cp = (eval.combined as i64 * 100) / 256;
                    println!(
                        "{:<22} | STM: {:?} | psqt: {:>6} | pos: {:>6} | comb: {:>6} | cp: {:>5}",
                        name, board.side_to_move, eval.psqt, eval.positional, eval.combined, cp
                    );
                }
            }
        }
        if !loaded_any {
            println!("No network fixtures present; skipping breakdown assertions");
            return;
        }
        let nets = loaded_nets().expect("network should be active after load");
        let stack = &nets.net.stacks[7];
        println!(
            "fc0_bias[30] = {}, fc0_bias[31] = {}",
            stack.fc0_bias[30], stack.fc0_bias[31]
        );
        let w30 = &stack.fc0_w[30 * 1024..(30 + 1) * 1024];
        let w31 = &stack.fc0_w[31 * 1024..(31 + 1) * 1024];
        let sum_w30_us: i64 = w30[..512].iter().map(|&x| x as i64).sum();
        let sum_w30_them: i64 = w30[512..].iter().map(|&x| x as i64).sum();
        let sum_w31_us: i64 = w31[..512].iter().map(|&x| x as i64).sum();
        let sum_w31_them: i64 = w31[512..].iter().map(|&x| x as i64).sum();
        println!("w30 us sum = {}, them sum = {}", sum_w30_us, sum_w30_them);
        println!("w31 us sum = {}, them sum = {}", sum_w31_us, sum_w31_them);
    }

    #[test]
    fn bench_eval_speed() {
        if !std::path::Path::new("v16/nn-1a298aa575a0.nnue").exists() {
            return;
        }
        load_net("v16/nn-1a298aa575a0.nnue").unwrap();
        let board =
            BoardState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let mut accs = Sfnn16Accs::empty();
        refresh_all(&pos, &mut accs);
        let start = std::time::Instant::now();
        let iters = 10_000;
        for _ in 0..iters {
            let _ = evaluate_nets(&pos, &mut accs, board.side_to_move);
        }
        let el = start.elapsed();
        println!(
            "BASELINE: {} evals in {:?} ({:.2} us/eval, {:.0} NPS)",
            iters,
            el,
            (el.as_micros() as f64) / (iters as f64),
            (iters as f64) / el.as_secs_f64()
        );
    }

    #[test]
    fn square_helpers_use_sf_numbering() {
        // Whale E2 = 52 flips to SF E2 = 12 (A1 = 0).
        assert_eq!(to_sf(52), 12);
        assert_eq!(to_sf(to_sf(36)), 36);
        // Kings on a-d files mirror, e-h files do not.
        assert_eq!(halfka_orient(0), 7);
        assert_eq!(halfka_orient(4), 0);
        assert_eq!(sf_piece_code(Side::White, Piece::Pawn), 1);
        assert_eq!(sf_piece_code(Side::White, Piece::King), 6);
        assert_eq!(sf_piece_code(Side::Black, Piece::Pawn), 9);
        assert_eq!(sf_piece_code(Side::Black, Piece::King), 14);
    }

    #[test]
    fn material_bucket_boundaries() {
        assert_eq!(material_bucket(0), 0);
        assert_eq!(material_bucket(1), 0);
        assert_eq!(material_bucket(4), 0);
        assert_eq!(material_bucket(5), 1);
        assert_eq!(material_bucket(32), 7);
        assert_eq!(material_bucket(1000), N_BUCKETS - 1);
    }

    #[test]
    fn simple_eval_symmetry_and_material() {
        use crate::common::helpers::STARTING_FEN;

        let board = BoardState::parse_fen(STARTING_FEN);
        let pos = SfnnPosition::from_board(&board);
        assert_eq!(pos.piece_count(), 32);
        assert_eq!(pos.occupied(), pos.white | pos.black);
        assert_eq!(pos.king_square(Side::White), 4);
        assert_eq!(simple_eval(&pos, Side::White), 0);
        assert_eq!(simple_eval(&pos, Side::Black), 0);

        let mut up = BoardState::parse_fen(STARTING_FEN);
        up.add_piece(Square::E4, Side::White, Piece::Queen, false);
        let up_pos = SfnnPosition::from_board(&up);
        assert_eq!(simple_eval(&up_pos, Side::White), QUEEN_VALUE);
        assert_eq!(simple_eval(&up_pos, Side::Black), -QUEEN_VALUE);
    }

    #[test]
    fn hash_utils_are_deterministic() {
        assert_eq!(combine_hash(&[]), 0);
        assert_eq!(rotl1(1), 2);
        assert_eq!(rotl1(u32::MAX), u32::MAX);
        assert_ne!(affine_hash(32, 0), affine_hash(31, 0));
        assert_ne!(affine_hash(32, 0), affine_hash(32, 1));
        assert_ne!(relu_hash(0), relu_hash(1));
        assert_eq!(arch_hash(1024), arch_hash(1024));
        assert_ne!(transformer_hash(true, 1024), transformer_hash(false, 1024));
        assert_eq!(
            network_hash(true, 1024),
            transformer_hash(true, 1024) ^ arch_hash(1024)
        );
    }

    #[test]
    fn div_trunc_matches_rust_division() {
        assert_eq!(div_trunc(7, 2), 3);
        assert_eq!(div_trunc(-7, 2), -3);
        assert_eq!(div_trunc(7, -2), -3);
        assert_eq!(div_trunc(0, 5), 0);
    }

    #[test]
    fn read_le_primitives() {
        let mut pos = 0;
        assert_eq!(read_u32_le(&[1, 0, 0, 0, 9], &mut pos), Ok(1));
        assert_eq!(pos, 4);
        assert_eq!(read_u32_le(&[9], &mut pos), Err("truncated u32"));
        let mut neg = 0;
        assert_eq!(read_i32_le(&[0xFF, 0xFF, 0xFF, 0xFF], &mut neg), Ok(-1));

        // LEB128 section: magic + length + two i16 payloads (1, -1).
        let mut data = Vec::new();
        data.extend_from_slice(LEB128_MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&[0x01, 0x7F]);
        let mut spos = 0;
        assert_eq!(
            read_leb128_section(&data, &mut spos, 2, decode_leb128_i16),
            Ok(vec![1, -1])
        );
        assert!(read_leb128_section(&[], &mut 0, 1, decode_leb128_i16).is_err());
        assert!(read_leb128_section(&[0u8; 17], &mut 0, 1, decode_leb128_i16).is_err());
    }

    #[test]
    fn decode_leb128_values_and_errors() {
        let mut pos = 0;
        assert_eq!(decode_leb128_i64(&[0x05], &mut pos, 64), Ok(5));
        let mut pos = 0;
        assert_eq!(decode_leb128_i64(&[0xAC, 0x02], &mut pos, 64), Ok(300));
        let mut pos = 0;
        assert_eq!(decode_leb128_i64(&[0x7F], &mut pos, 64), Ok(-1));
        let mut pos = 0;
        assert_eq!(decode_leb128_i32(&[0xFF, 0x7F], &mut pos), Ok(-1));
        let mut pos = 0;
        assert!(decode_leb128_i64(&[0x80], &mut pos, 64).is_err());
        let mut pos = 0;
        assert!(decode_leb128_i16(&[0x80, 0x80, 0x80, 0x80], &mut pos).is_err());
    }

    #[test]
    fn read_rudi_primitives() {
        let mut off = 0;
        assert_eq!(read_rudi_i8(&[0x01, 0xFF], &mut off, 2), Ok(vec![1, -1]));
        assert_eq!(off, 2);
        assert!(read_rudi_i8(&[0x01], &mut 0, 2).is_err());

        let mut off = 0;
        assert_eq!(
            read_rudi_i16(&[0x01, 0x00, 0xFF, 0xFF], &mut off, 2),
            Ok(vec![1, -1])
        );
        assert!(read_rudi_i16(&[0x01], &mut 0, 1).is_err());

        let mut off = 0;
        assert_eq!(
            read_rudi_i32(&[0x01, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF], &mut off, 2),
            Ok(vec![1, -1])
        );
        assert!(read_rudi_i32(&[], &mut 0, 1).is_err());

        let mut off = 0;
        assert_eq!(read_rudi_i8(&[0x01], &mut off, 0), Ok(vec![]));
    }

    #[test]
    fn unscramble_table_is_permutation() {
        let mut inv = unscramble_table(2, 8);
        inv.sort_unstable();
        assert_eq!(inv, (0..16).collect::<Vec<_>>());
        assert_eq!(unscramble_table(1, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn sfnn_pending_queue_and_overflow() {
        let ev = (12usize, Side::White, Piece::Pawn);
        let mut pending = SfnnPending::default();
        for _ in 0..8 {
            pending.push_add(ev);
        }
        assert_eq!(pending.n_adds, 8);
        assert!(!pending.overflowed);
        pending.push_add(ev);
        assert!(pending.overflowed);

        let mut pending = SfnnPending::default();
        for _ in 0..8 {
            pending.push_del(ev);
        }
        pending.push_del(ev);
        assert!(pending.overflowed);
        pending.clear();
        assert_eq!(pending.n_adds, 0);
        assert_eq!(pending.n_dels, 0);
        assert!(!pending.overflowed);

        let mut pending = SfnnPending::default();
        note_add(&mut pending, Square::E2, Side::White, Piece::King);
        assert!(pending.king_moved[Side::White as usize]);
        assert_eq!(pending.adds[0], Some((12, Side::White, Piece::King)));
        note_remove(&mut pending, Square::E7, Side::Black, Piece::Pawn);
        assert!(!pending.king_moved[Side::Black as usize]);
        assert_eq!(
            pending.dels[0],
            Some((to_sf(Square::E7 as usize), Side::Black, Piece::Pawn))
        );
    }

    #[test]
    fn load_paths_reject_garbage() {
        assert!(Sfnn16Net::load_bytes(&[], true, L1).is_err());
        assert!(Sfnn16Net::load_bytes(&[0, 0, 0, 0], true, L1).is_err());
        assert!(Sfnn16Net::load_rudi(&[]).is_err());
        assert!(Sfnn16Net::load_file("definitely/missing.nnue", true, L1).is_err());
        assert!(Sfnn16Net::load_file("definitely/missing.nnue", false, L1).is_err());
        assert!(set_eval_file("Bogus", "x").is_err());
        assert!(set_eval_file("EvalFileSmall", "x").is_ok());
    }

    #[test]
    fn halfka_features_cover_all_pieces() {
        use crate::common::helpers::STARTING_FEN;

        assert_eq!(
            halfka_index(Side::White, Side::White, Piece::None, 0, 4),
            None
        );

        // King lives on the dedicated 640 plane.
        let king_idx = halfka_index(Side::White, Side::White, Piece::King, 4, 4).unwrap();
        assert!(king_idx >= 640);

        let board = BoardState::parse_fen(STARTING_FEN);
        let pos = SfnnPosition::from_board(&board);
        for perspective in [Side::White, Side::Black] {
            let mut out = Vec::new();
            append_halfka(&pos, perspective, &mut out);
            assert_eq!(out.len(), 32);
            let mut again = Vec::new();
            append_halfka(&pos, perspective, &mut again);
            assert_eq!(out, again);
        }

        // No king: no features.
        assert!({
            let mut out = Vec::new();
            append_halfka(
                &SfnnPosition {
                    pieces: [0; 6],
                    white: 0,
                    black: 0,
                    mapping: [6; 64],
                },
                Side::White,
                &mut out,
            );
            out.is_empty()
        });
    }

    #[test]
    fn threat_and_pair_collectors_agree() {
        // Kings only: no threats, no pairs, both collectors agree.
        let quiet =
            SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
        let mut counted = 0;
        for_each_pair(&quiet, |_, _, _, _| counted += 1);
        assert_eq!(counted, 0);

        let board = BoardState::parse_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        );
        let pos = SfnnPosition::from_board(&board);
        for perspective in [Side::White, Side::Black] {
            let mut listed = Vec::new();
            append_threats(&pos, perspective, &mut listed);
            let mut buf = [0usize; MAX_THREAT_ACTIVE];
            let n = collect_threats(&pos, perspective, &mut buf);
            assert_eq!(&listed, &buf[..n]);
            assert!(!listed.is_empty());
            for &idx in &listed {
                assert!((PSQ_DIMS..PSQ_DIMS + THREAT_DIMS).contains(&idx));
            }

            let mut plist = Vec::new();
            append_pairs(&pos, perspective, &mut plist);
            let mut pbuf = [0usize; MAX_PAIR_ACTIVE];
            let pn = collect_pairs(&pos, perspective, &mut pbuf);
            assert_eq!(&plist, &pbuf[..pn]);
            for &idx in &plist {
                assert!(
                    (PSQ_DIMS + THREAT_DIMS..PSQ_DIMS + THREAT_DIMS + PAIR_DIMS).contains(&idx)
                );
            }
        }

        // pair_index_for is the relative form of pair_make_index.
        let rel = pair_index_for(Side::White, Side::White, 12, 20, Side::White, 4);
        let abs = pair_make_index(Side::White, Side::White, 12, 20, Side::White, 4);
        assert_eq!(rel, abs - (PSQ_DIMS + THREAT_DIMS));
        assert!(rel < PAIR_DIMS);

        // Trainer packs (halfka_w, halfka_b, threats+pairs_w, threats+pairs_b).
        let (w_h, b_h, w_t, b_t) =
            trainer_features(&pos.pieces, pos.white, pos.black, &pos.mapping);
        assert_eq!(w_h.len(), 32);
        assert_eq!(b_h.len(), 32);
        assert!(!w_t.is_empty());
        assert!(!b_t.is_empty());
    }

    #[test]
    fn whale_family_models_load() {
        // The Whale model family lives in models/ (git-ignored: present in
        // local checkouts, absent on CI — hence the skip when missing).
        // Pure load_file: no global NETS mutation, safe under parallel tests.
        for name in ["whale_big", "whale_medium", "whale_small"] {
            let path = format!("models/{name}.nnue");
            if !std::path::Path::new(&path).exists() {
                println!("Missing {path}, skipping");
                continue;
            }
            let net = Sfnn16Net::load_file(&path, true, L1)
                .or_else(|_| Sfnn16Net::load_file(&path, false, L1))
                .unwrap_or_else(|e| panic!("{path} failed to load: {e}"));
            assert!(!net.stacks.is_empty(), "{path} has no stacks");
            println!(
                "{name}: l1={} rudi={} threats={} stacks={}",
                net.l1,
                net.is_rudi,
                net.use_threats,
                net.stacks.len()
            );
        }
    }

    #[test]
    fn test_finny_cache_consistency() {
        if !try_load_default() {
            return;
        }
        let nets = loaded_nets().unwrap();
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
            "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
        ];
        for fen in fens {
            let board = BoardState::parse_fen(fen);
            let pos = SfnnPosition::from_board(&board);
            for perspective in [Side::White, Side::Black] {
                let mut acc_direct = Sfnn16Accs::empty();
                refresh_perspective(&pos, &nets, perspective, &mut acc_direct);

                let mut acc_finny = Sfnn16Accs::empty();
                update_perspective_finny_or_refresh(&pos, &nets, perspective, &mut acc_finny);
                assert_eq!(
                    acc_finny.halfka[perspective as usize],
                    acc_direct.halfka[perspective as usize]
                );
                assert_eq!(
                    acc_finny.psqt[perspective as usize],
                    acc_direct.psqt[perspective as usize]
                );

                let mut acc_finny_cached = Sfnn16Accs::empty();
                update_perspective_finny_or_refresh(
                    &pos,
                    &nets,
                    perspective,
                    &mut acc_finny_cached,
                );
                assert_eq!(
                    acc_finny_cached.halfka[perspective as usize],
                    acc_direct.halfka[perspective as usize]
                );
                assert_eq!(
                    acc_finny_cached.psqt[perspective as usize],
                    acc_direct.psqt[perspective as usize]
                );
            }
        }
    }

    #[test]
    fn halfka_black_perspective_and_ownership() {
        // King on e1 (SF 4): white own pawn vs enemy pawn differ by 64.
        let own = halfka_index(Side::White, Side::White, Piece::Pawn, 12, 4).unwrap();
        let enemy = halfka_index(Side::White, Side::Black, Piece::Pawn, 12, 4).unwrap();
        assert_eq!(enemy.wrapping_sub(own), 64);
        assert!(own < PSQ_DIMS && enemy < PSQ_DIMS);
        // Black perspective flips the board (square 12 -> 52 region).
        let black = halfka_index(Side::Black, Side::Black, Piece::Pawn, 12, 4).unwrap();
        assert!(black < PSQ_DIMS);
        assert_ne!(own, black);
        // Left-file vs right-file kings mirror the file.
        let left = halfka_index(Side::White, Side::White, Piece::Knight, 0, 0).unwrap();
        let right = halfka_index(Side::White, Side::White, Piece::Knight, 0, 7).unwrap();
        assert_ne!(left, right);
        // All piece types (except None) produce an index.
        for &pt in &Piece::ALL {
            if pt == Piece::None {
                continue;
            }
            assert!(halfka_index(Side::White, Side::White, pt, 27, 4).is_some());
            assert!(halfka_index(Side::Black, Side::Black, pt, 27, 60).is_some());
        }
    }

    #[test]
    fn append_halfka_skips_empty_mapping() {
        let pos = SfnnPosition {
            pieces: [0; 6],
            white: 1,
            black: 0,
            mapping: [6; 64],
        };
        let mut out = Vec::new();
        append_halfka(&pos, Side::White, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn threat_orient_covers_all_sides() {
        assert_eq!(threat_orient(Side::White, 0), 0);
        assert_eq!(threat_orient(Side::White, 7), 7);
        assert_eq!(threat_orient(Side::Black, 0), 56);
        assert_eq!(threat_orient(Side::Black, 7), 63);
        assert_eq!(threat_orient(Side::Both, 0), 0);
        assert_eq!(threat_orient(Side::Both, 63), 0);
        assert_eq!(sf_piece_type(0), 1);
        assert_eq!(sf_piece_type(5), 6);
    }

    #[test]
    fn for_each_threat_covers_piece_types() {
        let cases = [
            // Pawn captures.
            "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
            // Knights attacking occupied squares.
            "4k3/8/8/3n4/8/2N5/8/4K3 w - - 0 1",
            // Bishops on a shared diagonal.
            "4k3/8/8/3b4/4B3/8/8/4K3 w - - 0 1",
            // Rooks on a shared file.
            "4k3/8/8/8/3R4/8/8/3rK3 w - - 0 1",
            // Queens on a shared diagonal.
            "4k3/8/8/3q4/4Q3/8/8/4K3 w - - 0 1",
            // King adjacent to a pawn.
            "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1",
        ];
        for fen in cases {
            let pos = SfnnPosition::from_board(&BoardState::parse_fen(fen));
            let mut count = 0;
            for_each_threat(&pos, |_, _, _, _| count += 1);
            assert!(count > 0, "no threat edges for {fen}");
        }
        // Kings only: no edges.
        let quiet =
            SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
        let mut count = 0;
        for_each_threat(&quiet, |_, _, _, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn threat_index_excluded_semi_and_perspective() {
        let ksq = 4;
        // Pawn vs pawn is excluded.
        assert_eq!(threat_index_for(Side::White, 1, 12, 19, 9, ksq), None);
        // Knight vs king is excluded.
        assert_eq!(threat_index_for(Side::White, 2, 10, 4, 6, ksq), None);
        // Knight vs knight (enemy) is semi: ordering decides.
        assert_eq!(threat_index_for(Side::White, 2, 10, 20, 10, ksq), None);
        let some = threat_index_for(Side::White, 2, 20, 10, 10, ksq);
        assert!(some.is_some());
        assert!(some.unwrap() < THREAT_DIMS);
        // Non-semi pair (knight vs queen) is valid in both orderings.
        assert!(threat_index_for(Side::White, 2, 10, 20, 13, ksq).is_some());
        assert!(threat_index_for(Side::White, 2, 20, 10, 13, ksq).is_some());
        // Black perspective also resolves (colors flipped internally).
        assert!(threat_index_for(Side::Black, 1, 12, 19, 9, 60).is_none());
        assert!(
            threat_index_for(Side::Black, 2, 20, 10, 10, 60).is_some()
                || threat_index_for(Side::Black, 2, 10, 20, 10, 60).is_some()
        );
    }

    #[test]
    fn pair_orient_and_enumeration_variants() {
        // All perspective/king-file/color combos stay in the pair window.
        for &persp in &[Side::White, Side::Black] {
            for &ksq in &[0usize, 7, 56, 63] {
                for &(c, pc) in &[
                    (Side::White, Side::White),
                    (Side::White, Side::Black),
                    (Side::Black, Side::Black),
                ] {
                    let abs = pair_make_index(persp, c, 12, 20, pc, ksq);
                    assert!(
                        (PSQ_DIMS + THREAT_DIMS..PSQ_DIMS + THREAT_DIMS + PAIR_DIMS).contains(&abs)
                    );
                    let rel = pair_index_for(persp, c, 12, 20, pc, ksq);
                    assert_eq!(rel, abs - (PSQ_DIMS + THREAT_DIMS));
                }
            }
        }
        // Neighboring white pawns pair; distant pawns do not.
        let near =
            SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/2P5/3P4/4K3 w - - 0 1"));
        let mut n = 0;
        for_each_pair(&near, |_, _, _, _| n += 1);
        assert!(n > 0);
        let far =
            SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/P6P/4K3 w - - 0 1"));
        let mut m = 0;
        for_each_pair(&far, |_, _, _, _| m += 1);
        assert_eq!(m, 0);
        // Mixed colors on neighboring files pair.
        let mixed =
            SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/2p5/3P4/4K3 w - - 0 1"));
        let mut k = 0;
        for_each_pair(&mixed, |_, _, _, _| k += 1);
        assert!(k > 0);
    }

    #[test]
    fn read_leb128_section_extra_errors() {
        // Magic present but length missing.
        let mut only_magic = Vec::new();
        only_magic.extend_from_slice(LEB128_MAGIC);
        assert!(read_leb128_section(&only_magic, &mut 0, 1, decode_leb128_i16).is_err());
        // Bad magic.
        assert!(read_leb128_section(&[0u8; 17], &mut 0, 1, decode_leb128_i16).is_err());
        // Truncated payload after a valid header.
        let mut data = Vec::new();
        data.extend_from_slice(LEB128_MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.push(0x80);
        assert!(read_leb128_section(&data, &mut 0, 1, decode_leb128_i16).is_err());
    }

    #[test]
    fn decode_leb128_overflow_and_truncation() {
        // Empty input truncates.
        assert!(decode_leb128_i64(&[], &mut 0, 32).is_err());
        // 6 continuation bytes overflow a 32-bit payload.
        assert!(decode_leb128_i64(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80], &mut 0, 32).is_err());
        // Multi-byte positive decodes.
        assert_eq!(decode_leb128_i64(&[0xAC, 0x02], &mut 0, 32), Ok(300));
        // Sign extension for 16-bit -2 (0x7E).
        assert_eq!(decode_leb128_i16(&[0x7E], &mut 0), Ok(-2));
    }

    fn build_valid_arch_bytes(fc0_in: usize) -> Vec<u8> {
        let hash = arch_hash(fc0_in as u32);
        let padded0 = fc0_in.next_multiple_of(32);
        let mut data = Vec::new();
        data.extend_from_slice(&hash.to_le_bytes());
        for _ in 0..FC0_OUT {
            data.extend_from_slice(&0i32.to_le_bytes());
        }
        data.extend(std::iter::repeat_n(0u8, FC0_OUT * padded0));
        for _ in 0..FC1_OUT {
            data.extend_from_slice(&0i32.to_le_bytes());
        }
        data.extend(std::iter::repeat_n(0u8, FC1_OUT * 64));
        data.extend_from_slice(&0i32.to_le_bytes());
        data.extend(std::iter::repeat_n(0u8, FC0_OUT * 2 + FC1_OUT * 2));
        data
    }

    #[test]
    fn sfnn_arch_load_success_and_truncations() {
        let data = build_valid_arch_bytes(32);
        let mut pos = 0;
        let arch = SfnnArch::load(&data, &mut pos, 32, arch_hash(32)).unwrap();
        assert_eq!(pos, data.len());
        assert!(!arch.is_rudi);
        // Bad hash.
        assert!(SfnnArch::load(&data, &mut 0, 32, 0x12345678).is_err());
        // Truncated tails hit each section.
        for cut in [1usize, 100, 500, 1500, 3000] {
            let mut p = 0;
            assert!(
                SfnnArch::load(&data[..data.len() - cut], &mut p, 32, arch_hash(32)).is_err(),
                "cut {cut} should fail"
            );
        }
    }

    #[test]
    fn sfnn16_load_bytes_early_errors() {
        // Bad version.
        assert!(Sfnn16Net::load_bytes(&[0, 0, 0, 0], true, 2).is_err());
        // Correct version but wrong file hash (threat net).
        let mut bad_hash = Vec::new();
        bad_hash.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
        bad_hash.extend_from_slice(&0u32.to_le_bytes());
        assert!(Sfnn16Net::load_bytes(&bad_hash, true, 2).is_err());
        // Same prefix without threat check proceeds to description length.
        assert!(Sfnn16Net::load_bytes(&bad_hash, false, 2).is_err());
        // Truncated description.
        let mut desc = Vec::new();
        desc.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
        desc.extend_from_slice(&network_hash(true, 2).to_le_bytes());
        desc.extend_from_slice(&100u32.to_le_bytes());
        assert!(Sfnn16Net::load_bytes(&desc, true, 2).is_err());
        // Bad transformer hash.
        let mut th = Vec::new();
        th.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
        th.extend_from_slice(&network_hash(true, 2).to_le_bytes());
        th.extend_from_slice(&0u32.to_le_bytes());
        th.extend_from_slice(&0u32.to_le_bytes());
        assert!(Sfnn16Net::load_bytes(&th, true, 2).is_err());
        // Correct transformer hash but truncated bias section.
        let mut ok = Vec::new();
        ok.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
        ok.extend_from_slice(&network_hash(true, 2).to_le_bytes());
        ok.extend_from_slice(&0u32.to_le_bytes());
        ok.extend_from_slice(&transformer_hash(true, 2).to_le_bytes());
        assert!(Sfnn16Net::load_bytes(&ok, true, 2).is_err());
        // Bogus option never touches the global nets.
        assert!(set_eval_file("Bogus", "x").is_err());
        assert!(set_eval_file("EvalFileSmall", "x").is_ok());
    }

    #[test]
    fn load_rudi_extra_rejections() {
        // Wrong payload length with RUDI header.
        let mut bad = Vec::new();
        bad.extend_from_slice(b"RUDI");
        bad.extend_from_slice(&123u32.to_le_bytes());
        bad.extend(std::iter::repeat_n(0u8, 123));
        assert!(Sfnn16Net::load_rudi(&bad).is_err());
        // Declared length does not match actual length.
        let mut short = Vec::new();
        short.extend_from_slice(b"RUDI");
        short.extend_from_slice(&181_011_108u32.to_le_bytes());
        short.extend(std::iter::repeat_n(0u8, 100));
        assert!(Sfnn16Net::load_rudi(&short).is_err());
        // Plain blob of an unrelated size.
        assert!(Sfnn16Net::load_rudi(&[1u8; 100]).is_err());
        // Truncated RUDI helpers.
        assert!(read_rudi_i16(&[0x01], &mut 0, 1).is_err());
        assert!(read_rudi_i32(&[0x01, 0x02], &mut 0, 1).is_err());
    }

    #[test]
    fn propagate_zero_nets_are_deterministic() {
        let mk = |rudi: bool| SfnnArch {
            fc0_bias: [0; FC0_OUT],
            fc0_w: vec![0; 32 * FC0_OUT],
            fc1_bias: [0; FC1_OUT],
            fc1_w: vec![0; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
            is_rudi: rudi,
        };
        let input = [0u8; 32];
        // Rudi zero net: ((0 - 2.8) * 100) * 16 = -4480.
        assert_eq!(propagate(&mk(true), 32, &input), -4480);
        // Stockfish zero net: all-zero dot product.
        assert_eq!(propagate(&mk(false), 32, &input), 0);
        // Same call twice is deterministic.
        assert_eq!(
            propagate(&mk(true), 32, &input),
            propagate(&mk(true), 32, &input)
        );
    }

    #[test]
    fn scatter_add_sub_roundtrip() {
        let l1 = 32usize;
        let tr = SfnnTransformer {
            bias: vec![0; l1],
            weights: vec![3i16; 2 * l1],
            threat_w: Vec::new(),
            threat_w_i16: Vec::new(),
            psqt_w: vec![5i32; 2 * N_BUCKETS],
            threat_psqt_w: Vec::new(),
            pair_w: Vec::new(),
            pair_w_i16: Vec::new(),
            pair_psqt_w: Vec::new(),
        };
        let mut acc = [0i16; 32];
        let mut psqt = [0i32; N_BUCKETS];
        scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, 1);
        assert!(acc.iter().all(|&v| v == 3));
        assert!(psqt.iter().all(|&v| v == 5));
        scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, -1);
        assert!(acc.iter().all(|&v| v == 0));
        assert!(psqt.iter().all(|&v| v == 0));
        // Scalar tail (l1 not a multiple of 16) also round-trips.
        let l1s = 15usize;
        let trs = SfnnTransformer {
            bias: vec![0; l1s],
            weights: vec![2i16; 2 * l1s],
            threat_w: Vec::new(),
            threat_w_i16: Vec::new(),
            psqt_w: vec![1i32; 2 * N_BUCKETS],
            threat_psqt_w: Vec::new(),
            pair_w: Vec::new(),
            pair_w_i16: Vec::new(),
            pair_psqt_w: Vec::new(),
        };
        let mut accs = [0i16; 15];
        let mut ps = [0i32; N_BUCKETS];
        scatter_halfka(&trs, l1s, &[1], &mut accs, &mut ps, 1);
        scatter_halfka(&trs, l1s, &[1], &mut accs, &mut ps, -1);
        assert!(accs.iter().all(|&v| v == 0));
        assert!(ps.iter().all(|&v| v == 0));
    }

    #[test]
    fn kingless_paths_are_safe_without_global_mutation() {
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        assert!(!kings_present(&pos));
        // Evaluate bails out before touching any network.
        let mut accs = Sfnn16Accs::empty();
        assert!(evaluate_nets(&pos, &mut accs, Side::White).is_none());
        // Refresh on a kingless board is a no-op.
        let mut accs = Sfnn16Accs::empty();
        refresh_all(&pos, &mut accs);
        ensure_fresh(&pos, &mut accs);
        // Queued deltas on a kingless board are dropped.
        let mut pending = SfnnPending::default();
        note_add(&mut pending, Square::E2, Side::White, Piece::Pawn);
        flush_pending(&pos, &mut accs, &mut pending);
        assert_eq!(pending.n_adds, 0);
        // Incremental apply never panics, whatever the global net state.
        let start =
            BoardState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let spos = SfnnPosition::from_board(&start);
        let mut accs = Sfnn16Accs::empty();
        apply_queued(&spos, &mut accs, &[], &[], &[false, false]);
        apply_queued(&spos, &mut accs, &[], &[], &[true, true]);
    }

    #[test]
    fn eval_with_net_small_nets_cover_all_branches() {
        let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let base = [10i16, 20];
        let psqt = [3i32; N_BUCKETS];
        let mk = |rudi: bool, threats: bool| Sfnn16Net {
            l1: 2,
            use_threats: threats,
            is_rudi: rudi,
            transformer: SfnnTransformer {
                bias: Vec::new(),
                weights: Vec::new(),
                threat_w: if !rudi && threats {
                    vec![0i8; THREAT_DIMS * 2]
                } else {
                    Vec::new()
                },
                threat_w_i16: vec![0i16; THREAT_DIMS * 2],
                psqt_w: Vec::new(),
                threat_psqt_w: if !rudi && threats {
                    vec![0i32; THREAT_DIMS * N_BUCKETS]
                } else {
                    Vec::new()
                },
                pair_w: Vec::new(),
                pair_w_i16: if rudi {
                    vec![0i16; PAIR_DIMS * 2]
                } else {
                    Vec::new()
                },
                pair_psqt_w: Vec::new(),
            },
            stacks: vec![SfnnArch {
                fc0_bias: [0; FC0_OUT],
                fc0_w: vec![0; 2 * FC0_OUT],
                fc1_bias: [0; FC1_OUT],
                fc1_w: vec![0; FC1_IN * FC1_OUT],
                fc2_bias: 0,
                fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
                is_rudi: rudi,
            }],
        };
        // Rudi with and without threats exercises the /255 vs /512 split.
        let rudi_t = mk(true, true);
        let rudi_nt = mk(true, false);
        let (psqt_t, pos_t) = eval_with_net(&pos, &rudi_t, [&base; 2], [&psqt; 2], Side::White, 0);
        let (psqt_nt, pos_nt) =
            eval_with_net(&pos, &rudi_nt, [&base; 2], [&psqt; 2], Side::White, 0);
        assert_eq!(
            (psqt_t, pos_t),
            eval_with_net(&pos, &rudi_t, [&base; 2], [&psqt; 2], Side::White, 0)
        );
        // Stockfish net without threats takes the no-threat AVX2/scalar tail.
        let sf_nt = mk(false, false);
        let _ = eval_with_net(&pos, &sf_nt, [&base; 2], [&psqt; 2], Side::White, 0);
        let _ = eval_with_net(&pos, &sf_nt, [&base; 2], [&psqt; 2], Side::Black, 0);
        // Threats disabled must ignore threat weights entirely.
        assert_eq!(
            (psqt_nt, pos_nt),
            eval_with_net(&pos, &rudi_nt, [&base; 2], [&psqt; 2], Side::White, 0)
        );
        // Trainer helper stays consistent on the same position.
        let (w_h, b_h, w_t, b_t) =
            trainer_features(&pos.pieces, pos.white, pos.black, &pos.mapping);
        assert!(!w_h.is_empty() && !b_h.is_empty());
        assert!(!w_t.is_empty() && !b_t.is_empty());
        assert_eq!(
            simple_eval(&pos, Side::White),
            -simple_eval(&pos, Side::Black)
        );
    }

    fn push_leb128_zeros_for_test(buf: &mut Vec<u8>, count: usize) {
        buf.extend_from_slice(LEB128_MAGIC);
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        buf.extend(std::iter::repeat_n(0u8, count));
    }

    fn build_tiny_no_threats_for_test(l1: usize, version: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        push_leb128_zeros_for_test(&mut buf, l1);
        push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * l1);
        push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * N_BUCKETS);
        let arch = build_valid_arch_bytes(l1);
        for _ in 0..N_BUCKETS {
            buf.extend_from_slice(&arch);
        }
        buf
    }

    fn build_tiny_combined_threats_for_test(l1: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&network_hash(true, l1 as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&transformer_hash(true, l1 as u32).to_le_bytes());
        push_leb128_zeros_for_test(&mut buf, l1);
        push_leb128_zeros_for_test(&mut buf, (THREAT_DIMS + PSQ_DIMS) * l1);
        push_leb128_zeros_for_test(&mut buf, (THREAT_DIMS + PSQ_DIMS) * N_BUCKETS);
        let arch = build_valid_arch_bytes(l1);
        for _ in 0..N_BUCKETS {
            buf.extend_from_slice(&arch);
        }
        buf
    }

    fn build_tiny_sf17_for_test(l1: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
        buf.extend_from_slice(&network_hash(true, l1 as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&transformer_hash(true, l1 as u32).to_le_bytes());
        push_leb128_zeros_for_test(&mut buf, l1);
        buf.extend(std::iter::repeat_n(0u8, THREAT_DIMS * l1));
        push_leb128_zeros_for_test(&mut buf, THREAT_DIMS * N_BUCKETS);
        buf.extend(std::iter::repeat_n(0u8, PAIR_DIMS * l1));
        push_leb128_zeros_for_test(&mut buf, PAIR_DIMS * N_BUCKETS);
        push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * l1);
        push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * N_BUCKETS);
        let arch = build_valid_arch_bytes(l1);
        for _ in 0..N_BUCKETS {
            buf.extend_from_slice(&arch);
        }
        buf
    }

    fn dummy_full_loaded_nets_for_test() -> LoadedNets {
        LoadedNets {
            net: Sfnn16Net {
                l1: L1,
                use_threats: false,
                is_rudi: false,
                transformer: SfnnTransformer {
                    bias: vec![7i16; L1],
                    weights: vec![0i16; PSQ_DIMS * L1],
                    threat_w: Vec::new(),
                    threat_w_i16: Vec::new(),
                    psqt_w: vec![0i32; PSQ_DIMS * N_BUCKETS],
                    threat_psqt_w: Vec::new(),
                    pair_w: Vec::new(),
                    pair_w_i16: Vec::new(),
                    pair_psqt_w: Vec::new(),
                },
                stacks: Vec::new(),
            },
        }
    }

    #[test]
    fn propagate_rudi_scalar_clamp_determinism() {
        let l1 = 15usize;
        let mk = |bias: i32| SfnnArch {
            fc0_bias: [bias; FC0_OUT],
            fc0_w: vec![0i8; l1 * FC0_OUT],
            fc1_bias: [0i32; FC1_OUT],
            fc1_w: vec![1i8; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: {
                let mut w = [0i8; FC0_OUT * 2 + FC1_OUT * 2];
                w[0] = 64;
                w
            },
            is_rudi: true,
        };
        let input = vec![10u8; l1];
        let low = propagate(&mk(-1_000_000), l1, &input);
        let mid = propagate(&mk(8160), l1, &input);
        let high = propagate(&mk(1_000_000), l1, &input);
        assert_eq!(low, propagate(&mk(-1_000_000), l1, &input));
        assert_eq!(mid, propagate(&mk(8160), l1, &input));
        assert_eq!(high, propagate(&mk(1_000_000), l1, &input));
        for v in [low, mid, high] {
            assert!((-464_000..=464_000).contains(&v), "rudi {v} out of range");
        }
        assert!(low < mid, "low {low} should be < mid {mid}");
        assert!(mid <= high, "mid {mid} should be <= high {high}");
        let mut arch = mk(8160);
        arch.fc0_w.iter_mut().for_each(|w| *w = 2);
        let out = propagate(&arch, l1, &input);
        assert!((-464_000..=464_000).contains(&out));
        assert_eq!(out, propagate(&arch, l1, &input));
    }

    #[test]
    fn propagate_nonrudi_scalar_skip_and_loops() {
        let l1 = 15usize;
        let mut arch = SfnnArch {
            fc0_bias: [0i32; FC0_OUT],
            fc0_w: vec![0i8; l1 * FC0_OUT],
            fc1_bias: [0i32; FC1_OUT],
            fc1_w: vec![0i8; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: [0i8; FC0_OUT * 2 + FC1_OUT * 2],
            is_rudi: false,
        };
        arch.fc0_bias[30] = 1000;
        arch.fc0_bias[31] = 200;
        let input = vec![0u8; l1];
        let out = propagate(&arch, l1, &input);
        assert_eq!(out, ((800i64 * 600 * 16) / (128 * 64 * 2)) as i32);
        assert_eq!(out, propagate(&arch, l1, &input));
        arch.fc0_bias[30] = 200;
        arch.fc0_bias[31] = 1000;
        let out2 = propagate(&arch, l1, &input);
        assert_ne!(out, out2);
        arch.fc0_bias = [50_000; FC0_OUT];
        arch.fc1_w.iter_mut().for_each(|w| *w = 1);
        arch.fc2_w.iter_mut().for_each(|w| *w = 1);
        let big = propagate(&arch, l1, &input);
        assert_eq!(big, propagate(&arch, l1, &input));
        arch.fc0_bias = [-50_000; FC0_OUT];
        let small = propagate(&arch, l1, &input);
        assert_ne!(big, small);
        arch.fc0_bias = [0; FC0_OUT];
        arch.fc0_w.iter_mut().for_each(|w| *w = 3);
        arch.fc1_w.iter_mut().for_each(|w| *w = 2);
        arch.fc2_w.iter_mut().for_each(|w| *w = 1);
        arch.fc2_bias = 5;
        let input2 = vec![7u8; l1];
        let full = propagate(&arch, l1, &input2);
        assert_eq!(full, propagate(&arch, l1, &input2));
    }

    #[test]
    fn propagate_l1_1024_large_determinism() {
        let l1 = L1;
        let mk = |rudi: bool| SfnnArch {
            fc0_bias: [3i32; FC0_OUT],
            fc0_w: {
                let mut w = vec![0i8; l1 * FC0_OUT];
                w[0] = 1;
                w[1] = -1;
                w[l1] = 2;
                w
            },
            fc1_bias: [1i32; FC1_OUT],
            fc1_w: {
                let mut w = vec![0i8; FC1_IN * FC1_OUT];
                w[0] = 1;
                w[FC1_IN] = -2;
                w
            },
            fc2_bias: 7,
            fc2_w: {
                let mut w = [0i8; FC0_OUT * 2 + FC1_OUT * 2];
                w[0] = 1;
                w[64] = -1;
                w[127] = 2;
                w
            },
            is_rudi: rudi,
        };
        let input = vec![11u8; l1];
        for rudi in [true, false] {
            let arch = mk(rudi);
            let a = propagate(&arch, l1, &input);
            let b = propagate(&arch, l1, &input);
            assert_eq!(a, b);
            let zero_input = vec![0u8; l1];
            let c = propagate(&arch, l1, &zero_input);
            assert_eq!(c, propagate(&arch, l1, &zero_input));
        }
    }

    #[test]
    fn sfnn16_load_bytes_tiny_no_threats_success() {
        for version in [SF_FILE_VERSION, SF17_FILE_VERSION] {
            let buf = build_tiny_no_threats_for_test(2, version);
            let net = Sfnn16Net::load_bytes(&buf, false, 2).unwrap();
            assert_eq!(net.l1, 2);
            assert!(!net.use_threats);
            assert!(!net.is_rudi);
            assert_eq!(net.transformer.bias.len(), 2);
            assert_eq!(net.transformer.weights.len(), PSQ_DIMS * 2);
            assert_eq!(net.transformer.psqt_w.len(), PSQ_DIMS * N_BUCKETS);
            assert_eq!(net.stacks.len(), N_BUCKETS);
        }
        let mut buf = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
        buf.push(0);
        assert_eq!(
            Sfnn16Net::load_bytes(&buf, false, 2).unwrap_err(),
            "trailing data after network"
        );
        let mut bad = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
        let arch_len = build_valid_arch_bytes(2).len();
        let arch_start = bad.len() - arch_len * N_BUCKETS;
        bad[arch_start] ^= 0xFF;
        assert!(Sfnn16Net::load_bytes(&bad, false, 2).is_err());
        let mut bad_magic = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
        bad_magic[16] ^= 0xFF;
        assert!(Sfnn16Net::load_bytes(&bad_magic, false, 2).is_err());
        let good = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
        for cut in [20usize, 100, 1000, 50_000, 150_000] {
            if cut < good.len() {
                assert!(
                    Sfnn16Net::load_bytes(&good[..good.len() - cut], false, 2).is_err(),
                    "cut {cut} should fail"
                );
            }
        }
    }

    #[test]
    fn sfnn16_load_bytes_threat_combined_success() {
        let buf = build_tiny_combined_threats_for_test(2);
        let net = Sfnn16Net::load_bytes(&buf, true, 2).unwrap();
        assert_eq!(net.l1, 2);
        assert!(net.use_threats);
        assert_eq!(net.transformer.weights.len(), PSQ_DIMS * 2);
        assert_eq!(net.transformer.threat_w.len(), THREAT_DIMS * 2);
        assert_eq!(net.transformer.threat_psqt_w.len(), THREAT_DIMS * N_BUCKETS);
        assert_eq!(net.stacks.len(), N_BUCKETS);
        let mut trailing = buf.clone();
        trailing.push(0);
        assert!(Sfnn16Net::load_bytes(&trailing, true, 2).is_err());
        for cut in [20usize, 500, 50_000, 300_000] {
            assert!(
                Sfnn16Net::load_bytes(&buf[..buf.len() - cut], true, 2).is_err(),
                "cut {cut} should fail"
            );
        }
        let mut bad_arch = buf.clone();
        let arch_start = buf.len() - build_valid_arch_bytes(2).len();
        bad_arch[arch_start] ^= 0xFF;
        assert!(Sfnn16Net::load_bytes(&bad_arch, true, 2).is_err());
    }

    #[test]
    fn sfnn16_load_bytes_sf17_success() {
        let buf = build_tiny_sf17_for_test(2);
        let net = Sfnn16Net::load_bytes(&buf, true, 2).unwrap();
        assert_eq!(net.l1, 2);
        assert!(net.use_threats);
        assert_eq!(net.transformer.threat_w.len(), THREAT_DIMS * 2);
        assert_eq!(net.transformer.pair_w.len(), PAIR_DIMS * 2);
        assert_eq!(net.transformer.weights.len(), PSQ_DIMS * 2);
        assert_eq!(net.stacks.len(), N_BUCKETS);
        let pair_offset = {
            let mut pos = 0;
            pos += 4 + 4 + 4;
            pos += LEB128_MAGIC.len() + 4 + 2;
            pos += THREAT_DIMS * 2;
            pos += LEB128_MAGIC.len() + 4 + THREAT_DIMS * N_BUCKETS;
            pos
        };
        let truncated = buf[..pair_offset + 10].to_vec();
        assert_eq!(
            Sfnn16Net::load_bytes(&truncated, true, 2).unwrap_err(),
            "truncated pair weights"
        );
        let mut trailing = buf.clone();
        trailing.extend_from_slice(&[0u8; 4]);
        assert!(Sfnn16Net::load_bytes(&trailing, true, 2).is_err());
    }

    #[test]
    fn load_file_delegates_without_global() {
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let rudi_path = dir.join(format!("whale_v16_test_rudi_{pid}.tmp"));
        let normal_path = dir.join(format!("whale_v16_test_normal_{pid}.tmp"));
        let mut bad_rudi = Vec::new();
        bad_rudi.extend_from_slice(b"RUDI");
        bad_rudi.extend_from_slice(&123u32.to_le_bytes());
        bad_rudi.extend(std::iter::repeat_n(0u8, 123));
        std::fs::write(&rudi_path, &bad_rudi).unwrap();
        let rudi_str = rudi_path.to_string_lossy().into_owned();
        assert!(Sfnn16Net::load_file(&rudi_str, true, L1).is_err());
        std::fs::write(&normal_path, b"bad!").unwrap();
        let normal_str = normal_path.to_string_lossy().into_owned();
        assert!(Sfnn16Net::load_file(&normal_str, true, L1).is_err());
        assert!(Sfnn16Net::load_file(&normal_str, false, 2).is_err());
        let _ = std::fs::remove_file(&rudi_path);
        let _ = std::fs::remove_file(&normal_path);
    }

    #[test]
    fn eval_with_net_sf_threats_covers_branches() {
        let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let base = [10i16, 20];
        let psqt = [3i32; N_BUCKETS];
        let mk = |with_pairs: bool| Sfnn16Net {
            l1: 2,
            use_threats: true,
            is_rudi: false,
            transformer: SfnnTransformer {
                bias: Vec::new(),
                weights: Vec::new(),
                threat_w: vec![0i8; THREAT_DIMS * 2],
                threat_w_i16: Vec::new(),
                psqt_w: Vec::new(),
                threat_psqt_w: vec![0i32; THREAT_DIMS * N_BUCKETS],
                pair_w: if with_pairs {
                    vec![0i8; PAIR_DIMS * 2]
                } else {
                    Vec::new()
                },
                pair_w_i16: Vec::new(),
                pair_psqt_w: if with_pairs {
                    vec![0i32; PAIR_DIMS * N_BUCKETS]
                } else {
                    Vec::new()
                },
            },
            stacks: vec![
                SfnnArch {
                    fc0_bias: [0; FC0_OUT],
                    fc0_w: vec![127i8; 2 * FC0_OUT],
                    fc1_bias: [0; FC1_OUT],
                    fc1_w: vec![127i8; FC1_IN * FC1_OUT],
                    fc2_bias: 0,
                    fc2_w: [127i8; FC0_OUT * 2 + FC1_OUT * 2],
                    is_rudi: false,
                };
                1
            ],
        };
        let mut net_pairs = mk(true);
        let net_nopairs = mk(false);
        let base_pairs = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
        let base_nopairs =
            eval_with_net(&pos, &net_nopairs, [&base; 2], [&psqt; 2], Side::White, 0);
        assert_eq!(
            base_pairs,
            eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0)
        );
        let mut threats = [0usize; MAX_THREAT_ACTIVE];
        let nt = collect_threats(&pos, Side::White, &mut threats);
        assert!(nt > 0);
        let t = (threats[0] - PSQ_DIMS) * 2;
        net_pairs.transformer.threat_w[t] = 127;
        net_pairs.transformer.threat_w[t + 1] = 127;
        let after_w = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
        assert_ne!(base_pairs, after_w);
        net_pairs.transformer.threat_w[t] = 0;
        net_pairs.transformer.threat_w[t + 1] = 0;
        net_pairs.transformer.threat_psqt_w[(threats[0] - PSQ_DIMS) * N_BUCKETS] = 64;
        let after_psqt = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
        assert_ne!(base_pairs, after_psqt);
        let mut pairs = [0usize; MAX_PAIR_ACTIVE];
        let np = collect_pairs(&pos, Side::White, &mut pairs);
        if np > 0 {
            let pt = (pairs[0] - PSQ_DIMS - THREAT_DIMS) * 2;
            net_pairs.transformer.pair_w[pt] = 127;
            net_pairs.transformer.pair_w[pt + 1] = 127;
            let after_pair =
                eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
            assert_ne!(base_pairs, after_pair);
        }
        let _ = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::Black, 0);
        let _ = eval_with_net(&pos, &net_nopairs, [&base; 2], [&psqt; 2], Side::Black, 0);
        let _ = base_nopairs;
    }

    #[test]
    fn eval_with_net_l1_1024_kings_only() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let base = [0i16; L1];
        let psqt = [0i32; N_BUCKETS];
        let mk = |rudi: bool, threats: bool| Sfnn16Net {
            l1: L1,
            use_threats: threats,
            is_rudi: rudi,
            transformer: SfnnTransformer {
                bias: Vec::new(),
                weights: Vec::new(),
                threat_w: Vec::new(),
                threat_w_i16: Vec::new(),
                psqt_w: Vec::new(),
                threat_psqt_w: Vec::new(),
                pair_w: Vec::new(),
                pair_w_i16: Vec::new(),
                pair_psqt_w: Vec::new(),
            },
            stacks: vec![
                SfnnArch {
                    fc0_bias: [0; FC0_OUT],
                    fc0_w: vec![0; L1 * FC0_OUT],
                    fc1_bias: [0; FC1_OUT],
                    fc1_w: vec![0; FC1_IN * FC1_OUT],
                    fc2_bias: 0,
                    fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
                    is_rudi: rudi,
                };
                1
            ],
        };
        for rudi in [true, false] {
            for threats in [true, false] {
                let net = mk(rudi, threats);
                for stm in [Side::White, Side::Black] {
                    let a = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], stm, 0);
                    let b = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], stm, 0);
                    assert_eq!(a, b);
                }
            }
        }
    }

    #[test]
    fn refresh_perspective_manual_full_net() {
        let loaded = dummy_full_loaded_nets_for_test();
        let board = BoardState::parse_fen("K7/8/8/8/8/8/8/k7 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let mut accs = Sfnn16Accs::empty();
        refresh_perspective(&pos, &loaded, Side::White, &mut accs);
        assert!(accs.halfka[0].iter().all(|&v| v == 7));
        assert!(accs.psqt[0].iter().all(|&v| v == 0));
        refresh_perspective(&pos, &loaded, Side::Black, &mut accs);
        assert!(accs.halfka[1].iter().all(|&v| v == 7));
        let mut accs2 = Sfnn16Accs::empty();
        refresh_perspective(&pos, &loaded, Side::White, &mut accs2);
        assert_eq!(accs.halfka[0], accs2.halfka[0]);
        assert_eq!(accs.psqt[0], accs2.psqt[0]);
        let board2 = BoardState::parse_fen("K7/8/8/8/2P5/8/8/k7 w - - 0 1");
        let pos2 = SfnnPosition::from_board(&board2);
        refresh_perspective(&pos2, &loaded, Side::White, &mut accs2);
        assert!(accs2.halfka[0].iter().all(|&v| v == 7));
    }

    #[test]
    fn finny_incremental_manual_full_net() {
        let loaded = dummy_full_loaded_nets_for_test();
        let fens = [
            "K7/8/8/8/8/8/8/k7 w - - 0 1",
            "K7/8/8/8/8/8/1P6/k7 w - - 0 1",
            "K7/8/8/8/8/8/1N6/k7 w - - 0 1",
            "K6R/PPPPPPPP/8/8/8/8/pppppppp/k6r w - - 0 1",
        ];
        for fen in fens {
            let board = BoardState::parse_fen(fen);
            let pos = SfnnPosition::from_board(&board);
            for perspective in [Side::White, Side::Black] {
                let mut direct = Sfnn16Accs::empty();
                refresh_perspective(&pos, &loaded, perspective, &mut direct);
                let mut via_finny = Sfnn16Accs::empty();
                update_perspective_finny_or_refresh(&pos, &loaded, perspective, &mut via_finny);
                assert_eq!(
                    direct.halfka[perspective as usize], via_finny.halfka[perspective as usize],
                    "finny mismatch for {fen} {perspective:?} first touch"
                );
                let mut via_finny2 = Sfnn16Accs::empty();
                update_perspective_finny_or_refresh(&pos, &loaded, perspective, &mut via_finny2);
                assert_eq!(
                    direct.halfka[perspective as usize], via_finny2.halfka[perspective as usize],
                    "finny mismatch for {fen} {perspective:?} second touch"
                );
                assert_eq!(
                    direct.psqt[perspective as usize],
                    via_finny2.psqt[perspective as usize]
                );
            }
        }
        let board_b = BoardState::parse_fen(fens[1]);
        let mut pos_b = SfnnPosition::from_board(&board_b);
        pos_b.white |= 1u64 << 18;
        let mut acc_inc = Sfnn16Accs::empty();
        let loaded_ref = &loaded;
        for perspective in [Side::White, Side::Black] {
            let mut direct = Sfnn16Accs::empty();
            refresh_perspective(&pos_b, loaded_ref, perspective, &mut direct);
            update_perspective_finny_or_refresh(&pos_b, loaded_ref, perspective, &mut acc_inc);
            assert_eq!(
                direct.halfka[perspective as usize],
                acc_inc.halfka[perspective as usize]
            );
        }
    }

    #[test]
    fn scatter_empty_feats_safe() {
        let l1 = 8usize;
        let tr = SfnnTransformer {
            bias: vec![0; l1],
            weights: vec![1i16; 2 * l1],
            threat_w: Vec::new(),
            threat_w_i16: Vec::new(),
            psqt_w: vec![2i32; 2 * N_BUCKETS],
            threat_psqt_w: Vec::new(),
            pair_w: Vec::new(),
            pair_w_i16: Vec::new(),
            pair_psqt_w: Vec::new(),
        };
        let mut acc = [0i16; 8];
        let mut psqt = [0i32; N_BUCKETS];
        scatter_halfka(&tr, l1, &[], &mut acc, &mut psqt, 1);
        assert!(acc.iter().all(|&v| v == 0));
        assert!(psqt.iter().all(|&v| v == 0));
        scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, 1);
        assert!(acc.iter().all(|&v| v == 1));
        scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, -1);
        assert!(acc.iter().all(|&v| v == 0));
    }

    #[test]
    fn threat_pair_edge_cases_no_global() {
        assert_eq!(pseudo_attacks_sf(0, 0), 0);
        assert_eq!(pseudo_attacks_sf(7, 10), 0);
        assert_eq!(pseudo_attacks_sf(8, 10), 0);
        assert_eq!(pseudo_attacks_sf(15, 10), 0);
        let kingless_attack =
            SfnnPosition::from_board(&BoardState::parse_fen("8/8/8/3p4/4P3/8/8/8 w - - 0 1"));
        let mut n = 0;
        for_each_threat(&kingless_attack, |_, _, _, _| n += 1);
        assert!(n > 0);
        let mut inconsistent =
            SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
        inconsistent.white |= 1u64 << 18;
        let mut skipped = 0;
        for_each_threat(&inconsistent, |_, _, _, _| skipped += 1);
        let _ = skipped;
        let mut dense_pieces = [0u64; 6];
        dense_pieces[Piece::Queen as usize] = u64::MAX;
        let dense = SfnnPosition {
            pieces: dense_pieces,
            white: 0x0000_0000_FFFF_FFFF,
            black: 0xFFFF_FFFF_0000_0000,
            mapping: [4u8; 64],
        };
        let mut tbuf = [0usize; MAX_THREAT_ACTIVE];
        let tn = collect_threats(&dense, Side::White, &mut tbuf);
        assert!(tn > 100 && tn <= MAX_THREAT_ACTIVE);
        let dense_p = SfnnPosition::from_board(&BoardState::parse_fen(
            "8/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/8 w - - 0 1",
        ));
        let mut pbuf = [0usize; MAX_PAIR_ACTIVE];
        let pn = collect_pairs(&dense_p, Side::White, &mut pbuf);
        assert!(pn > 0);
        for &idx in &pbuf[..pn] {
            assert!((PSQ_DIMS + THREAT_DIMS..PSQ_DIMS + THREAT_DIMS + PAIR_DIMS).contains(&idx));
        }
        let mut found_none = false;
        let mut found_some = false;
        for attacker in [1usize, 2, 3, 4, 5] {
            for attacked in [1usize, 2, 3, 4, 5, 9, 10, 11, 12, 13] {
                for (from, to) in [(10usize, 20usize), (20, 10)] {
                    match threat_index_for(Side::White, attacker, from, to, attacked, 4) {
                        None => found_none = true,
                        Some(idx) => {
                            found_some = true;
                            assert!(idx < THREAT_DIMS);
                        }
                    }
                }
            }
        }
        assert!(found_none && found_some);
    }

    #[test]
    fn kingless_evaluate_board_deterministic() {
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let mut accs = Sfnn16Accs::empty();
        assert!(evaluate_nets(&pos, &mut accs, Side::White).is_none());
        let gen_before = accs.generation;
        refresh_all(&pos, &mut accs);
        assert_eq!(accs.generation, gen_before);
        ensure_fresh(&pos, &mut accs);
        let mut pending = SfnnPending::default();
        note_add(&mut pending, Square::E2, Side::White, Piece::Pawn);
        flush_pending(&pos, &mut accs, &mut pending);
        assert_eq!(pending.n_adds, 0);
        apply_queued(&pos, &mut accs, &[], &[], &[false, false]);
        let mut board_mut = BoardState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");
        assert!(evaluate_board(&mut board_mut).is_none());
        assert!(evaluate_board_detailed(&mut board_mut).is_none());
    }

    #[test]
    fn sfnn_arch_load_various_paddings() {
        for fc0_in in [2usize, 15, 32, 33, 64] {
            let data = build_valid_arch_bytes(fc0_in);
            let mut pos = 0;
            let arch = SfnnArch::load(&data, &mut pos, fc0_in, arch_hash(fc0_in as u32)).unwrap();
            assert_eq!(pos, data.len());
            assert_eq!(arch.fc0_w.len(), FC0_OUT * fc0_in.next_multiple_of(32));
        }
    }

    #[test]
    fn read_rudi_overflow_guards() {
        assert_eq!(
            read_rudi_i16(&[], &mut 0, usize::MAX),
            Err("bad whale length")
        );
        assert_eq!(
            read_rudi_i32(&[], &mut 0, usize::MAX),
            Err("bad whale length")
        );
    }

    #[test]
    fn cover_for_each_threat_skips_empty_mapping() {
        // Lines 469 (pawn) and 499 (non-pawn): `continue` when the victim square
        // is occupied in the bitboards but empty in `mapping` (> 5). Crafted
        // inconsistent positions trigger those guards without touching globals.
        let pawn_pos = SfnnPosition {
            pieces: {
                let mut p = [0u64; 6];
                p[Piece::Pawn as usize] = 1u64 << 12;
                p[Piece::King as usize] = (1u64 << 4) | (1u64 << 60);
                p
            },
            white: (1u64 << 12) | (1u64 << 4),
            black: (1u64 << 19) | (1u64 << 60),
            mapping: {
                let mut m = [6u8; 64];
                m[12] = Piece::Pawn as u8;
                m[4] = Piece::King as u8;
                m[60] = Piece::King as u8;
                m[19] = 6;
                m
            },
        };
        // White pawn on SF 12 attacks SF 19; victim mapping 6 skips the edge.
        let mut pawn_edges = 0;
        for_each_threat(&pawn_pos, |_, _, _, _| pawn_edges += 1);
        let mut pawn_fixed = pawn_pos;
        pawn_fixed.mapping[19] = Piece::Pawn as u8;
        pawn_fixed.pieces[Piece::Pawn as usize] |= 1u64 << 19;
        let mut pawn_fixed_edges = 0;
        for_each_threat(&pawn_fixed, |_, _, _, _| pawn_fixed_edges += 1);
        assert!(pawn_fixed_edges > pawn_edges);

        // Knight on SF 10 reaches SF 20; victim mapping 6 skips the edge.
        assert_ne!(knight_attacks_sf(10) & (1u64 << 20), 0);
        let knight_pos = SfnnPosition {
            pieces: {
                let mut p = [0u64; 6];
                p[Piece::Knight as usize] = 1u64 << 10;
                p[Piece::King as usize] = (1u64 << 4) | (1u64 << 60);
                p
            },
            white: (1u64 << 10) | (1u64 << 4),
            black: (1u64 << 20) | (1u64 << 60),
            mapping: {
                let mut m = [6u8; 64];
                m[10] = Piece::Knight as u8;
                m[4] = Piece::King as u8;
                m[60] = Piece::King as u8;
                m[20] = 6;
                m
            },
        };
        let mut knight_edges = 0;
        for_each_threat(&knight_pos, |_, _, _, _| knight_edges += 1);
        let mut knight_fixed = knight_pos;
        knight_fixed.mapping[20] = Piece::Pawn as u8;
        knight_fixed.pieces[Piece::Pawn as usize] |= 1u64 << 20;
        let mut knight_fixed_edges = 0;
        for_each_threat(&knight_fixed, |_, _, _, _| knight_fixed_edges += 1);
        assert!(knight_fixed_edges > knight_edges);
    }

    #[test]
    fn cover_threat_index_never_overflows() {
        // Line 434 `return None` when `index >= THREAT_DIMS` looks defensive:
        // exhaustive sweep documents that no valid combo reaches it.
        let mut max_seen = 0usize;
        for perspective in [Side::White, Side::Black] {
            for attacker in 0..16usize {
                for from in 0..64usize {
                    for to in 0..64usize {
                        for attacked in 0..16usize {
                            for ksq in [0usize, 4, 60] {
                                if let Some(idx) = threat_make_index(
                                    perspective,
                                    attacker,
                                    from,
                                    to,
                                    attacked,
                                    ksq,
                                ) {
                                    max_seen = max_seen.max(idx);
                                    assert!(idx < THREAT_DIMS);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(max_seen < THREAT_DIMS);
    }

    #[test]
    fn cover_load_bytes_truncation_hits_pp_and_combined() {
        // Lines 1043 (SF17 pp `?`) and 1056 (combined weights `?`): truncate
        // inside those exact sections so the `?` propagates an error.
        let sf17 = build_tiny_sf17_for_test(2);
        // Sweep truncations; several must land inside the 36k pp payload and fail.
        let mut saw_err = 0;
        let mut len = sf17.len();
        while len > 100 {
            len = len.saturating_sub(20_000);
            if Sfnn16Net::load_bytes(&sf17[..len], true, 2).is_err() {
                saw_err += 1;
            }
            if len <= 100 {
                break;
            }
        }
        assert!(saw_err > 0);
        // Targeted cut inside pp: after threat-PSQT + pair-raw, inside pp section.
        // Offsets for l1=2: header/thash/bias=38, threat-raw=119_616,
        // threat-PSQT~=478_484, pair-raw=9_120, then pp MAGIC+len+payload.
        let pp_start = 38 + THREAT_DIMS * 2 + (16 + 4 + THREAT_DIMS * N_BUCKETS) + PAIR_DIMS * 2;
        assert!(Sfnn16Net::load_bytes(&sf17[..pp_start + 10], true, 2).is_err());
        assert!(Sfnn16Net::load_bytes(&sf17[..pp_start + 30], true, 2).is_err());

        // Combined weights section starts right after header/bias (38).
        let combined = build_tiny_combined_threats_for_test(2);
        let early = 38 + 16 + 4 + 1000;
        assert!(Sfnn16Net::load_bytes(&combined[..early], true, 2).is_err());
        let mid = combined.len() - 50_000;
        assert!(Sfnn16Net::load_bytes(&combined[..mid], true, 2).is_err());
    }

    #[test]
    fn cover_load_rudi_raw_payload() {
        // Line 1119 `data` for header-less raw payloads of exact length.
        let raw = vec![0u8; 181_011_108];
        let net = Sfnn16Net::load_rudi(&raw).expect("raw zero payload should parse");
        assert!(net.use_threats);
        assert_eq!(net.transformer.threat_w_i16.len(), THREAT_DIMS * L1);
        assert_eq!(net.transformer.pair_w_i16.len(), PAIR_DIMS * L1);
    }

    #[test]
    fn cover_finny_changed_common_overflow() {
        // Lines 1807 (`break` when diff exceeds 8 via changed_common) and 1854
        // (fallthrough to refresh when inner `diff <= 8` is false). Same
        // occupancy, 9 mapping flips on common squares.
        let loaded = dummy_full_loaded_nets_for_test();
        let board = BoardState::parse_fen("6k1/8/8/8/8/P7/PPPPPPPP/1K6 w - - 0 1");
        let pos_a = SfnnPosition::from_board(&board);
        let occ = pos_a.occupied();
        assert!(occ.count_ones() >= 11);
        let mut pos_b = SfnnPosition {
            pieces: pos_a.pieces,
            white: pos_a.white,
            black: pos_a.black,
            mapping: pos_a.mapping,
        };
        let mut flipped = 0;
        for (s, cell) in pos_b.mapping.iter_mut().enumerate() {
            if ((occ >> s) & 1) == 1 && *cell == Piece::Pawn as u8 && flipped < 9 {
                *cell = Piece::Knight as u8;
                flipped += 1;
            }
        }
        assert_eq!(flipped, 9);
        let mut acc = Sfnn16Accs::empty();
        update_perspective_finny_or_refresh(&pos_a, &loaded, Side::White, &mut acc);
        update_perspective_finny_or_refresh(&pos_b, &loaded, Side::White, &mut acc);
        let mut direct = Sfnn16Accs::empty();
        refresh_perspective(&pos_b, &loaded, Side::White, &mut direct);
        assert_eq!(
            acc.halfka[Side::White as usize],
            direct.halfka[Side::White as usize]
        );
        assert_eq!(
            acc.psqt[Side::White as usize],
            direct.psqt[Side::White as usize]
        );
    }

    #[test]
    fn cover_finny_skips_invalid_mapping() {
        // Lines 1830/1845: `if pt <= 5` false when entry/pos mapping is 6 for a
        // square in the incremental del/add sets. Small diff keeps the fast path.
        let loaded = dummy_full_loaded_nets_for_test();
        // Deletion side: entry has occ bit with mapping 6, then it disappears.
        let mut pos_a =
            SfnnPosition::from_board(&BoardState::parse_fen("6k1/8/8/8/8/8/1P6/1K6 w - - 0 1"));
        pos_a.white |= 1u64 << 18;
        // Keep pieces/white consistent except for the extra occ bit; mapping stays 6.
        assert_eq!(pos_a.mapping[18], 6);
        let pos_b =
            SfnnPosition::from_board(&BoardState::parse_fen("6k1/8/8/8/8/8/1P6/1K6 w - - 0 1"));
        let mut acc = Sfnn16Accs::empty();
        update_perspective_finny_or_refresh(&pos_a, &loaded, Side::White, &mut acc);
        update_perspective_finny_or_refresh(&pos_b, &loaded, Side::White, &mut acc);
        let mut direct = Sfnn16Accs::empty();
        refresh_perspective(&pos_b, &loaded, Side::White, &mut direct);
        assert_eq!(
            acc.halfka[Side::White as usize],
            direct.halfka[Side::White as usize]
        );
        // Addition side: new occ bit with mapping 6 appears.
        let pos_c =
            SfnnPosition::from_board(&BoardState::parse_fen("6k1/8/8/8/8/8/1P6/1K6 w - - 0 1"));
        let mut pos_d = SfnnPosition {
            pieces: pos_c.pieces,
            white: pos_c.white | (1u64 << 18),
            black: pos_c.black,
            mapping: pos_c.mapping,
        };
        pos_d.mapping[18] = 6;
        let mut acc2 = Sfnn16Accs::empty();
        update_perspective_finny_or_refresh(&pos_c, &loaded, Side::White, &mut acc2);
        update_perspective_finny_or_refresh(&pos_d, &loaded, Side::White, &mut acc2);
        let mut direct2 = Sfnn16Accs::empty();
        refresh_perspective(&pos_d, &loaded, Side::White, &mut direct2);
        assert_eq!(
            acc2.halfka[Side::White as usize],
            direct2.halfka[Side::White as usize]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cover_add_threat_w_i16_avx2_direct() {
        // Lines 1941/1944-1952/1954: direct AVX2 add helper, no globals.
        // Server and CI run on AVX2 x86_64; `cfg` gates compilation, no runtime branch.
        let mut buf = [0i32; L1];
        let w = [1i16; L1];
        unsafe { add_threat_w_i16_avx2(&mut buf, &w) };
        assert!(buf.iter().all(|&v| v == 1));
        let w2 = [-1i16; L1];
        unsafe { add_threat_w_i16_avx2(&mut buf, &w2) };
        assert!(buf.iter().all(|&v| v == 0));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn cover_rudi_1024_threat_pair_avx2() {
        // Lines 2222-2224/2236-2238: rudi AVX2 call sites need l1==1024 plus
        // non-empty threat/pair lists. Server runs AVX2 x86_64; no runtime branch.
        let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
        let pos = SfnnPosition::from_board(&board);
        let base = [5i16; L1];
        let psqt = [7i32; N_BUCKETS];
        let net = Sfnn16Net {
            l1: L1,
            use_threats: true,
            is_rudi: true,
            transformer: SfnnTransformer {
                bias: Vec::new(),
                weights: Vec::new(),
                threat_w: Vec::new(),
                threat_w_i16: vec![1i16; THREAT_DIMS * L1],
                psqt_w: Vec::new(),
                threat_psqt_w: Vec::new(),
                pair_w: Vec::new(),
                pair_w_i16: vec![2i16; PAIR_DIMS * L1],
                pair_psqt_w: Vec::new(),
            },
            stacks: vec![SfnnArch {
                fc0_bias: [0; FC0_OUT],
                fc0_w: vec![0; L1 * FC0_OUT],
                fc1_bias: [0; FC1_OUT],
                fc1_w: vec![0; FC1_IN * FC1_OUT],
                fc2_bias: 0,
                fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
                is_rudi: true,
            }],
        };
        let mut tbuf = [0usize; MAX_THREAT_ACTIVE];
        let mut pbuf = [0usize; MAX_PAIR_ACTIVE];
        assert!(collect_threats(&pos, Side::White, &mut tbuf) > 0);
        assert!(collect_pairs(&pos, Side::White, &mut pbuf) > 0);
        let a = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], Side::White, 0);
        let b = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], Side::White, 0);
        assert_eq!(a, b);
    }
}
