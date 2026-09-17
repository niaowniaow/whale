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
        for i in 0..64 {
            let rp = board.piece_mapping[i ^ 56];
            if rp != Piece::None {
                mapping[i] = rp as u8;
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

fn knight_attacks_sf(sq: usize) -> u64 {
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
    attacks
}

fn king_attacks_sf(sq: usize) -> u64 {
    let b = 1u64 << sq;
    let mut attacks = 0u64;
    attacks |= (b << 8) | (b >> 8);
    attacks |= ((b << 1) & !FILE_A_BB) | ((b >> 1) & !FILE_H_BB);
    attacks |= ((b << 9) & !FILE_A_BB) | ((b << 7) & !FILE_H_BB);
    attacks |= ((b >> 7) & !FILE_A_BB) | ((b >> 9) & !FILE_H_BB);
    attacks
}

fn pawn_attacks_sf(side: Side, sq: usize) -> u64 {
    let b = 1u64 << sq;
    if side == Side::White {
        ((b << 9) & !FILE_A_BB) | ((b << 7) & !FILE_H_BB)
    } else {
        ((b >> 7) & !FILE_A_BB) | ((b >> 9) & !FILE_H_BB)
    }
}

fn slider_attacks_sf(sf_piece_type: usize, sq: usize, occ: u64) -> u64 {
    // sf_piece_type: 3=Bishop, 4=Rook, 5=Queen.
    let f = (sq & 7) as i32;
    let r = (sq >> 3) as i32;
    let dirs: &[(i32, i32)] = match sf_piece_type {
        3 => &[(1, 1), (1, -1), (-1, 1), (-1, -1)], // Bishop
        4 => &[(1, 0), (-1, 0), (0, 1), (0, -1)],   // Rook
        _ => &[
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ], // Queen
    };
    let mut attacks = 0u64;
    for &(df, dr) in dirs {
        let (mut cf, mut cr) = (f + df, r + dr);
        while (0..8).contains(&cf) && (0..8).contains(&cr) {
            let t = (cr * 8 + cf) as usize;
            attacks |= 1u64 << t;
            if (occ >> t) & 1 == 1 {
                break;
            }
            cf += df;
            cr += dr;
        }
    }
    attacks
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

fn pawn_pair_bb(s: usize) -> u64 {
    let file = s & 7;
    let file_bb = 0x0101_0101_0101_0101u64 << file;
    let east = (file_bb << 1) & !0x0101_0101_0101_0101u64;
    let west = (file_bb >> 1) & !0x8080_8080_8080_8080u64;
    let files = file_bb | east | west;
    let rank18 = 0xFF00_0000_0000_00FFu64;
    files & !rank18 & !(1u64 << s)
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
        let in_ptr = in_row.as_ptr() as *const __m256i;
        let chunks = in_row.len() / 32;
        let l1 = in_row.len();

        for o in 0..FC0_OUT {
            let mut sum = _mm256_setzero_si256();
            let w_ptr = arch.fc0_w.as_ptr().add(o * l1) as *const __m256i;

            for j in 0..chunks {
                let in_vec = _mm256_loadu_si256(in_ptr.add(j));
                let w_vec = _mm256_loadu_si256(w_ptr.add(j));
                let mask = _mm256_set1_epi8(0x7f);
                let low = _mm256_and_si256(in_vec, mask);
                let high = _mm256_andnot_si256(mask, in_vec);
                let low32 = _mm256_madd_epi16(_mm256_maddubs_epi16(low, w_vec), ones);
                let high32 = _mm256_madd_epi16(_mm256_maddubs_epi16(high, w_vec), ones);
                sum = _mm256_add_epi32(sum, _mm256_add_epi32(low32, high32));
            }

            fc0[o] = arch.fc0_bias[o] + hsum256_ps_avx2(sum);
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

        for o in 0..FC1_OUT {
            let w_ptr = arch.fc1_w.as_ptr().add(o * 64) as *const __m256i;
            let w0 = _mm256_loadu_si256(w_ptr);
            let w1 = _mm256_loadu_si256(w_ptr.add(1));

            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(in0, w0), ones);
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(in1, w1), ones);
            let sum = _mm256_add_epi32(p0, p1);

            fc1[o] = arch.fc1_bias[o] + hsum256_ps_avx2(sum);
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
        let use_avx2 = is_x86_feature_detected!("avx2") && l1.is_multiple_of(32);
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        let mut fc0 = [0.0f32; FC0_OUT];
        if use_avx2 {
            let mut fc0_i32 = [0i32; FC0_OUT];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                fc0_avx2(arch, &input[..l1], &mut fc0_i32);
            }
            for o in 0..FC0_OUT {
                fc0[o] = (fc0_i32[o] as f32 / 16320.0).clamp(0.0, 1.0);
            }
        } else {
            for o in 0..FC0_OUT {
                let mut sum = arch.fc0_bias[o];
                let row = o * l1;
                for j in 0..l1 {
                    sum += (input[j] as i32) * (arch.fc0_w[row + j] as i32);
                }
                fc0[o] = (sum as f32 / 16320.0).clamp(0.0, 1.0);
            }
        }

        let mut pair = [0.0f32; FC1_IN];
        for o in 0..FC0_OUT {
            let c = fc0[o];
            pair[o] = c * c;
            pair[FC0_OUT + o] = c;
        }

        let mut fc1 = [0.0f32; FC1_OUT];
        for o in 0..FC1_OUT {
            let mut sum = arch.fc1_bias[o] as f32 / 16320.0;
            let row = o * FC1_IN;
            for j in 0..FC1_IN {
                let w = arch.fc1_w[row + j] as f32 / 64.0;
                sum += pair[j] * w;
            }
            fc1[o] = sum.clamp(0.0, 1.0);
        }

        let mut out = arch.fc2_bias as f32 / 16320.0;
        for j in 0..FC1_OUT {
            let w = arch.fc2_w[j] as f32 / 64.0;
            out += fc1[j] * w;
        }

        let score_cp = ((out - 2.80) * 100.0).clamp(-29000.0, 29000.0) as i32;
        return score_cp * OUTPUT_SCALE;
    }

    let mut fc0 = [0i32; FC0_OUT];
    let in_row = &input[..l1];

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && l1 == 1024;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    if use_avx2 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            fc0_avx2(arch, in_row, &mut fc0);
        }
    } else {
        for o in 0..FC0_OUT {
            let mut v = arch.fc0_bias[o];
            let row = o * l1;
            let w_row = &arch.fc0_w[row..row + l1];
            for j in 0..l1 {
                v += i32::from(in_row[j]) * i32::from(w_row[j]);
            }
            fc0[o] = v;
        }
    }

    let mut concat = [0u8; FC0_OUT * 2 + FC1_OUT * 2];
    for i in 0..FC0_OUT {
        let sqr = (((fc0[i] as i64 * fc0[i] as i64) >> 21).min(127)) as u8;
        let clip = (fc0[i] >> 7).clamp(0, 127) as u8;
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
        for o in 0..FC1_OUT {
            let mut v = arch.fc1_bias[o];
            let row = o * (FC0_OUT * 2);
            let w_row = &arch.fc1_w[row..row + FC0_OUT * 2];
            for j in 0..FC0_OUT * 2 {
                v += i32::from(concat_fc0[j]) * i32::from(w_row[j]);
            }
            fc1[o] = v;
        }
    }

    for i in 0..FC1_OUT {
        let sqr = (((fc1[i] as i64 * fc1[i] as i64) >> 19).min(127)) as u8;
        let clip = (fc1[i] >> 6).clamp(0, 127) as u8;
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
        let mut d = 0i32;
        for j in 0..arch.fc2_w.len() {
            d += i32::from(concat[j]) * i32::from(arch.fc2_w[j]);
        }
        d
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
static NETS_GEN: AtomicU64 = AtomicU64::new(0);
static PENDING_PATH: RwLock<Option<String>> = RwLock::new(None);

fn loaded_nets() -> Option<Arc<LoadedNets>> {
    NETS.read().ok().and_then(|guard| guard.clone())
}

pub fn try_load_default_path() -> Option<&'static str> {
    if maintenance_active() {
        return Some("active");
    }
    let candidates = [
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

pub fn maintenance_active() -> bool {
    loaded_nets().is_some()
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
    if let Ok(mut guard) = NETS.write() {
        *guard = Some(Arc::new(LoadedNets { net }));
    }
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    crate::eval::nnue::clear_eval_cache();
    Ok(())
}

pub fn unload_nets() {
    if let Ok(mut guard) = NETS.write() {
        *guard = None;
    }
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    crate::eval::nnue::clear_eval_cache();
}

pub fn set_eval_file(which: &str, path: &str) -> Result<&'static str, &'static str> {
    if which.eq_ignore_ascii_case("EvalFileSmall") {
        return Ok("EvalFileSmall is deprecated and ignored (SFNNv16 uses a single network)");
    }
    if which.eq_ignore_ascii_case("EvalFile") {
        if let Ok(mut guard) = PENDING_PATH.write() {
            *guard = if path.is_empty() || path == "<empty>" {
                None
            } else {
                Some(path.to_string())
            };
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
        let n = acc.len() / 16;
        for i in 0..n {
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
        let n = acc.len() / 16;
        for i in 0..n {
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
    let use_avx2 = is_x86_feature_detected!("avx2") && l1.is_multiple_of(16);
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
    let nets = loaded_nets();
    if let Some(nets) = nets.as_ref() {
        refresh_perspective(pos, nets, Side::White, accs);
        refresh_perspective(pos, nets, Side::Black, accs);
        // Threat PSQT is recomputed per eval; keep stored part zeroed.
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

// Queued HalfKA deltas applied at flush time. Threats/pairs recompute per eval.
pub fn apply_queued(
    pos: &SfnnPosition,
    accs: &mut Sfnn16Accs,
    adds: &[Option<SfnnEvent>],
    dels: &[Option<SfnnEvent>],
    king_moved: &[bool; 2],
) {
    let nets = loaded_nets();
    if let Some(nets) = nets.as_ref() {
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
unsafe fn add_threat_w_avx2(buf: &mut [i32; L1], w: &[i8]) {
    unsafe {
        use std::arch::x86_64::*;
        let b_ptr = buf.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m128i;
        for i in 0..64 {
            let w16 = _mm_loadu_si128(w_ptr.add(i));
            let w_low = _mm256_cvtepi8_epi32(w16);
            let w_high = _mm256_cvtepi8_epi32(_mm_srli_si128(w16, 8));

            let b_idx = i * 2;
            let b0 = _mm256_loadu_si256(b_ptr.add(b_idx));
            let b1 = _mm256_loadu_si256(b_ptr.add(b_idx + 1));

            _mm256_storeu_si256(b_ptr.add(b_idx), _mm256_add_epi32(b0, w_low));
            _mm256_storeu_si256(b_ptr.add(b_idx + 1), _mm256_add_epi32(b1, w_high));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_psqt_avx2(dst: &mut [i32; N_BUCKETS], src: &[i32]) {
    unsafe {
        use std::arch::x86_64::*;
        let d = _mm256_loadu_si256(dst.as_ptr() as *const __m256i);
        let s = _mm256_loadu_si256(src.as_ptr() as *const __m256i);
        _mm256_storeu_si256(dst.as_mut_ptr() as *mut __m256i, _mm256_add_epi32(d, s));
    }
}

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
    // Stockfish FtMaxVal (nnue_common.h): always 255, with threats or not.
    let clip_max: i32 = 255;
    let perspectives = [stm, stm.other()];

    let mut threat_lists = [[0usize; MAX_THREAT_ACTIVE]; 2];
    let mut threat_lens = [0usize; 2];
    let mut pair_lists = [[0usize; MAX_PAIR_ACTIVE]; 2];
    let mut pair_lens = [0usize; 2];

    if net.use_threats {
        for (slot, &persp) in perspectives.iter().enumerate() {
            threat_lens[slot] = collect_threats(pos, persp, &mut threat_lists[slot]);
            if !net.transformer.pair_w.is_empty() || !net.transformer.pair_w_i16.is_empty() {
                pair_lens[slot] = collect_pairs(pos, persp, &mut pair_lists[slot]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = is_x86_feature_detected!("avx2") && l1 == 1024 && !net.is_rudi;
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    let mut feats = [0u8; L1];
    let mut per_psqt = [[0i32; N_BUCKETS]; 2];
    for (slot, &persp) in perspectives.iter().enumerate() {
        let pi = persp as usize;
        let base_acc = halfka_accs[pi];
        let mut threat_buf = [0i32; L1];
        if net.use_threats {
            if net.is_rudi {
                #[cfg(target_arch = "x86_64")]
                let use_rudi_avx2 = is_x86_feature_detected!("avx2") && l1 == L1;
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
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            add_threat_w_avx2(&mut threat_buf, w_slice);
                        }
                    } else {
                        for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                            *acc += i32::from(w);
                        }
                    }
                    let base_psqt = t * N_BUCKETS;
                    let p_slice = &net.transformer.threat_psqt_w[base_psqt..base_psqt + N_BUCKETS];
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            add_psqt_avx2(&mut per_psqt[slot], p_slice);
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    } else {
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    }
                }
                for &f in &pair_lists[slot][..pair_lens[slot]] {
                    let t = f - PSQ_DIMS - THREAT_DIMS;
                    let base = t * l1;
                    let w_slice = &net.transformer.pair_w[base..base + l1];
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            add_threat_w_avx2(&mut threat_buf, w_slice);
                        }
                    } else {
                        for (acc, &w) in threat_buf.iter_mut().zip(w_slice) {
                            *acc += i32::from(w);
                        }
                    }
                    let base_psqt = t * N_BUCKETS;
                    let p_slice = &net.transformer.pair_psqt_w[base_psqt..base_psqt + N_BUCKETS];
                    if use_avx2 {
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            add_psqt_avx2(&mut per_psqt[slot], p_slice);
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    } else {
                        for b in 0..N_BUCKETS {
                            per_psqt[slot][b] = per_psqt[slot][b].wrapping_add(p_slice[b]);
                        }
                    }
                }
            }
        }
        let dst = &mut feats[slot * half..(slot + 1) * half];
        if net.is_rudi {
            #[cfg(target_arch = "x86_64")]
            let use_rudi_avx2 = is_x86_feature_detected!("avx2") && half.is_multiple_of(16);
            #[cfg(not(target_arch = "x86_64"))]
            let use_rudi_avx2 = false;

            if use_rudi_avx2 {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    pairwise_transform_rudi_avx2(base_acc, &threat_buf, dst, half, net.use_threats);
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
        } else if use_avx2 && half == 512 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if net.use_threats {
                    pairwise_transform_threats_avx2(base_acc, &threat_buf, dst, half, clip_max);
                } else {
                    pairwise_transform_no_threats_avx2(base_acc, dst, half, clip_max);
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
                dst[j] = if net.is_rudi {
                    ((c0 * c1) / 255) as u8
                } else {
                    ((c0 * c1) / 512) as u8
                };
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
    let nets = loaded_nets()?;
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

pub fn evaluate_board(board: &mut BoardState) -> Option<i16> {
    if !maintenance_active() {
        return None;
    }
    let pos = SfnnPosition::from_board(board);
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];
    ensure_fresh(&pos, accs);
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
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];
    ensure_fresh(&pos, accs);
    evaluate_nets(&pos, accs, board.side_to_move)
}

// ---------------------------------------------------------------------------
// Board integration helpers (wired from board/nnue.rs).
// ---------------------------------------------------------------------------

/// Queued v10 feature event in SF numbering: (sq_sf, side, piece).
pub type SfnnEvent = (usize, Side, Piece);

#[derive(Clone, Debug, Default)]
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
}
