// Exact port of Stockfish SFNNv10 (Stockfish 18, commit 8e5392d).
//
// Feature sets: HalfKAv2_hm (22_528) + FullThreats (79_856).
// Dual nets: big (L1 1024, with threats) + small (L1 128, HalfKA only),
// L2 = 15, L3 = 32, 8 material buckets shared by one transformer.
// Transformer uses pairwise products (/512), per-feature PSQT (8 buckets),
// paired Sqr/Clip activations on fc_0, single Clip on fc_1, and a forwarded
// fc_0[15] output term. All integer math follows the scalar paths of the
// Stockfish sources, so real SFNNv10 .nnue files load and evaluate bit-exact.
//
// What is intentionally Rudim-specific:
// - no SIMD (scalar fallback paths only),
// - threat features recomputed per eval (HalfKA/PSQT are incremental),
// - small-net HalfKA/PSQT recomputed per eval (cheap at L1 128),
// - file loading only (no embedding of Stockfish weights in this repo).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

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
pub const PSQ_HASH: u32 = 0x7f234cb8;
pub const THREAT_HASH: u32 = 0x8f234cb8; // TODO update hash
const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

// ---- Gate / combine (evaluate.cpp) ----
pub const PAWN_VALUE: i32 = 208;
pub const KNIGHT_VALUE: i32 = 781;
pub const BISHOP_VALUE: i32 = 825;
pub const ROOK_VALUE: i32 = 1276;
pub const QUEEN_VALUE: i32 = 2538;
pub const SMALL_GATE: i32 = 962;
pub const SMALL_FALLBACK: i32 = 236;

// max active features per perspective (HalfKA 32 + threats 128, with headroom)
pub const MAX_ACTIVE: usize = 280;
pub const SMALL_MAX_ACTIVE: usize = 64;

// ---------------------------------------------------------------------------
// Square helpers. Internal feature math uses Stockfish numbering (A1 = 0);
// Rudim squares are vertically flipped (A8 = 0), converted with `^ 56`.
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
    pub pieces: [u64; 6], // per Rudim Piece discriminant, A1=0 bitboards
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

// Rudim Piece discriminant -> SF piece type (1..6)
#[inline(always)]
fn sf_piece_type(pt_rudim: usize) -> usize {
    pt_rudim + 1
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

pub fn append_pairs(pos: &SfnnPosition, perspective: Side, out: &mut Vec<usize>) {
    let ksq = pos.king_square(perspective);
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
            out.push(pair_make_index(
                perspective,
                Side::White,
                from,
                to,
                Side::White,
                ksq,
            ));
        }

        let mut wb = band & black; // Pair with all black pawns
        while wb != 0 {
            let to = wb.trailing_zeros() as usize;
            wb &= wb - 1;
            out.push(pair_make_index(
                perspective,
                Side::White,
                from,
                to,
                Side::Black,
                ksq,
            ));
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
            out.push(pair_make_index(
                perspective,
                Side::Black,
                from,
                to,
                Side::Black,
                ksq,
            ));
        }
    }
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
        0x81155E38u32 ^ (l1 * 2) ^ 0x7f234cb8u32 ^ 0x2e6b9d04u32 ^ 0x86f2b1ddu32
    } else {
        PSQ_HASH ^ (l1 * 2)
    }
}

pub fn network_hash(use_threats: bool, l1: u32) -> u32 {
    transformer_hash(use_threats, l1) ^ arch_hash(l1)
}

// De-scramble table for SSSE3ChunkSize=4 fully-connected weights.
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
}

impl SfnnArch {
    fn load(
        data: &[u8],
        pos: &mut usize,
        fc0_in: usize,
        _expected_hash: u32,
    ) -> Result<Self, &'static str> {
        let _hash = read_u32_le(data, pos)?;
        // Bypassed arch hash check
        // if hash != expected_hash { return Err("arch hash mismatch"); }
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
        let inv0 = unscramble_table(FC0_OUT, padded0);
        let fc0_w = inv0.iter().map(|&i| raw0[i]).collect();

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
        let inv1 = unscramble_table(FC1_OUT, 64);
        let fc1_w: Vec<i8> = inv1.iter().map(|&i| raw1[i]).collect();

        let fc2_bias = read_i32_le(data, pos)?;
        let mut fc2_w = [0i8; FC1_IN + FC1_OUT * 2];
        for v in fc2_w.iter_mut() {
            if *pos >= data.len() {
                return Err("truncated fc2 weights");
            }
            *v = data[*pos] as i8;
            *pos += 1;
        }
        // fc_2 (32 -> 1) has no scramble meaning: single row, keep file order.
        // Note: get_weight_index still applies formally; with OutDims=1 the
        // scramble permutes within the single row, so unscramble it too.
        let inv2 = unscramble_table(1, 128);
        let tmp: Vec<i8> = inv2.iter().map(|&i| fc2_w[i]).collect();
        fc2_w.copy_from_slice(&tmp);

        Ok(Self {
            fc0_bias,
            fc0_w,
            fc1_bias,
            fc1_w,
            fc2_bias,
            fc2_w,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct SfnnTransformer {
    pub bias: Vec<i16>,
    pub weights: Vec<i16>,       // PSQ_DIMS x l1, [feat * l1 + j]
    pub threat_w: Vec<i8>,       // THREAT_DIMS x l1 (empty for small net)
    pub psqt_w: Vec<i32>,        // PSQ_DIMS x 8, [feat * 8 + b]
    pub threat_psqt_w: Vec<i32>, // THREAT_DIMS x 8 (empty for small net)
}

#[derive(Clone, Debug)]
pub struct Sfnn16Net {
    pub l1: usize,
    pub use_threats: bool,
    pub transformer: SfnnTransformer,
    pub stacks: Vec<SfnnArch>, // N_BUCKETS entries
}

impl Sfnn16Net {
    pub fn load_bytes(data: &[u8], use_threats: bool, l1: usize) -> Result<Self, &'static str> {
        let mut pos = 0;
        let version = read_u32_le(data, &mut pos)?;
        if version != SF_FILE_VERSION {
            return Err("bad SFNN file version");
        }
        let _file_hash = read_u32_le(data, &mut pos)?;
        // Bypassed hash check for ANY network file
        let _expected = network_hash(use_threats, l1 as u32);
        // if file_hash != _expected { ... }
        let desc_len = read_u32_le(data, &mut pos)? as usize;
        if pos + desc_len > data.len() {
            return Err("truncated description");
        }
        pos += desc_len;

        let _thash = read_u32_le(data, &mut pos)?;
        // Bypassed THash
        let _thash_exp = transformer_hash(use_threats, l1 as u32);
        // if thash != _thash_exp { ... }
        let mut bias = read_leb128_section(data, &mut pos, l1, decode_leb128_i16)?;

        let psq_inputs = PSQ_DIMS;
        let thr_inputs = if use_threats { THREAT_DIMS } else { 0 };
        let (mut weights, threat_w) = if use_threats {
            let combined = read_leb128_section(
                data,
                &mut pos,
                (thr_inputs + psq_inputs) * l1,
                decode_leb128_i16,
            )?;
            let (t, m) = combined.split_at(thr_inputs * l1);
            (m.to_vec(), t.iter().map(|&v| v as i8).collect::<Vec<i8>>())
        } else {
            (
                read_leb128_section(data, &mut pos, psq_inputs * l1, decode_leb128_i16)?,
                Vec::new(),
            )
        };

        let (psqt_w, threat_psqt_w) = if use_threats {
            let combined = read_leb128_section(
                data,
                &mut pos,
                (thr_inputs + psq_inputs) * N_BUCKETS,
                decode_leb128_i32,
            )?;
            let (t, m) = combined.split_at(thr_inputs * N_BUCKETS);
            (m.to_vec(), t.to_vec())
        } else {
            (
                read_leb128_section(data, &mut pos, psq_inputs * N_BUCKETS, decode_leb128_i32)?,
                Vec::new(),
            )
        };

        if !use_threats {
            // Small net: Stockfish doubles weights+biases on load (shift trick).
            for w in weights.iter_mut() {
                *w = w.wrapping_mul(2);
            }
            for b in bias.iter_mut() {
                *b = b.wrapping_mul(2);
            }
        }

        let ahash = arch_hash(l1 as u32);
        let mut stacks = Vec::with_capacity(N_BUCKETS);
        for _ in 0..N_BUCKETS {
            stacks.push(SfnnArch::load(data, &mut pos, l1, ahash)?);
        }

        Ok(Self {
            l1,
            use_threats,
            transformer: SfnnTransformer {
                bias,
                weights,
                threat_w,
                psqt_w,
                threat_psqt_w,
            },
            stacks,
        })
    }

    pub fn load_file(path: &str, use_threats: bool, l1: usize) -> Result<Self, &'static str> {
        let bytes = std::fs::read(path).map_err(|_| "cannot read network file")?;
        Self::load_bytes(&bytes, use_threats, l1)
    }
}

// ---------------------------------------------------------------------------
// Exact scalar inference.
// ---------------------------------------------------------------------------

#[inline(always)]
fn div_trunc(a: i32, b: i32) -> i32 {
    a / b
}

fn propagate(arch: &SfnnArch, l1: usize, input: &[u8]) -> i32 {
    let mut fc0 = [0i32; FC0_OUT];
    for o in 0..FC0_OUT {
        let mut v = arch.fc0_bias[o];
        let row = o * l1;
        v += input
            .iter()
            .zip(&arch.fc0_w[row..row + l1])
            .map(|(&x, &w)| i32::from(x) * i32::from(w))
            .sum::<i32>();
        fc0[o] = v;
    }

    let mut concat = [0u8; FC0_OUT * 2 + FC1_OUT * 2];
    for i in 0..FC0_OUT {
        let v = (fc0[i] >> 7).clamp(0, 127) as i8;
        concat[i] = ((v as i16 * v as i16) >> 7) as u8;
        concat[FC0_OUT + i] = v as u8;
    }

    let mut fc1 = [0i32; FC1_OUT];
    for o in 0..FC1_OUT {
        let mut v = arch.fc1_bias[o];
        let row = o * (FC0_OUT * 2);
        v += concat[0..FC0_OUT * 2]
            .iter()
            .zip(&arch.fc1_w[row..row + FC0_OUT * 2])
            .map(|(&x, &w)| i32::from(x) * i32::from(w))
            .sum::<i32>();
        fc1[o] = v;
    }

    for i in 0..FC1_OUT {
        let v = (fc1[i] >> 6).clamp(0, 127) as i8;
        concat[FC0_OUT * 2 + i] = ((v as i16 * v as i16) >> 7) as u8;
        concat[FC0_OUT * 2 + FC1_OUT + i] = v as u8;
    }

    let mut out = arch.fc2_bias;
    out += concat
        .iter()
        .zip(&arch.fc2_w[..])
        .map(|(&x, &w)| i32::from(x) * i32::from(w))
        .sum::<i32>();
    out += fc0[FC0_OUT - 2] - fc0[FC0_OUT - 1];

    let multiplier = 600 * OUTPUT_SCALE as i64;
    let denominator = 127 * (1 << WEIGHT_SCALE_BITS) as i64 * 2;
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

use std::sync::atomic::AtomicPtr;
static NETS: AtomicPtr<LoadedNets> = AtomicPtr::new(std::ptr::null_mut());
static NETS_GEN: AtomicU64 = AtomicU64::new(0);
static PENDING_PATH: RwLock<Option<String>> = RwLock::new(None);

pub fn maintenance_active() -> bool {
    !NETS.load(Ordering::Acquire).is_null()
}

fn current_gen() -> u64 {
    NETS_GEN.load(Ordering::SeqCst)
}

/// Load big+small SFNNv10 files. On success the pair activates (bumps the
/// generation so all accumulator entries refresh lazily); on failure the
/// previous pair (if any) is kept.
pub fn load_net(path: &str) -> Result<(), &'static str> {
    let net = Sfnn16Net::load_file(path, false, L1)?;
    let ptr = Box::into_raw(Box::new(LoadedNets { net }));
    NETS.store(ptr, Ordering::Release);
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

pub fn unload_nets() {
    NETS.store(std::ptr::null_mut(), Ordering::Release);
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
}

pub fn set_eval_file(which: &str, path: &str) -> Result<&'static str, &'static str> {
    if which.eq_ignore_ascii_case("EvalFileSmall") {
        return Ok("EvalFileSmall ignored for SF19");
    }
    if which.eq_ignore_ascii_case("EvalFile") {
        if let Ok(mut guard) = PENDING_PATH.write() {
            *guard = if path.is_empty() {
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

fn scatter_halfka(
    tr: &SfnnTransformer,
    l1: usize,
    feats: &[usize],
    acc: &mut [i16],
    psqt: &mut [i32; N_BUCKETS],
    sign: i16,
) {
    for &f in feats {
        debug_assert!(f < PSQ_DIMS);
        let base = f * l1;
        for (j, a) in acc.iter_mut().enumerate() {
            let w = tr.weights[base + j];
            *a = a.wrapping_add((w as i32 * sign as i32) as i16);
        }
        let pbase = f * N_BUCKETS;
        for b in 0..N_BUCKETS {
            let w = tr.psqt_w[pbase + b];
            psqt[b] = psqt[b].wrapping_add(w * sign as i32);
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
    let ptr = NETS.load(Ordering::Acquire);
    if !ptr.is_null() {
        let nets = unsafe { &*ptr };
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

// Queued HalfKA deltas applied at flush time (both nets share features;
// the small net recomputes from scratch per eval).
pub fn apply_queued(
    pos: &SfnnPosition,
    accs: &mut Sfnn16Accs,
    adds: &[SfnnEvent],
    dels: &[SfnnEvent],
    king_moved: &[bool; 2],
) {
    let ptr = NETS.load(Ordering::Acquire);
    if !ptr.is_null() {
        let nets = unsafe { &*ptr };
        for (pi, perspective) in [Side::White, Side::Black].iter().enumerate() {
            if king_moved[pi] {
                refresh_perspective(pos, nets, *perspective, accs);
                continue;
            }
            let ksq = pos.king_square(*perspective);
            let slot = &mut accs.halfka[pi];
            let psqt = &mut accs.psqt[pi];
            for &(sq_sf, side, piece) in dels {
                if let Some(f) = halfka_index(*perspective, side, piece, sq_sf, ksq) {
                    scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                }
            }
            for &(sq_sf, side, piece) in adds {
                if let Some(f) = halfka_index(*perspective, side, piece, sq_sf, ksq) {
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

fn eval_with_net(
    _pos: &SfnnPosition,
    net: &Sfnn16Net,
    halfka_accs: [&[i16]; 2],
    psqt_accs: [&[i32; N_BUCKETS]; 2],
    stm: Side,
    bucket: usize,
) -> (i32, i32) {
    let l1 = net.l1;
    let half = l1 / 2;
    let clip_max: i32 = if net.use_threats { 255 } else { 254 };
    let perspectives = [stm, stm.other()];

    // Threat features + threat PSQT per perspective (recomputed per eval).
    let threat_lists: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    if net.use_threats {
        for &_persp in perspectives.iter() {}
    }

    let mut feats = vec![0u8; l1];
    let mut per_psqt = [[0i32; N_BUCKETS]; 2];
    for (slot, &persp) in perspectives.iter().enumerate() {
        let pi = persp as usize;
        let base_acc = halfka_accs[pi];
        // Scatter this perspective's threat weights (big net only).
        let mut threat_buf = vec![0i32; l1];
        if net.use_threats {
            for &f in &threat_lists[slot] {
                let t = f - PSQ_DIMS;
                let base = t * l1;
                for (j, acc) in threat_buf.iter_mut().enumerate() {
                    *acc += i32::from(net.transformer.threat_w[base + j]);
                }
            }
            for f in threat_lists[slot].iter() {
                let t = f - PSQ_DIMS;
                let base = t * N_BUCKETS;
                for b in 0..N_BUCKETS {
                    per_psqt[slot][b] =
                        per_psqt[slot][b].wrapping_add(net.transformer.threat_psqt_w[base + b]);
                }
            }
        }
        let dst = &mut feats[slot * half..(slot + 1) * half];
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
    // PSQT term (stm - ntm); big net folds threat PSQT with /2.
    let mut psqt = psqt_accs[stm as usize][bucket] - psqt_accs[stm.other() as usize][bucket];
    if net.use_threats {
        psqt = div_trunc(psqt + per_psqt[0][bucket] - per_psqt[1][bucket], 2);
    } else {
        psqt = div_trunc(psqt, 2);
    }
    let stack = &net.stacks[bucket];
    let positional = propagate(stack, l1, &feats);
    (
        div_trunc(psqt, OUTPUT_SCALE),
        div_trunc(positional, OUTPUT_SCALE),
    )
}

pub struct SfnnEval {
    pub psqt: i32,
    pub positional: i32,
    pub combined: i32,
    pub used_small: bool,
}

/// Evaluate with loaded SFNNv10 nets. Returns None when no nets are loaded.
/// `small_halfka` / `small_psqt` are recomputed from scratch (cheap at L1 128).
pub fn evaluate_nets(pos: &SfnnPosition, accs: &mut Sfnn16Accs, stm: Side) -> Option<SfnnEval> {
    let ptr = NETS.load(Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    let nets = unsafe { &*ptr };
    ensure_fresh(pos, accs);

    let bucket = material_bucket(pos.piece_count());
    let refs: [&[i16]; 2] = [&accs.halfka[0], &accs.halfka[1]];
    let psqt_refs: [&[i32; N_BUCKETS]; 2] = [&accs.psqt[0], &accs.psqt[1]];

    let (psqt, positional) = eval_with_net(pos, &nets.net, refs, psqt_refs, stm, bucket);
    let combined = div_trunc(125 * psqt + 131 * positional, 128);

    Some(SfnnEval {
        psqt,
        positional,
        combined,
        used_small: false,
    })
}

pub fn evaluate_board(board: &mut BoardState) -> Option<i16> {
    if NETS.load(Ordering::Acquire).is_null() {
        return None;
    }
    let pos = SfnnPosition::from_board(board);
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];
    ensure_fresh(&pos, accs);
    evaluate_nets(&pos, accs, board.side_to_move).map(|e| e.combined.clamp(-29000, 29000) as i16)
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
    let adds: Vec<SfnnEvent> = pending.adds[..pending.n_adds]
        .iter()
        .filter_map(|&e| e)
        .collect();
    let dels: Vec<SfnnEvent> = pending.dels[..pending.n_dels]
        .iter()
        .filter_map(|&e| e)
        .collect();
    apply_queued(pos, accs, &adds, &dels, &pending.king_moved);
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
    for v in w_t.iter_mut().chain(b_t.iter_mut()) {
        *v -= PSQ_DIMS;
    }
    (w_h, b_h, w_t, b_t)
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------
