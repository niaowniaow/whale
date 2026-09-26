use super::*;

pub const VERSION: u8 = 16;
pub const N_BUCKETS: usize = 8;

pub const PSQ_DIMS: usize = 22_528;
pub const THREAT_DIMS: usize = 59_808;
pub const PAIR_DIMS: usize = 4_560;
pub const BIG_INPUT_DIMS: usize = PSQ_DIMS + THREAT_DIMS + PAIR_DIMS;
pub const PS_PLANES: usize = 704;
pub const KING_BUCKET_COUNT: usize = 32;

pub const L1: usize = 1024;

pub const FC0_OUT: usize = 32;
pub const FC0_ACT: usize = 32;
pub const FC1_IN: usize = 64;
pub const FC1_OUT: usize = 32;

pub const OUTPUT_SCALE: i32 = 16;
pub const WEIGHT_SCALE_BITS: u32 = 6;
pub const SF_FILE_VERSION: u32 = 0x7AF32F20;
pub const SF17_FILE_VERSION: u32 = 0x6A448AFA;
pub const PSQ_HASH: u32 = 0x7f234cb8;
pub const THREAT_HASH: u32 = 0x2e6b9d04;
pub const PAIR_HASH: u32 = 0x86f2b1dd;
pub(super) const LEB128_MAGIC: &[u8] = b"COMPRESSED_LEB128";

pub const PAWN_VALUE: i32 = 208;
pub const KNIGHT_VALUE: i32 = 781;
pub const BISHOP_VALUE: i32 = 825;
pub const ROOK_VALUE: i32 = 1276;
pub const QUEEN_VALUE: i32 = 2538;
pub const SMALL_GATE: i32 = 962;
pub const SMALL_FALLBACK: i32 = 236;

pub const MAX_THREAT_ACTIVE: usize = 256;
pub const MAX_PAIR_ACTIVE: usize = 256;
pub const MAX_ACTIVE: usize = 32 + MAX_THREAT_ACTIVE + MAX_PAIR_ACTIVE;
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

pub(super) fn relu_hash(prev: u32) -> u32 {
    0x538D24C7u32.wrapping_add(prev)
}

pub(super) fn rotl1(x: u32) -> u32 {
    x.rotate_left(1)
}

pub(super) fn combine_hash(hashes: &[u32]) -> u32 {
    let mut h = 0u32;
    for &c in hashes {
        h = rotl1(h) ^ c;
    }
    h
}

pub fn arch_hash(l1: u32) -> u32 {
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
#[cfg(test)]
pub(super) fn unscramble_table(out_dims: usize, padded_in: usize) -> Vec<usize> {
    let n = out_dims * padded_in;
    let mut inv = vec![0usize; n];
    for i in 0..n {
        let m = (i / 4) % (padded_in / 4) * out_dims * 4 + (i / padded_in) * 4 + i % 4;
        inv[m] = i;
    }
    inv
}
#[inline(always)]
pub(super) fn div_trunc(a: i32, b: i32) -> i32 {
    a / b
}
