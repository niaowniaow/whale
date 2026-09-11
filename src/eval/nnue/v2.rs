use crate::board::state::BoardState;
use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{get_bishop_attacks_from_table, get_queen_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks, pawn_attacks};
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

pub const VERSION: u8 = 2;
pub const KING_BUCKETS: usize = 32;
pub const PIECE_FEATURES_PER_BUCKET: usize = 768;
pub const PIECE_FEATURES: usize = KING_BUCKETS * PIECE_FEATURES_PER_BUCKET;
pub const THREAT_FEATURES: usize = 59_808;
pub const PAIR_FEATURES: usize = 4_560;
pub const RAW_INPUT_SIZE: usize = PIECE_FEATURES + THREAT_FEATURES + PAIR_FEATURES;
pub const TRANSFORMED_SIZE: usize = 1_024;
pub const ACC_SIZE: usize = 256;
pub const HIDDEN_SIZE: usize = 32;
pub const OUTPUT_SIZE: usize = 1;
pub const MAX_ACTIVE_FEATURES: usize = 640;
pub const FILE_MAGIC: [u8; 8] = *b"RUDIMV2\0";
pub const LAYER0_INPUTS: usize = TRANSFORMED_SIZE * 2;
pub const LAYER1_INPUTS: usize = HIDDEN_SIZE * 2;
pub const OUTPUT_INPUTS: usize = HIDDEN_SIZE * 2;
pub const HEADER_SIZE: usize = 24;

pub struct RawBoard {
    pub pieces: [u64; 6],
    pub white: u64,
    pub black: u64,
    pub mapping: [Piece; 64],
}

impl RawBoard {
    pub fn empty() -> Self {
        Self {
            pieces: [0; 6],
            white: 0,
            black: 0,
            mapping: [Piece::None; 64],
        }
    }

    pub fn from_board(board: &BoardState) -> Self {
        let mut pieces = [0u64; 6];
        for &p in &Piece::ALL {
            pieces[p as usize] = board.pieces[p].0;
        }
        Self {
            pieces,
            white: board.occupancies[Side::White].0,
            black: board.occupancies[Side::Black].0,
            mapping: board.piece_mapping,
        }
    }

    pub fn occupancy(&self) -> Bitboard {
        Bitboard(self.white | self.black)
    }

    pub fn get_pieces(&self, side: Side, piece: Piece) -> Bitboard {
        let occ = match side {
            Side::White => self.white,
            Side::Black => self.black,
            Side::Both => self.white | self.black,
        };
        Bitboard(self.pieces[piece as usize] & occ)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct NetworkHeader {
    pub magic: [u8; 8],
    pub version: u8,
    pub _reserved: [u8; 3],
    pub raw_input_size: u32,
    pub transformed_size: u16,
    pub accumulator_size: u16,
    pub hidden_size: u16,
    pub output_size: u16,
}

impl NetworkHeader {
    pub const fn current() -> Self {
        Self {
            magic: FILE_MAGIC,
            version: VERSION,
            _reserved: [0; 3],
            raw_input_size: RAW_INPUT_SIZE as u32,
            transformed_size: TRANSFORMED_SIZE as u16,
            accumulator_size: ACC_SIZE as u16,
            hidden_size: HIDDEN_SIZE as u16,
            output_size: OUTPUT_SIZE as u16,
        }
    }

    pub fn is_compatible(&self) -> bool {
        self.magic == FILE_MAGIC
            && self.version == VERSION
            && self.raw_input_size == RAW_INPUT_SIZE as u32
            && self.transformed_size == TRANSFORMED_SIZE as u16
            && self.accumulator_size == ACC_SIZE as u16
            && self.hidden_size == HIDDEN_SIZE as u16
            && self.output_size == OUTPUT_SIZE as u16
    }

    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut out = [0u8; HEADER_SIZE];
        out[0..8].copy_from_slice(&self.magic);
        out[8] = self.version;
        out[9..12].copy_from_slice(&self._reserved);
        out[12..16].copy_from_slice(&self.raw_input_size.to_le_bytes());
        out[16..18].copy_from_slice(&self.transformed_size.to_le_bytes());
        out[18..20].copy_from_slice(&self.accumulator_size.to_le_bytes());
        out[20..22].copy_from_slice(&self.hidden_size.to_le_bytes());
        out[22..24].copy_from_slice(&self.output_size.to_le_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < HEADER_SIZE {
            return None;
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        let version = bytes[8];
        let mut reserved = [0u8; 3];
        reserved.copy_from_slice(&bytes[9..12]);
        let raw_input_size = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        let transformed_size = u16::from_le_bytes([bytes[16], bytes[17]]);
        let accumulator_size = u16::from_le_bytes([bytes[18], bytes[19]]);
        let hidden_size = u16::from_le_bytes([bytes[20], bytes[21]]);
        let output_size = u16::from_le_bytes([bytes[22], bytes[23]]);
        Some(Self {
            magic,
            version,
            _reserved: reserved,
            raw_input_size,
            transformed_size,
            accumulator_size,
            hidden_size,
            output_size,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkShape {
    pub transformer_weights: usize,
    pub transformer_biases: usize,
    pub layer0_weights: usize,
    pub layer0_biases: usize,
    pub layer1_weights: usize,
    pub layer1_biases: usize,
    pub output_weights: usize,
    pub output_biases: usize,
}

impl NetworkShape {
    pub const fn current() -> Self {
        Self {
            transformer_weights: RAW_INPUT_SIZE * TRANSFORMED_SIZE,
            transformer_biases: TRANSFORMED_SIZE,
            layer0_weights: LAYER0_INPUTS * HIDDEN_SIZE,
            layer0_biases: HIDDEN_SIZE,
            layer1_weights: LAYER1_INPUTS * HIDDEN_SIZE,
            layer1_biases: HIDDEN_SIZE,
            output_weights: OUTPUT_INPUTS,
            output_biases: OUTPUT_SIZE,
        }
    }
}

pub struct NetworkWeights<'a> {
    pub transformer_weights: &'a [i16],
    pub transformer_biases: &'a [i16],
    pub layer0_weights: &'a [i8],
    pub layer0_biases: &'a [i32],
    pub layer1_weights: &'a [i8],
    pub layer1_biases: &'a [i32],
    pub output_weights: &'a [i8],
    pub output_bias: i32,
}

impl<'a> NetworkWeights<'a> {
    pub fn validate(&self) -> Result<(), &'static str> {
        let shape = NetworkShape::current();
        if self.transformer_weights.len() != shape.transformer_weights {
            return Err("invalid transformer weight count");
        }
        if self.transformer_biases.len() != shape.transformer_biases {
            return Err("invalid transformer bias count");
        }
        if self.layer0_weights.len() != shape.layer0_weights
            || self.layer0_biases.len() != shape.layer0_biases
        {
            return Err("invalid first hidden layer shape");
        }
        if self.layer1_weights.len() != shape.layer1_weights
            || self.layer1_biases.len() != shape.layer1_biases
        {
            return Err("invalid second hidden layer shape");
        }
        if self.output_weights.len() != shape.output_weights {
            return Err("invalid output layer shape");
        }
        Ok(())
    }
}

pub struct V2NetworkData {
    pub header: NetworkHeader,
    pub transformer_weights: Vec<i16>,
    pub transformer_biases: Vec<i16>,
    pub layer0_weights: Vec<i8>,
    pub layer0_biases: Vec<i32>,
    pub layer1_weights: Vec<i8>,
    pub layer1_biases: Vec<i32>,
    pub output_weights: Vec<i8>,
    pub output_bias: i32,
}

impl V2NetworkData {
    pub fn load_from_file(path: &str) -> Result<Self, &'static str> {
        let bytes = std::fs::read(path).map_err(|_| "failed to read v2 network file")?;
        Self::load_from_bytes(&bytes)
    }

    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < HEADER_SIZE {
            return Err("v2 network file too small");
        }
        let header = NetworkHeader::from_bytes(&bytes[0..HEADER_SIZE]).ok_or("invalid v2 header")?;
        if !header.is_compatible() {
            return Err("incompatible v2 network");
        }
        let shape = NetworkShape::current();
        let mut offset = HEADER_SIZE;
        let need = |count: usize, size: usize| count.checked_mul(size).ok_or("invalid v2 shape");
        let take = |bytes: &[u8], offset: &mut usize, count: usize, size: usize| -> Result<(), &'static str> {
            let n = need(count, size)?;
            if bytes.len() < *offset + n {
                return Err("truncated v2 network file");
            }
            *offset += n;
            Ok(())
        };
        take(bytes, &mut offset, shape.transformer_weights, 2)?;
        take(bytes, &mut offset, shape.transformer_biases, 2)?;
        take(bytes, &mut offset, shape.layer0_weights, 1)?;
        take(bytes, &mut offset, shape.layer0_biases, 4)?;
        take(bytes, &mut offset, shape.layer1_weights, 1)?;
        take(bytes, &mut offset, shape.layer1_biases, 4)?;
        take(bytes, &mut offset, shape.output_weights, 1)?;
        take(bytes, &mut offset, shape.output_biases, 4)?;
        let mut pos = HEADER_SIZE;
        let mut transformer_weights = Vec::with_capacity(shape.transformer_weights);
        for i in 0..shape.transformer_weights {
            let lo = bytes[pos + i * 2];
            let hi = bytes[pos + i * 2 + 1];
            transformer_weights.push(i16::from_le_bytes([lo, hi]));
        }
        pos += shape.transformer_weights * 2;
        let mut transformer_biases = Vec::with_capacity(shape.transformer_biases);
        for i in 0..shape.transformer_biases {
            let lo = bytes[pos + i * 2];
            let hi = bytes[pos + i * 2 + 1];
            transformer_biases.push(i16::from_le_bytes([lo, hi]));
        }
        pos += shape.transformer_biases * 2;
        let mut layer0_weights = Vec::with_capacity(shape.layer0_weights);
        for i in 0..shape.layer0_weights {
            layer0_weights.push(bytes[pos + i] as i8);
        }
        pos += shape.layer0_weights;
        let mut layer0_biases = Vec::with_capacity(shape.layer0_biases);
        for i in 0..shape.layer0_biases {
            let b = [bytes[pos + i * 4], bytes[pos + i * 4 + 1], bytes[pos + i * 4 + 2], bytes[pos + i * 4 + 3]];
            layer0_biases.push(i32::from_le_bytes(b));
        }
        pos += shape.layer0_biases * 4;
        let mut layer1_weights = Vec::with_capacity(shape.layer1_weights);
        for i in 0..shape.layer1_weights {
            layer1_weights.push(bytes[pos + i] as i8);
        }
        pos += shape.layer1_weights;
        let mut layer1_biases = Vec::with_capacity(shape.layer1_biases);
        for i in 0..shape.layer1_biases {
            let b = [bytes[pos + i * 4], bytes[pos + i * 4 + 1], bytes[pos + i * 4 + 2], bytes[pos + i * 4 + 3]];
            layer1_biases.push(i32::from_le_bytes(b));
        }
        pos += shape.layer1_biases * 4;
        let mut output_weights = Vec::with_capacity(shape.output_weights);
        for i in 0..shape.output_weights {
            output_weights.push(bytes[pos + i] as i8);
        }
        pos += shape.output_weights;
        let b = [bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]];
        let output_bias = i32::from_le_bytes(b);
        Ok(Self {
            header,
            transformer_weights,
            transformer_biases,
            layer0_weights,
            layer0_biases,
            layer1_weights,
            layer1_biases,
            output_weights,
            output_bias,
        })
    }

    pub fn as_weights(&self) -> NetworkWeights<'_> {
        NetworkWeights {
            transformer_weights: &self.transformer_weights,
            transformer_biases: &self.transformer_biases,
            layer0_weights: &self.layer0_weights,
            layer0_biases: &self.layer0_biases,
            layer1_weights: &self.layer1_weights,
            layer1_biases: &self.layer1_biases,
            output_weights: &self.output_weights,
            output_bias: self.output_bias,
        }
    }
}

#[inline(always)]
fn screlu(value: i32) -> i32 {
    let clipped = value.clamp(0, 255);
    clipped * clipped
}

#[inline(always)]
fn crelu(value: i32) -> i32 {
    value.clamp(0, 127)
}

pub fn evaluate_v2(board: &BoardState, weights: &NetworkWeights<'_>) -> Result<i16, &'static str> {
    weights.validate()?;
    let stm = board.side_to_move;
    let ntm = stm.other();
    let stm_features = active_features(board, stm);
    let ntm_features = active_features(board, ntm);
    let mut transformed = [[0i32; TRANSFORMED_SIZE]; 2];

    for (perspective, features) in [stm_features, ntm_features].iter().enumerate() {
        for index in 0..TRANSFORMED_SIZE {
            transformed[perspective][index] = i32::from(weights.transformer_biases[index]);
        }
        for &feature in &features.indices[..features.len] {
            let row = &weights.transformer_weights[feature * TRANSFORMED_SIZE
                ..(feature + 1) * TRANSFORMED_SIZE];
            for (value, weight) in transformed[perspective].iter_mut().zip(row) {
                *value += i32::from(*weight);
            }
        }
    }

    let mut hidden0 = [0i32; HIDDEN_SIZE];
    for output in 0..HIDDEN_SIZE {
        let mut value = weights.layer0_biases[output];
        for input in 0..LAYER0_INPUTS {
            let source = transformed[input / TRANSFORMED_SIZE][input % TRANSFORMED_SIZE];
            value += screlu(source) * i32::from(weights.layer0_weights[input * HIDDEN_SIZE + output]);
        }
        hidden0[output] = crelu(value / 256);
    }

    let mut hidden1 = [0i32; HIDDEN_SIZE];
    for output in 0..HIDDEN_SIZE {
        let mut value = weights.layer1_biases[output];
        for input in 0..LAYER1_INPUTS {
            let source = if input < HIDDEN_SIZE {
                hidden0[input]
            } else {
                hidden0[input - HIDDEN_SIZE]
            };
            value += source * i32::from(weights.layer1_weights[input * HIDDEN_SIZE + output]);
        }
        hidden1[output] = crelu(value / 128);
    }

    let mut output = weights.output_bias;
    for input in 0..OUTPUT_INPUTS {
        output += hidden1[input % HIDDEN_SIZE]
            * i32::from(weights.output_weights[input]);
    }
    Ok(output.clamp(-29_000, 29_000) as i16)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchitectureManifest {
    pub version: u8,
    pub piece_features: usize,
    pub threat_features: usize,
    pub pair_features: usize,
    pub raw_input_size: usize,
    pub input_size: usize,
    pub accumulator_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl ArchitectureManifest {
    pub const fn current() -> Self {
        Self {
            version: VERSION,
            piece_features: PIECE_FEATURES,
            threat_features: THREAT_FEATURES,
            pair_features: PAIR_FEATURES,
            raw_input_size: RAW_INPUT_SIZE,
            input_size: TRANSFORMED_SIZE,
            accumulator_size: ACC_SIZE,
            hidden_size: HIDDEN_SIZE,
            output_size: OUTPUT_SIZE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseFeatureSet {
    pub indices: [usize; MAX_ACTIVE_FEATURES],
    pub len: usize,
}

impl SparseFeatureSet {
    pub const fn empty() -> Self {
        Self {
            indices: [0; MAX_ACTIVE_FEATURES],
            len: 0,
        }
    }

    fn push(&mut self, index: usize) {
        if self.len < self.indices.len() {
            self.indices[self.len] = index;
            self.len += 1;
        }
    }
}

pub fn piece_feature_index(piece: Piece, relative_side: Side, square: Square) -> Option<usize> {
    if piece == Piece::None {
        return None;
    }
    piece_feature_index_with_king(piece, relative_side, square, 0)
}

pub fn piece_feature_index_with_king(
    piece: Piece,
    relative_side: Side,
    square: Square,
    king_bucket: usize,
) -> Option<usize> {
    if piece == Piece::None || king_bucket >= KING_BUCKETS {
        return None;
    }
    let side_offset = relative_side as usize * 384;
    let piece_offset = piece as usize * 64;
    Some(
        king_bucket * PIECE_FEATURES_PER_BUCKET
            + side_offset
            + piece_offset
            + ((square as usize) ^ 56),
    )
}

pub fn threat_feature_index(channel: usize, square: Square) -> Option<usize> {
    (channel < 4).then_some(PIECE_FEATURES + channel * 64 + ((square as usize) ^ 56))
}

pub fn relational_threat_index(
    perspective: Side,
    attacker: Piece,
    from: Square,
    to: Square,
    victim: Piece,
    king_square: Square,
) -> usize {
    let orientation = if king_square as usize % 8 < 4 { 7 } else { 0 };
    let orient = |square: Square| {
        let value = (square as usize) ^ if perspective == Side::Black { 56 } else { 0 };
        value ^ orientation
    };
    let mut hash = 0x9e37_79b9usize;
    for value in [
        perspective as usize,
        attacker as usize,
        orient(from),
        orient(to),
        victim as usize,
        orient(king_square),
    ] {
        hash ^= value.wrapping_add(0x9e37_79b9).wrapping_add(hash << 6).wrapping_add(hash >> 2);
    }
    PIECE_FEATURES + (hash % THREAT_FEATURES)
}

pub fn pawn_pair_index(first: (Side, Square), second: (Side, Square)) -> Option<usize> {
    let first_id = pawn_id(first.0, first.1)?;
    let second_id = pawn_id(second.0, second.1)?;
    if first_id == second_id {
        return None;
    }
    let (low, high) = if first_id < second_id {
        (first_id, second_id)
    } else {
        (second_id, first_id)
    };
    let rank = low * (191 - low) / 2 + (high - low - 1);
    Some(PIECE_FEATURES + THREAT_FEATURES + rank)
}

fn pawn_id(side: Side, square: Square) -> Option<usize> {
    let rank = square as usize / 8;
    if rank == 0 || rank == 7 {
        return None;
    }
    Some(side as usize * 48 + (rank - 1) * 8 + square as usize % 8)
}

pub fn king_bucket_raw(board: &RawBoard, perspective: Side) -> usize {
    let king = board.get_pieces(perspective, Piece::King);
    if king.is_empty() {
        return 0;
    }
    let square = king.get_lsb() as usize;
    let oriented = if perspective == Side::White {
        square
    } else {
        square ^ 56
    };
    let rank = oriented / 8;
    let file = (oriented % 8).min(7 - (oriented % 8));
    rank * 4 + file
}

pub fn king_square_raw(board: &RawBoard, perspective: Side) -> Square {
    Square::from(board.get_pieces(perspective, Piece::King).get_lsb() as usize)
}

pub fn append_relational_threats_raw(
    board: &RawBoard,
    perspective: Side,
    features: &mut SparseFeatureSet,
) {
    let king = king_square_raw(board, perspective);
    let occupancy = board.occupancy();
    for &side in &[Side::White, Side::Black] {
        for &attacker in &Piece::ALL {
            let mut pieces = board.get_pieces(side, attacker);
            while pieces.is_not_empty() {
                let from = Square::from(pieces.get_lsb() as usize);
                let attacks = match attacker {
                    Piece::Pawn => Bitboard(pawn_attacks()[side as usize][from as usize]),
                    Piece::Knight => Bitboard(knight_attacks()[from as usize]),
                    Piece::Bishop => get_bishop_attacks_from_table(from, occupancy),
                    Piece::Rook => get_rook_attacks_from_table(from, occupancy),
                    Piece::Queen => get_queen_attacks_from_table(from, occupancy),
                    Piece::King => Bitboard(king_attacks()[from as usize]),
                    Piece::None => Bitboard::EMPTY,
                };
                let opp_occ = match side {
                    Side::White => Bitboard(board.black),
                    Side::Black => Bitboard(board.white),
                    Side::Both => Bitboard::EMPTY,
                };
                let mut targets = attacks & opp_occ;
                while targets.is_not_empty() {
                    let to = Square::from(targets.get_lsb() as usize);
                    let victim = board.mapping[to as usize];
                    features.push(relational_threat_index(perspective, attacker, from, to, victim, king));
                    targets.clear_lsb();
                }
                pieces.clear_lsb();
            }
        }
    }
}

pub fn active_features_raw(board: &RawBoard, perspective: Side) -> SparseFeatureSet {
    let mut features = SparseFeatureSet::empty();
    let bucket = king_bucket_raw(board, perspective);
    for &side in &[Side::White, Side::Black] {
        let relative_side = if side == perspective {
            Side::White
        } else {
            Side::Black
        };
        for &piece in &Piece::ALL {
            let mut pieces = board.get_pieces(side, piece);
            while pieces.is_not_empty() {
                let square = Square::from(pieces.get_lsb() as usize);
                if let Some(index) = piece_feature_index_with_king(
                    piece,
                    relative_side,
                    square,
                    bucket,
                ) {
                    features.push(index);
                }
                pieces.clear_lsb();
            }
        }
    }
    append_relational_threats_raw(board, perspective, &mut features);
    let mut pawns = Vec::new();
    for &side in &[Side::White, Side::Black] {
        let mut pieces = board.get_pieces(side, Piece::Pawn);
        while pieces.is_not_empty() {
            pawns.push((side, Square::from(pieces.get_lsb() as usize)));
            pieces.clear_lsb();
        }
    }
    for (index, &first) in pawns.iter().enumerate() {
        for &second in pawns.iter().skip(index + 1) {
            if let Some(feature) = pawn_pair_index(first, second) {
                features.push(feature);
            }
        }
    }
    features
}

pub fn active_feature_diff(
    before: &SparseFeatureSet,
    after: &SparseFeatureSet,
) -> (SparseFeatureSet, SparseFeatureSet) {
    let mut removed = SparseFeatureSet::empty();
    let mut added = SparseFeatureSet::empty();
    for &index in &before.indices[..before.len] {
        if !after.indices[..after.len].contains(&index) {
            removed.push(index);
        }
    }
    for &index in &after.indices[..after.len] {
        if !before.indices[..before.len].contains(&index) {
            added.push(index);
        }
    }
    (removed, added)
}

pub fn active_features(board: &BoardState, perspective: Side) -> SparseFeatureSet {
    active_features_raw(&RawBoard::from_board(board), perspective)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn manifest_describes_v2_layout() {
        let manifest = ArchitectureManifest::current();
        assert_eq!(manifest.version, 2);
        assert_eq!(manifest.input_size, 1024);
        assert_eq!(manifest.raw_input_size, 88_944);
        assert_eq!(manifest.accumulator_size, 256);
        assert_eq!(manifest.hidden_size, 32);
        assert!(NetworkHeader::current().is_compatible());
    }

    #[test]
    fn v2_piece_and_threat_features_do_not_overlap() {
        let piece = piece_feature_index_with_king(Piece::King, Side::Black, Square::H8, 31).unwrap();
        let threat = threat_feature_index(0, Square::A1).unwrap();
        assert!(piece < PIECE_FEATURES);
        assert!(threat >= PIECE_FEATURES);
    }

    #[test]
    fn king_bucket_changes_piece_feature_slice() {
        let low = piece_feature_index_with_king(Piece::Pawn, Side::White, Square::E4, 0);
        let high = piece_feature_index_with_king(Piece::Pawn, Side::White, Square::E4, 1);
        assert_ne!(low, high);
        assert_eq!(high.unwrap() - low.unwrap(), PIECE_FEATURES_PER_BUCKET);
    }

    #[test]
    fn active_features_include_start_position_pieces_and_threats() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let features = active_features(&board, Side::White);
        assert!(features.len >= 32);
        assert!(features.indices[..features.len]
            .iter()
            .any(|&index| index >= PIECE_FEATURES + THREAT_FEATURES));
    }

    #[test]
    fn pawn_pair_indices_cover_the_declared_space() {
        let first = pawn_pair_index((Side::White, Square::A7), (Side::Black, Square::H2));
        let last = pawn_pair_index((Side::White, Square::H2), (Side::Black, Square::H7));
        assert!(first.is_some());
        assert!(last.is_some());
        assert!(first.unwrap() >= PIECE_FEATURES + THREAT_FEATURES);
        assert!(last.unwrap() < RAW_INPUT_SIZE);
    }

    #[test]
    fn feature_diff_reports_added_and_removed_indices() {
        let mut before = SparseFeatureSet::empty();
        before.push(10);
        before.push(20);
        let mut after = SparseFeatureSet::empty();
        after.push(20);
        after.push(30);

        let (removed, added) = active_feature_diff(&before, &after);
        assert_eq!(&removed.indices[..removed.len], &[10]);
        assert_eq!(&added.indices[..added.len], &[30]);
    }

    #[test]
    fn v2_runtime_rejects_incomplete_weights() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let weights = NetworkWeights {
            transformer_weights: &[],
            transformer_biases: &[],
            layer0_weights: &[],
            layer0_biases: &[],
            layer1_weights: &[],
            layer1_biases: &[],
            output_weights: &[],
            output_bias: 0,
        };
        assert_eq!(evaluate_v2(&board, &weights), Err("invalid transformer weight count"));
    }
}