pub mod attacks;
pub mod cache;
pub mod core;
pub mod legality;
pub mod tables;
#[cfg(test)]
mod tests;

pub use tables::{BETWEEN_BB, CASTLING_CONSTANTS, LINE_BB, PASSED_PAWN_MASKS, RAY_TABLES};

use crate::bitboard::Bitboard;
use crate::board::history::History;
use crate::common::castle::Castle;
use crate::common::constants::SQUARES;
use crate::common::piece::{Piece, PieceMap};
use crate::common::side::{Side, SideMap};
use crate::common::square::Square;
use crate::eval::nnue::v16::SfnnPending;

#[derive(Debug, Clone)]
pub struct BoardState {
    pub pieces: PieceMap<Bitboard>,
    pub occupancies: SideMap<Bitboard>,
    pub piece_mapping: [Piece; SQUARES],
    pub side_to_move: Side,
    pub en_passant_square: Square,
    pub castle: Castle,
    pub move_count: i32,
    pub phase: i32,
    pub board_hash: u64,
    pub half_move_clock: u8,
    pub history: History,
    pub pending_adds_w: [usize; 2],
    pub pending_dels_w: [usize; 2],
    pub pending_adds_b: [usize; 2],
    pub pending_dels_b: [usize; 2],
    pub pending_adds: u8,
    pub pending_removes: u8,
    pub sfnn16_pending: SfnnPending,
}

impl Default for BoardState {
    fn default() -> Self {
        Self::starting_position()
    }
}

impl PartialEq for BoardState {
    fn eq(&self, other: &Self) -> bool {
        self.pieces == other.pieces
            && self.occupancies == other.occupancies
            && self.side_to_move == other.side_to_move
            && self.en_passant_square == other.en_passant_square
            && self.castle == other.castle
    }
}

impl Eq for BoardState {}
