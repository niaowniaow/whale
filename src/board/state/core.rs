use super::BoardState;
use super::tables::PASSED_PAWN_MASKS;
use crate::bitboard::Bitboard;
use crate::board::history::History;
use crate::common::castle::Castle;
use crate::common::constants::{PIECES, SIDES, SQUARES};
use crate::common::game_phase::{add_phase, get_clipped_phase, remove_phase};
use crate::common::piece::{Piece, PieceMap};
use crate::common::side::{Side, SideMap};
use crate::common::square::Square;
use crate::eval::nnue::loader::Network;
use crate::eval::nnue::v16::SfnnPending;

impl BoardState {
    pub fn new() -> Self {
        let network = Network::get_embedded();
        let mut history = History::new();
        history.accumulators[0].white.init_with_biases(network);
        history.accumulators[0].black.init_with_biases(network);

        Self {
            pieces: PieceMap([Bitboard(0); PIECES]),
            occupancies: SideMap([Bitboard(0); SIDES]),
            piece_mapping: [Piece::None; SQUARES],
            side_to_move: Side::White,
            en_passant_square: Square::NoSquare,
            castle: Castle::NONE,
            move_count: 0,
            phase: 0,
            board_hash: 0,
            half_move_clock: 0,
            history,
            pending_adds_w: [0; 2],
            pending_dels_w: [0; 2],
            pending_adds_b: [0; 2],
            pending_dels_b: [0; 2],
            pending_adds: 0,
            pending_removes: 0,
            sfnn16_pending: SfnnPending::default(),
        }
    }

    #[inline(always)]
    pub fn occupancy(&self) -> Bitboard {
        self.occupancies[Side::White] | self.occupancies[Side::Black]
    }

    #[inline(always)]
    pub fn get_pieces(&self, side: Side, piece: Piece) -> Bitboard {
        self.pieces[piece] & self.occupancies[side]
    }

    #[inline(always)]
    pub fn has_non_pawn_material(&self, side: Side) -> bool {
        (self.occupancies[side]
            ^ self.get_pieces(side, Piece::Pawn)
            ^ self.get_pieces(side, Piece::King))
        .is_not_empty()
    }

    #[inline(always)]
    pub fn is_passed_pawn(&self, sq: usize, side: Side) -> bool {
        let enemy_pawns = self.get_pieces(side.other(), Piece::Pawn).0;
        (enemy_pawns & PASSED_PAWN_MASKS[side as usize][sq]) == 0
    }

    #[inline(always)]
    pub fn get_passed_pawns(&self, side: Side) -> Bitboard {
        let mut passed = Bitboard::EMPTY;
        let mut pawns = self.get_pieces(side, Piece::Pawn);
        while pawns.is_not_empty() {
            let sq = pawns.get_lsb() as usize;
            if self.is_passed_pawn(sq, side) {
                passed.set_bit(sq);
            }
            pawns.clear_lsb();
        }
        passed
    }

    #[inline(always)]
    pub fn add_piece(&mut self, square: Square, side: Side, piece: Piece, update_nnue: bool) {
        let sq = square as usize;
        if sq >= SQUARES {
            return;
        }
        self.pieces[piece].set_bit(sq);
        self.occupancies[side].set_bit(sq);
        self.piece_mapping[sq] = piece;
        self.phase = add_phase(self.phase, piece);
        self.history.invalidate_cache();

        if update_nnue {
            self.nnue_add_piece(square, side, piece);
        }
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, square: Square, update_nnue: bool) -> Piece {
        let sq = square as usize;
        if sq >= SQUARES {
            return Piece::None;
        }
        let piece = self.piece_mapping[sq];
        if piece == Piece::None {
            return Piece::None;
        }

        let side = if self.occupancies[Side::White].get_bit(sq) == 1 {
            Side::White
        } else {
            Side::Black
        };

        self.pieces[piece].clear_bit(sq);
        self.occupancies[Side::White].clear_bit(sq);
        self.occupancies[Side::Black].clear_bit(sq);
        self.piece_mapping[sq] = Piece::None;
        self.phase = remove_phase(self.phase, piece);
        self.history.invalidate_cache();

        if update_nnue {
            self.nnue_remove_piece(square, side, piece);
        }

        piece
    }

    pub fn get_piece_on_side(&self, square: Square, side: Side) -> usize {
        let sq = square as usize;
        if sq >= SQUARES {
            return Piece::None as usize;
        }
        let piece = self.piece_mapping[sq];
        if self.occupancies[side].get_bit(sq) == 1 {
            piece as usize
        } else {
            Piece::None as usize
        }
    }

    pub fn get_piece_on(&self, square: Square) -> i32 {
        let sq = square as usize;
        if sq >= SQUARES {
            return -1;
        }
        let piece = self.piece_mapping[sq];
        if piece == Piece::None {
            return -1;
        }
        if self.occupancies[Side::White].get_bit(sq) == 1 {
            piece as i32
        } else {
            6 + piece as i32
        }
    }

    pub fn clipped_phase(&self) -> i32 {
        get_clipped_phase(self.phase)
    }
}
