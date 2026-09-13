use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks,
    pawn_attacks,
};
use crate::board::history::History;
use crate::common::castle::Castle;
use crate::common::constants::{PIECES, SIDES, SQUARES};
use crate::common::game_phase::{add_phase, get_clipped_phase, remove_phase};
use crate::common::moves::Move;
use crate::common::piece::{Piece, PieceMap};
use crate::common::side::{Side, SideMap};
use crate::common::square::Square;
use crate::eval::nnue::loader::Network;
use crate::eval::nnue::v16::SfnnPending;

#[rustfmt::skip]
pub const CASTLING_CONSTANTS: [u8; SQUARES] = [
     7, 15, 15, 15,  3, 15, 15, 11,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    13, 15, 15, 15, 12, 15, 15, 14,
];

const fn generate_passed_pawn_masks() -> [[u64; SQUARES]; SIDES] {
    let mut masks = [[0u64; SQUARES]; SIDES];
    let mut sq = 0;
    while sq < 64 {
        let file = (sq % 8) as i32;
        let rank = (sq / 8) as i32;

        let mut w_mask = 0u64;
        let mut r = 0;
        while r < rank {
            let mut f = file - 1;
            while f <= file + 1 {
                if f >= 0 && f <= 7 {
                    w_mask |= 1u64 << (r * 8 + f);
                }
                f += 1;
            }
            r += 1;
        }
        masks[0][sq] = w_mask;

        let mut b_mask = 0u64;
        let mut r = rank + 1;
        while r <= 7 {
            let mut f = file - 1;
            while f <= file + 1 {
                if f >= 0 && f <= 7 {
                    b_mask |= 1u64 << (r * 8 + f);
                }
                f += 1;
            }
            r += 1;
        }
        masks[1][sq] = b_mask;

        sq += 1;
    }
    masks
}

pub const PASSED_PAWN_MASKS: [[u64; SQUARES]; SIDES] = generate_passed_pawn_masks();

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
        self.pieces[piece].set_bit(sq);
        self.occupancies[side].set_bit(sq);
        self.piece_mapping[sq] = piece;
        self.phase = add_phase(self.phase, piece);

        if update_nnue {
            self.nnue_add_piece(square, side, piece);
        }
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, square: Square, update_nnue: bool) -> Piece {
        let sq = square as usize;
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

        if update_nnue {
            self.nnue_remove_piece(square, side, piece);
        }

        piece
    }

    pub fn get_piece_on_side(&self, square: Square, side: Side) -> usize {
        let piece = self.piece_mapping[square as usize];
        if self.occupancies[side].get_bit(square as usize) == 1 {
            piece as usize
        } else {
            Piece::None as usize
        }
    }

    pub fn get_piece_on(&self, square: Square) -> i32 {
        let piece = self.piece_mapping[square as usize];
        if piece == Piece::None {
            return -1;
        }
        if self.occupancies[Side::White].get_bit(square as usize) == 1 {
            piece as i32
        } else {
            6 + piece as i32
        }
    }

    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m == Move::NO_MOVE || m.source == m.target {
            return false;
        }
        let src = m.source as usize;
        let tgt = m.target as usize;
        if src >= 64 || tgt >= 64 {
            return false;
        }
        let piece = self.piece_mapping[src];
        if piece == Piece::None {
            return false;
        }
        if self.occupancies[self.side_to_move].get_bit(src) == 0 {
            return false;
        }
        if self.occupancies[self.side_to_move].get_bit(tgt) == 1 {
            return false;
        }
        let occ = self.occupancy();
        let is_target_occupied = occ.get_bit(tgt) == 1;

        match piece {
            Piece::Pawn => {
                if m.is_castle() {
                    return false;
                }
                let (forward, start_rank_min, start_rank_max, promo_rank_min, promo_rank_max) =
                    if self.side_to_move == Side::White {
                        (-8isize, 48usize, 55usize, 0usize, 7usize)
                    } else {
                        (8isize, 8usize, 15usize, 56usize, 63usize)
                    };
                let is_promo_target = (promo_rank_min..=promo_rank_max).contains(&tgt);
                if is_promo_target != m.is_promotion() {
                    return false;
                }
                let diff = tgt as isize - src as isize;
                if diff == forward {
                    if is_target_occupied || m.is_capture() {
                        return false;
                    }
                } else if diff == 2 * forward {
                    if (start_rank_min..=start_rank_max).contains(&src) {
                        let intermediate = (src as isize + forward) as usize;
                        if is_target_occupied || occ.get_bit(intermediate) == 1 || m.is_capture() {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    let pawn_attack_mask =
                        crate::bitboard::lookups::pawn_attacks()[self.side_to_move as usize][src];
                    if (pawn_attack_mask & (1u64 << tgt)) == 0 {
                        return false;
                    }
                    if m.move_type.is_en_passant() {
                        if m.target != self.en_passant_square {
                            return false;
                        }
                    } else if !is_target_occupied {
                        return false;
                    }
                }
            }
            Piece::Knight => {
                if m.is_castle() || m.is_promotion() {
                    return false;
                }
                let attacks = crate::bitboard::lookups::knight_attacks()[src];
                if (attacks & (1u64 << tgt)) == 0 {
                    return false;
                }
                if m.is_capture() != is_target_occupied {
                    return false;
                }
            }
            Piece::Bishop => {
                if m.is_castle() || m.is_promotion() {
                    return false;
                }
                let attacks =
                    crate::bitboard::lookups::get_bishop_attacks_from_table(m.source, occ);
                if (attacks.0 & (1u64 << tgt)) == 0 {
                    return false;
                }
                if m.is_capture() != is_target_occupied {
                    return false;
                }
            }
            Piece::Rook => {
                if m.is_castle() || m.is_promotion() {
                    return false;
                }
                let attacks = crate::bitboard::lookups::get_rook_attacks_from_table(m.source, occ);
                if (attacks.0 & (1u64 << tgt)) == 0 {
                    return false;
                }
                if m.is_capture() != is_target_occupied {
                    return false;
                }
            }
            Piece::Queen => {
                if m.is_castle() || m.is_promotion() {
                    return false;
                }
                let attacks = crate::bitboard::lookups::get_queen_attacks_from_table(m.source, occ);
                if (attacks.0 & (1u64 << tgt)) == 0 {
                    return false;
                }
                if m.is_capture() != is_target_occupied {
                    return false;
                }
            }
            Piece::King => {
                if m.is_promotion() {
                    return false;
                }
                if m.is_castle() {
                    if self.side_to_move == Side::White {
                        if m.source != Square::E1 {
                            return false;
                        }
                        if m.target == Square::G1 {
                            return self.castle.contains(Castle::WHITE_SHORT)
                                && occ.get_bit(Square::F1 as usize) == 0
                                && occ.get_bit(Square::G1 as usize) == 0
                                && !self.is_square_attacked(Square::E1, Side::Black)
                                && !self.is_square_attacked(Square::F1, Side::Black)
                                && !self.is_square_attacked(Square::G1, Side::Black);
                        } else if m.target == Square::C1 {
                            return self.castle.contains(Castle::WHITE_LONG)
                                && occ.get_bit(Square::D1 as usize) == 0
                                && occ.get_bit(Square::C1 as usize) == 0
                                && occ.get_bit(Square::B1 as usize) == 0
                                && !self.is_square_attacked(Square::E1, Side::Black)
                                && !self.is_square_attacked(Square::D1, Side::Black)
                                && !self.is_square_attacked(Square::C1, Side::Black);
                        } else {
                            return false;
                        }
                    } else {
                        if m.source != Square::E8 {
                            return false;
                        }
                        if m.target == Square::G8 {
                            return self.castle.contains(Castle::BLACK_SHORT)
                                && occ.get_bit(Square::F8 as usize) == 0
                                && occ.get_bit(Square::G8 as usize) == 0
                                && !self.is_square_attacked(Square::E8, Side::White)
                                && !self.is_square_attacked(Square::F8, Side::White)
                                && !self.is_square_attacked(Square::G8, Side::White);
                        } else if m.target == Square::C8 {
                            return self.castle.contains(Castle::BLACK_LONG)
                                && occ.get_bit(Square::D8 as usize) == 0
                                && occ.get_bit(Square::C8 as usize) == 0
                                && occ.get_bit(Square::B8 as usize) == 0
                                && !self.is_square_attacked(Square::E8, Side::White)
                                && !self.is_square_attacked(Square::D8, Side::White)
                                && !self.is_square_attacked(Square::C8, Side::White);
                        } else {
                            return false;
                        }
                    }
                }
                let attacks = crate::bitboard::lookups::king_attacks()[src];
                if (attacks & (1u64 << tgt)) == 0 {
                    return false;
                }
                if m.is_capture() != is_target_occupied {
                    return false;
                }
            }
            Piece::None => return false,
        }
        true
    }

    pub fn is_in_check(&self, side: Side) -> bool {
        let king_bb = self.get_pieces(side, Piece::King);
        if king_bb.is_empty() {
            return false;
        }
        let king_sq = Square::from(king_bb.get_lsb() as usize);
        self.is_square_attacked(king_sq, side.other())
    }

    pub fn is_square_attacked(&self, square: Square, attacking_side: Side) -> bool {
        let sq = square as usize;
        let occupancy = self.occupancy();
        let defending_side = attacking_side.other();

        if (self.get_pieces(attacking_side, Piece::Pawn)
            & pawn_attacks()[defending_side as usize][sq])
            .is_not_empty()
        {
            return true;
        }
        if (self.get_pieces(attacking_side, Piece::Knight) & knight_attacks()[sq]).is_not_empty() {
            return true;
        }
        if (self.get_pieces(attacking_side, Piece::King) & king_attacks()[sq]).is_not_empty() {
            return true;
        }

        if (get_bishop_attacks_from_table(square, occupancy)
            & (self.get_pieces(attacking_side, Piece::Bishop)
                | self.get_pieces(attacking_side, Piece::Queen)))
        .is_not_empty()
        {
            return true;
        }
        if (get_rook_attacks_from_table(square, occupancy)
            & (self.get_pieces(attacking_side, Piece::Rook)
                | self.get_pieces(attacking_side, Piece::Queen)))
        .is_not_empty()
        {
            return true;
        }

        false
    }

    pub fn clipped_phase(&self) -> i32 {
        get_clipped_phase(self.phase)
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_piece_sets_all_structures() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);

        let sq = Square::E4 as usize;

        assert_eq!(board.get_pieces(Side::White, Piece::Pawn).get_bit(sq), 1);
        assert_eq!(board.occupancies[Side::White].get_bit(sq), 1);
        assert_eq!(board.occupancy().get_bit(sq), 1);
        assert_eq!(board.occupancies[Side::Black].get_bit(sq), 0);
        assert_eq!(board.piece_mapping[sq], Piece::Pawn);
        assert_eq!(board.phase, 0);
    }

    #[test]
    fn remove_piece_clears_all_structures() {
        let mut board = BoardState::new();
        board.add_piece(Square::D5, Side::Black, Piece::Queen, false);

        let sq = Square::D5 as usize;
        let removed = board.remove_piece(Square::D5, false);

        assert_eq!(removed, Piece::Queen);
        assert_eq!(board.get_pieces(Side::Black, Piece::Queen).get_bit(sq), 0);
        assert_eq!(board.occupancies[Side::Black].get_bit(sq), 0);
        assert_eq!(board.occupancy().get_bit(sq), 0);
        assert_eq!(board.piece_mapping[sq], Piece::None);
    }

    #[test]
    fn get_piece_on_side_returns_correct_piece() {
        let mut board = BoardState::new();
        board.add_piece(Square::C3, Side::White, Piece::Knight, false);

        assert_eq!(
            board.get_piece_on_side(Square::C3, Side::White),
            Piece::Knight as usize
        );
        assert_eq!(
            board.get_piece_on_side(Square::C3, Side::Black),
            Piece::None as usize
        );
        assert_eq!(
            board.get_piece_on_side(Square::D4, Side::White),
            Piece::None as usize
        );
    }

    #[test]
    fn get_piece_on_returns_signed_index() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E8, Side::Black, Piece::King, false);

        assert_eq!(board.get_piece_on(Square::E1), 5);
        assert_eq!(board.get_piece_on(Square::E8), 11);
        assert_eq!(board.get_piece_on(Square::D4), -1);
    }

    #[test]
    fn equality_works_for_blank_boards() {
        let b1 = BoardState::new();
        let b2 = BoardState::new();
        assert_eq!(b1, b2);
    }

    #[test]
    fn equality_detects_difference() {
        let mut b1 = BoardState::new();
        let b2 = BoardState::new();
        b1.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        assert_ne!(b1, b2);
    }

    #[test]
    fn is_in_check_not_in_check_on_empty_board() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        assert!(!board.is_in_check(Side::White));
    }

    #[test]
    fn is_in_check_handles_missing_king_without_panicking() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        assert!(!board.is_in_check(Side::Black));
    }

    #[test]
    fn is_in_check_detects_rook_check() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        assert!(board.is_in_check(Side::White));
    }

    #[test]
    fn is_in_check_detects_knight_check() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::King, false);
        // D6 knight attacks E4
        board.add_piece(Square::D6, Side::Black, Piece::Knight, false);
        assert!(board.is_in_check(Side::White));
    }

    #[test]
    fn is_in_check_detects_bishop_check() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::King, false);
        board.add_piece(Square::H7, Side::Black, Piece::Bishop, false);
        assert!(board.is_in_check(Side::White));
    }

    #[test]
    fn is_in_check_detects_queen_check() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::King, false);
        board.add_piece(Square::E8, Side::Black, Piece::Queen, false);
        assert!(board.is_in_check(Side::White));
    }

    #[test]
    fn is_in_check_detects_pawn_check() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::King, false);
        board.add_piece(Square::D5, Side::Black, Piece::Pawn, false);
        assert!(board.is_in_check(Side::White));
    }

    #[test]
    fn blocker_prevents_rook_check() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        assert!(!board.is_in_check(Side::White));
    }

    #[test]
    fn test_accumulator_make_unmake_consistency() {
        use crate::common::move_list::MoveList;
        use crate::eval::nnue::loader::Network;

        let network = Network::get_embedded();

        let mut board = BoardState::starting_position();

        let expected_white = board.history.accumulators[board.history.index].white;
        let expected_black = board.history.accumulators[board.history.index].black;
        board.refresh_accumulator(Side::White, network);
        board.refresh_accumulator(Side::Black, network);
        assert_eq!(
            board.history.accumulators[board.history.index].white,
            expected_white
        );
        assert_eq!(
            board.history.accumulators[board.history.index].black,
            expected_black
        );

        let mut move_history = Vec::new();
        for _ in 0..10 {
            let mut moves = MoveList::new();
            board.generate_moves(&mut moves);
            if moves.count == 0 {
                break;
            }

            // Find the first legal move
            let mut legal_move = None;
            for m_entry in moves.iter() {
                let m = m_entry.mv;
                board.make_move(m);
                let is_legal = !board.is_in_check(board.side_to_move.other());
                board.unmake_move(m);
                if is_legal {
                    legal_move = Some(m);
                    break;
                }
            }

            let Some(m) = legal_move else {
                break;
            };

            board.make_move(m);
            move_history.push(m);

            let current_white = board.history.accumulators[board.history.index].white;
            let current_black = board.history.accumulators[board.history.index].black;

            board.refresh_accumulator(Side::White, network);
            board.refresh_accumulator(Side::Black, network);

            assert_eq!(
                board.history.accumulators[board.history.index].white, current_white,
                "Failed white accumulator check after move {:?}",
                m
            );
            assert_eq!(
                board.history.accumulators[board.history.index].black, current_black,
                "Failed black accumulator check after move {:?}",
                m
            );
        }

        while let Some(m) = move_history.pop() {
            board.unmake_move(m);

            let current_white = board.history.accumulators[board.history.index].white;
            let current_black = board.history.accumulators[board.history.index].black;

            board.refresh_accumulator(Side::White, network);
            board.refresh_accumulator(Side::Black, network);

            assert_eq!(
                board.history.accumulators[board.history.index].white, current_white,
                "Failed white accumulator check after unmake {:?}",
                m
            );
            assert_eq!(
                board.history.accumulators[board.history.index].black, current_black,
                "Failed black accumulator check after unmake {:?}",
                m
            );
        }

        assert_eq!(
            board.history.accumulators[board.history.index].white,
            expected_white
        );
        assert_eq!(
            board.history.accumulators[board.history.index].black,
            expected_black
        );
    }
}
