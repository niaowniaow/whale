use std::cmp::max;

use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks,
    pawn_attacks,
};
use crate::board::state::BoardState;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

impl Piece {
    #[inline]
    pub const fn see_value(self) -> i16 {
        match self {
            Piece::Pawn => 100,
            Piece::Knight => 300,
            Piece::Bishop => 300,
            Piece::Rook => 500,
            Piece::Queen => 900,
            Piece::King => 20000,
            Piece::None => 0,
        }
    }
}

impl BoardState {
    pub fn see_ge(&self, mv: Move, threshold: i16) -> bool {
        let captured = self.get_initial_captured_piece(mv);
        let mut initial = captured.see_value() as i32;
        let attacker_val = if mv.is_promotion() {
            let promo = mv.move_type.promotion_piece();
            initial += promo.see_value() as i32 - Piece::Pawn.see_value() as i32;
            promo.see_value() as i32
        } else {
            let piece = self.piece_mapping[mv.source as usize];
            if piece == Piece::None {
                return self.see(mv) >= threshold;
            }
            piece.see_value() as i32
        };

        let target_rank = mv.target as usize / 8;
        let enemy_promo_rank = if self.side_to_move == Side::White {
            target_rank == 0
        } else {
            target_rank == 7
        };
        if !enemy_promo_rank && initial - attacker_val >= threshold as i32 {
            return true;
        }
        self.see(mv) >= threshold
    }

    pub fn see(&self, mv: Move) -> i16 {
        let (source, target) = (mv.source, mv.target);
        let mut occupancy = self.occupancy();
        let mut side = self.side_to_move;

        let mut gain = [0i16; 32];
        gain[0] = self.get_initial_gain(mv, self.get_initial_captured_piece(mv));
        self.clear_en_passant_square(mv, &mut occupancy, side);

        let mut attackers = self.get_all_attackers(target, occupancy);

        occupancy.clear_bit(source as usize);
        attackers.clear_bit(source as usize);

        self.update_xrays(&mut attackers, target, occupancy);

        let mut depth = 1;
        let mut last_captured_piece = if mv.is_promotion() {
            mv.move_type.promotion_piece()
        } else {
            self.piece_mapping[source as usize]
        };
        side = side.other();

        while attackers.is_not_empty() {
            let side_attackers = attackers & self.occupancies[side];
            if side_attackers.is_empty() {
                break;
            }

            let (from_sq, piece) = self.get_least_valuable_attacker(side_attackers, side);
            if from_sq == Square::NoSquare {
                break;
            }

            let (step_gain, next_captured) =
                self.get_recapture_gain_and_piece(piece, target, side, last_captured_piece);

            occupancy.clear_bit(from_sq as usize);
            attackers.clear_bit(from_sq as usize);
            self.update_xrays(&mut attackers, target, occupancy);

            last_captured_piece = next_captured;
            side = side.other();
            gain[depth] = step_gain;
            depth += 1;
        }

        for i in (1..depth).rev() {
            gain[i - 1] -= max(0, gain[i]);
        }

        gain[0]
    }

    #[inline(always)]
    fn get_initial_captured_piece(&self, mv: Move) -> Piece {
        if mv.move_type == MoveType::EnPassant {
            Piece::Pawn
        } else {
            self.piece_mapping[mv.target as usize]
        }
    }

    #[inline(always)]
    fn get_initial_gain(&self, mv: Move, captured: Piece) -> i16 {
        let mut gain = captured.see_value();
        if mv.is_promotion() {
            gain += mv.move_type.promotion_piece().see_value() - Piece::Pawn.see_value();
        }
        gain
    }

    #[inline(always)]
    fn clear_en_passant_square(&self, mv: Move, occupancy: &mut Bitboard, side: Side) {
        if mv.move_type == MoveType::EnPassant {
            let ep_sq = if side == Side::White {
                mv.target as usize + 8
            } else {
                mv.target as usize - 8
            };
            occupancy.clear_bit(ep_sq);
        }
    }

    #[inline(always)]
    fn get_recapture_gain_and_piece(
        &self,
        piece: Piece,
        target: Square,
        side: Side,
        last_captured: Piece,
    ) -> (i16, Piece) {
        let mut gain = last_captured.see_value();
        let mut next_captured = piece;

        if piece == Piece::Pawn {
            let rank = target as usize / 8;
            if (side == Side::White && rank == 0) || (side == Side::Black && rank == 7) {
                gain += Piece::Queen.see_value() - Piece::Pawn.see_value();
                next_captured = Piece::Queen;
            }
        }
        (gain, next_captured)
    }

    fn get_all_attackers(&self, sq: Square, occupancy: Bitboard) -> Bitboard {
        let white_pawns = self.get_pieces(Side::White, Piece::Pawn);
        let black_pawns = self.get_pieces(Side::Black, Piece::Pawn);
        let knights = self.pieces[Piece::Knight];
        let bishops = self.pieces[Piece::Bishop];
        let rooks = self.pieces[Piece::Rook];
        let queens = self.pieces[Piece::Queen];
        let kings = self.pieces[Piece::King];

        let pawn_attacks = (white_pawns & pawn_attacks()[Side::Black as usize][sq as usize])
            | (black_pawns & pawn_attacks()[Side::White as usize][sq as usize]);
        let knight_attacks = knights & knight_attacks()[sq as usize];
        let bishop_attacks = get_bishop_attacks_from_table(sq, occupancy) & (bishops | queens);
        let rook_attacks = get_rook_attacks_from_table(sq, occupancy) & (rooks | queens);
        let king_attacks = kings & king_attacks()[sq as usize];

        pawn_attacks | knight_attacks | bishop_attacks | rook_attacks | king_attacks
    }

    fn update_xrays(&self, attackers: &mut Bitboard, target: Square, occupancy: Bitboard) {
        let bishops = self.pieces[Piece::Bishop];
        let rooks = self.pieces[Piece::Rook];
        let queens = self.pieces[Piece::Queen];

        let diagonal_attackers =
            get_bishop_attacks_from_table(target, occupancy) & (bishops | queens) & occupancy;
        *attackers |= diagonal_attackers;

        let orthogonal_attackers =
            get_rook_attacks_from_table(target, occupancy) & (rooks | queens) & occupancy;
        *attackers |= orthogonal_attackers;
    }

    fn get_least_valuable_attacker(&self, side_attackers: Bitboard, side: Side) -> (Square, Piece) {
        for piece in Piece::ALL {
            let pieces_bb = self.get_pieces(side, piece);
            let intersection = side_attackers & pieces_bb;
            if intersection.is_not_empty() {
                let sq = Square::from(intersection.get_lsb() as usize);
                return (sq, piece);
            }
        }
        (Square::NoSquare, Piece::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;

    #[test]
    fn test_see_hanging_piece() {
        let board = BoardState::parse_fen("k7/8/8/8/3q4/8/3R4/K7 w - - 0 1");
        let mv = Move::new(Square::D2, Square::D4, MoveType::Capture);

        let score = board.see(mv);
        assert_eq!(score, 900);
    }

    #[test]
    fn test_see_defended_piece() {
        let board = BoardState::parse_fen("k7/8/8/4p3/3q4/8/3R4/K7 w - - 0 1");
        let mv = Move::new(Square::D2, Square::D4, MoveType::Capture);

        let score = board.see(mv);
        assert_eq!(score, 400);
    }

    #[test]
    fn test_see_equal_exchange() {
        let board = BoardState::parse_fen("k7/8/4p3/3b4/8/2N5/8/K7 w - - 0 1");
        let mv = Move::new(Square::C3, Square::D5, MoveType::Capture);

        let score = board.see(mv);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_see_xrays() {
        let board = BoardState::parse_fen("k7/8/3r4/8/3q4/8/3R4/3R3K w - - 0 1");
        let mv = Move::new(Square::D2, Square::D4, MoveType::Capture);

        let score = board.see(mv);
        assert_eq!(score, 900);
    }

    #[test]
    fn test_see_bad_capture() {
        let board = BoardState::parse_fen("k7/8/8/5n2/3p4/8/3R4/K7 w - - 0 1");
        let mv = Move::new(Square::D2, Square::D4, MoveType::Capture);

        let score = board.see(mv);
        assert_eq!(score, -400);
    }

    #[test]
    fn test_see_promotion_capture() {
        let board = BoardState::parse_fen("rr6/P7/k7/8/8/8/8/K7 w - - 0 1");

        let mv_queen = Move::new(Square::A7, Square::B8, MoveType::QueenPromotionCapture);
        assert_eq!(board.see(mv_queen), 400);

        let mv_rook = Move::new(Square::A7, Square::B8, MoveType::RookPromotionCapture);
        assert_eq!(board.see(mv_rook), 400);

        let mv_bishop = Move::new(Square::A7, Square::B8, MoveType::BishopPromotionCapture);
        assert_eq!(board.see(mv_bishop), 400);

        let mv_knight = Move::new(Square::A7, Square::B8, MoveType::KnightPromotionCapture);
        assert_eq!(board.see(mv_knight), 400);
    }

    #[test]
    fn see_ge_matches_threshold_compare_everywhere() {
        use crate::common::helpers::{ADVANCED_MOVE_FEN, ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN};
        use crate::common::move_list::MoveList;

        let thresholds: [i16; 9] = [-600, -400, -200, -100, -75, -74, -38, 0, 100];
        for fen in [
            STARTING_FEN,
            KIWI_PETE_FEN,
            ENDGAME_FEN,
            ADVANCED_MOVE_FEN,
            "7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1",
            "7k/8/8/8/3p4/8/3R4/K7 w - - 0 1",
            "rr6/P7/k7/8/8/8/8/K7 w - - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        ] {
            let board = BoardState::parse_fen(fen);
            let mut moves = MoveList::new();
            board.generate_moves(&mut moves);
            assert!(!moves.is_empty());
            for entry in moves.iter() {
                for &t in &thresholds {
                    assert_eq!(
                        board.see_ge(entry.mv, t),
                        board.see(entry.mv) >= t,
                        "see_ge mismatch {:?} thr={t} in {fen}",
                        entry.mv
                    );
                }
            }
        }
    }

    #[test]
    fn see_values_cover_all_pieces() {
        assert_eq!(Piece::Pawn.see_value(), 100);
        assert_eq!(Piece::Knight.see_value(), 300);
        assert_eq!(Piece::Bishop.see_value(), 300);
        assert_eq!(Piece::Rook.see_value(), 500);
        assert_eq!(Piece::Queen.see_value(), 900);
        assert_eq!(Piece::King.see_value(), 20000);
        assert_eq!(Piece::None.see_value(), 0);
    }

    #[test]
    fn see_quiet_move_scores_zero() {
        let board =
            BoardState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mv = Move::new(Square::E2, Square::E3, MoveType::Quiet);
        assert_eq!(board.see(mv), 0);
    }

    #[test]
    fn see_en_passant_captures_pawn() {
        let board = BoardState::parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        assert_eq!(board.en_passant_square, Square::D6);
        let mv = Move::new(Square::E5, Square::D6, MoveType::EnPassant);
        assert_eq!(board.see(mv), 100);
    }

    #[test]
    fn see_black_en_passant_captures_pawn() {
        let board = BoardState::parse_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
        assert_eq!(board.en_passant_square, Square::E3);
        let mv = Move::new(Square::D4, Square::E3, MoveType::EnPassant);
        assert_eq!(board.see(mv), 100);
    }

    #[test]
    fn see_quiet_promotion_adds_piece_value() {
        let board = BoardState::parse_fen("k7/6P1/8/8/8/8/8/4K3 w - - 0 1");
        let mv = Move::new(Square::G7, Square::G8, MoveType::QueenPromotion);
        assert_eq!(board.see(mv), 800);
    }

    #[test]
    fn see_white_pawn_recapture_promotes() {
        let board = BoardState::parse_fen("1R2k3/q1P5/8/8/8/8/8/4K3 b - - 0 1");
        let mv = Move::new(Square::A7, Square::B8, MoveType::Capture);
        assert_eq!(board.see(mv), -1200);
    }

    #[test]
    fn see_black_pawn_recapture_promotes() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/8/Q1p5/1r2K3 w - - 0 1");
        let mv = Move::new(Square::A2, Square::B1, MoveType::Capture);
        assert_eq!(board.see(mv), -1200);
    }

    #[test]
    fn see_king_recapture() {
        let board = BoardState::parse_fen("4k3/4p3/8/8/8/8/4R3/4K3 w - - 0 1");
        let mv = Move::new(Square::E2, Square::E7, MoveType::Capture);
        assert_eq!(board.see(mv), -400);
    }

    #[test]
    fn see_diagonal_xray() {
        let board = BoardState::parse_fen("k7/8/3b4/8/3q4/8/3B4/3B3K w - - 0 1");
        let mv = Move::new(Square::D2, Square::D4, MoveType::Capture);
        assert_eq!(board.see(mv), 900);
    }

    #[test]
    fn see_queen_recapture() {
        let board = BoardState::parse_fen("3qk3/8/8/3n4/4P3/8/8/4K3 w - - 0 1");
        let mv = Move::new(Square::E4, Square::D5, MoveType::Capture);
        assert_eq!(board.see(mv), 200);
    }
}
