use super::BoardState;
use super::tables::BETWEEN_BB;
use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks,
    pawn_attacks,
};
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

impl BoardState {
    pub fn is_in_check(&self, side: Side) -> bool {
        if side == self.side_to_move && self.history.is_cache_valid() {
            return self.history.current_cache().checkers != 0;
        }
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

    #[inline(always)]
    pub fn is_square_attacked_with_occ(
        &self,
        square: Square,
        attacking_side: Side,
        occupancy: Bitboard,
    ) -> bool {
        let sq = square as usize;
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

    #[inline(always)]
    pub fn is_aligned(sq1: Square, sq2: Square) -> bool {
        let r1 = sq1.rank() as i32;
        let f1 = sq1.file() as i32;
        let r2 = sq2.rank() as i32;
        let f2 = sq2.file() as i32;
        r1 == r2 || f1 == f2 || (r1 - r2).abs() == (f1 - f2).abs()
    }

    pub fn pinned_pieces(&self, side: Side) -> Bitboard {
        let king_bb = self.get_pieces(side, Piece::King);
        if king_bb.is_empty() {
            return Bitboard(0);
        }
        let ksq = Square::from(king_bb.get_lsb() as usize);
        let them = side.other();
        let occ = self.occupancy();
        let our_occ = self.occupancies[side].0;

        let enemy_bq =
            self.get_pieces(them, Piece::Bishop).0 | self.get_pieces(them, Piece::Queen).0;
        let enemy_rq = self.get_pieces(them, Piece::Rook).0 | self.get_pieces(them, Piece::Queen).0;

        let diag_pinners = get_bishop_attacks_from_table(ksq, Bitboard(0)).0 & enemy_bq;
        let orth_pinners = get_rook_attacks_from_table(ksq, Bitboard(0)).0 & enemy_rq;

        let mut pinned = 0u64;
        let mut pinners = diag_pinners | orth_pinners;
        while pinners != 0 {
            let psq = pinners.trailing_zeros() as usize;
            pinners &= pinners - 1;
            let ray = BETWEEN_BB[ksq as usize][psq] & occ.0;
            if ray != 0 && (ray & (ray - 1)) == 0 && (ray & our_occ) != 0 {
                pinned |= ray;
            }
        }
        Bitboard(pinned)
    }

    pub fn checkers(&self, side: Side) -> Bitboard {
        let king_bb = self.get_pieces(side, Piece::King);
        if king_bb.is_empty() {
            return Bitboard(0);
        }
        let ksq = Square::from(king_bb.get_lsb() as usize);
        let them = side.other();
        let occ = self.occupancy();

        let mut checkers = 0u64;
        checkers |=
            self.get_pieces(them, Piece::Pawn).0 & pawn_attacks()[side as usize][ksq as usize];
        checkers |= self.get_pieces(them, Piece::Knight).0 & knight_attacks()[ksq as usize];
        let enemy_bq =
            self.get_pieces(them, Piece::Bishop).0 | self.get_pieces(them, Piece::Queen).0;
        checkers |= get_bishop_attacks_from_table(ksq, occ).0 & enemy_bq;
        let enemy_rq = self.get_pieces(them, Piece::Rook).0 | self.get_pieces(them, Piece::Queen).0;
        checkers |= get_rook_attacks_from_table(ksq, occ).0 & enemy_rq;

        Bitboard(checkers)
    }

    pub fn check_squares(&self, side: Side) -> [u64; 6] {
        let king_bb = self.get_pieces(side, Piece::King);
        if king_bb.is_empty() {
            return [0; 6];
        }
        let ksq = Square::from(king_bb.get_lsb() as usize);
        let occ = self.occupancy();
        let bishop_attacks = get_bishop_attacks_from_table(ksq, occ).0;
        let rook_attacks = get_rook_attacks_from_table(ksq, occ).0;

        let mut squares = [0u64; 6];
        squares[Piece::Pawn as usize] = pawn_attacks()[side as usize][ksq as usize];
        squares[Piece::Knight as usize] = knight_attacks()[ksq as usize];
        squares[Piece::Bishop as usize] = bishop_attacks;
        squares[Piece::Rook as usize] = rook_attacks;
        squares[Piece::Queen as usize] = bishop_attacks | rook_attacks;
        squares[Piece::King as usize] = 0;
        squares
    }

    pub fn threat_by_lesser(&self, side: Side) -> [u64; 6] {
        let them = side.other();
        let pawns = self.get_pieces(them, Piece::Pawn).0;
        let pawn_threats = if them == Side::White {
            ((pawns >> 9) & !crate::bitboard::attacks::FILE_H)
                | ((pawns >> 7) & !crate::bitboard::attacks::FILE_A)
        } else {
            ((pawns << 7) & !crate::bitboard::attacks::FILE_H)
                | ((pawns << 9) & !crate::bitboard::attacks::FILE_A)
        };

        let mut knights = self.get_pieces(them, Piece::Knight);
        let mut knight_threats = 0u64;
        while !knights.is_empty() {
            let sq = knights.get_lsb();
            knights.clear_lsb();
            knight_threats |= knight_attacks()[sq as usize];
        }

        let occ = self.occupancy();
        let enemy_bq =
            self.get_pieces(them, Piece::Bishop).0 | self.get_pieces(them, Piece::Queen).0;
        let mut bishops = Bitboard(enemy_bq);
        let mut bishop_threats = 0u64;
        while !bishops.is_empty() {
            let sq = bishops.get_lsb();
            bishops.clear_lsb();
            bishop_threats |= get_bishop_attacks_from_table(Square::from(sq as usize), occ).0;
        }

        let enemy_rq = self.get_pieces(them, Piece::Rook).0 | self.get_pieces(them, Piece::Queen).0;
        let mut rooks = Bitboard(enemy_rq);
        let mut rook_threats = 0u64;
        while !rooks.is_empty() {
            let sq = rooks.get_lsb();
            rooks.clear_lsb();
            rook_threats |= get_rook_attacks_from_table(Square::from(sq as usize), occ).0;
        }

        let mut threats = [0u64; 6];
        threats[Piece::Pawn as usize] = 0;
        threats[Piece::Knight as usize] = pawn_threats;
        threats[Piece::Bishop as usize] = pawn_threats;
        threats[Piece::Rook as usize] = pawn_threats | knight_threats | bishop_threats;
        threats[Piece::Queen as usize] = threats[Piece::Rook as usize] | rook_threats;
        threats[Piece::King as usize] = 0;
        threats
    }

    pub fn compute_pinners(&self, side: Side) -> Bitboard {
        let king_bb = self.get_pieces(side, Piece::King);
        if king_bb.is_empty() {
            return Bitboard(0);
        }
        let ksq = Square::from(king_bb.get_lsb() as usize);
        let them = side.other();
        let occ = self.occupancy();
        let own_occ = self.occupancies[side].0;
        let enemy_bq =
            self.get_pieces(them, Piece::Bishop).0 | self.get_pieces(them, Piece::Queen).0;
        let enemy_rq = self.get_pieces(them, Piece::Rook).0 | self.get_pieces(them, Piece::Queen).0;
        let diag = get_bishop_attacks_from_table(ksq, Bitboard(0)).0 & enemy_bq;
        let orth = get_rook_attacks_from_table(ksq, Bitboard(0)).0 & enemy_rq;
        let mut pinners = 0u64;
        let mut cand = diag | orth;
        while cand != 0 {
            let psq = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            let ray = BETWEEN_BB[ksq as usize][psq] & occ.0;
            if ray != 0 && (ray & (ray - 1)) == 0 && (ray & own_occ) != 0 {
                pinners |= 1u64 << psq;
            }
        }
        Bitboard(pinners)
    }

    pub fn compute_all_threats(&self, attacking_side: Side) -> Bitboard {
        let pawns = self.get_pieces(attacking_side, Piece::Pawn).0;
        let mut threats = if attacking_side == Side::White {
            ((pawns >> 9) & !crate::bitboard::attacks::FILE_H)
                | ((pawns >> 7) & !crate::bitboard::attacks::FILE_A)
        } else {
            ((pawns << 7) & !crate::bitboard::attacks::FILE_H)
                | ((pawns << 9) & !crate::bitboard::attacks::FILE_A)
        };
        let mut knights = self.get_pieces(attacking_side, Piece::Knight);
        while !knights.is_empty() {
            let sq = knights.get_lsb();
            knights.clear_lsb();
            threats |= knight_attacks()[sq as usize];
        }
        let stm = self.side_to_move;
        let mut occ = self.occupancy();
        if attacking_side != stm {
            let kbb = self.get_pieces(stm, Piece::King);
            if !kbb.is_empty() {
                occ = Bitboard(occ.0 ^ (1u64 << (kbb.get_lsb() as usize)));
            }
        } else {
            let kbb = self.get_pieces(stm.other(), Piece::King);
            if !kbb.is_empty() {
                occ = Bitboard(occ.0 ^ (1u64 << (kbb.get_lsb() as usize)));
            }
        }
        let enemy_bq = self.get_pieces(attacking_side, Piece::Bishop).0
            | self.get_pieces(attacking_side, Piece::Queen).0;
        let mut bishops = Bitboard(enemy_bq);
        while !bishops.is_empty() {
            let sq = bishops.get_lsb();
            bishops.clear_lsb();
            threats |= get_bishop_attacks_from_table(Square::from(sq as usize), occ).0;
        }
        let enemy_rq = self.get_pieces(attacking_side, Piece::Rook).0
            | self.get_pieces(attacking_side, Piece::Queen).0;
        let mut rooks = Bitboard(enemy_rq);
        while !rooks.is_empty() {
            let sq = rooks.get_lsb();
            rooks.clear_lsb();
            threats |= get_rook_attacks_from_table(Square::from(sq as usize), occ).0;
        }
        let kbb = self.get_pieces(attacking_side, Piece::King);
        if !kbb.is_empty() {
            threats |= king_attacks()[kbb.get_lsb() as usize];
        }
        Bitboard(threats)
    }
}
