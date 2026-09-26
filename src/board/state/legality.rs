use super::BoardState;
use super::tables::{BETWEEN_BB, LINE_BB};
use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, knight_attacks, pawn_attacks,
};
use crate::common::castle::Castle;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

impl BoardState {
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
                    if !m.move_type.is_double_push() {
                        return false;
                    }
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
                    } else if m.is_capture() != is_target_occupied {
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

    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move;
        if self.history.is_cache_valid() {
            let c = self.history.current_cache();
            return self.is_legal_with(m, c.checkers, c.pinned);
        }
        self.is_legal_with(m, self.checkers(us).0, self.pinned_pieces(us).0)
    }

    pub fn is_legal_with(&self, m: Move, checkers: u64, pinned: u64) -> bool {
        if !self.is_pseudo_legal(m) {
            return false;
        }
        let us = self.side_to_move;
        let them = us.other();
        let king_bb = self.get_pieces(us, Piece::King);
        if king_bb.is_empty() {
            return false;
        }
        let ksq = Square::from(king_bb.get_lsb() as usize);
        let from = m.source;
        let to = m.target;
        let from_piece = self.piece_mapping[from as usize];

        if from_piece == Piece::King {
            if m.move_type == MoveType::Castle {
                let occ = self.occupancy();
                let castle_ok = if us == Side::White {
                    if from != Square::E1 {
                        return false;
                    }
                    if to == Square::G1 {
                        self.castle.contains(Castle::WHITE_SHORT)
                            && occ.get_bit(Square::F1 as usize) == 0
                            && occ.get_bit(Square::G1 as usize) == 0
                            && self.piece_mapping[Square::H1 as usize] == Piece::Rook
                            && self.occupancies[Side::White].get_bit(Square::H1 as usize) == 1
                    } else if to == Square::C1 {
                        self.castle.contains(Castle::WHITE_LONG)
                            && occ.get_bit(Square::D1 as usize) == 0
                            && occ.get_bit(Square::C1 as usize) == 0
                            && occ.get_bit(Square::B1 as usize) == 0
                            && self.piece_mapping[Square::A1 as usize] == Piece::Rook
                            && self.occupancies[Side::White].get_bit(Square::A1 as usize) == 1
                    } else {
                        return false;
                    }
                } else {
                    if from != Square::E8 {
                        return false;
                    }
                    if to == Square::G8 {
                        self.castle.contains(Castle::BLACK_SHORT)
                            && occ.get_bit(Square::F8 as usize) == 0
                            && occ.get_bit(Square::G8 as usize) == 0
                            && self.piece_mapping[Square::H8 as usize] == Piece::Rook
                            && self.occupancies[Side::Black].get_bit(Square::H8 as usize) == 1
                    } else if to == Square::C8 {
                        self.castle.contains(Castle::BLACK_LONG)
                            && occ.get_bit(Square::D8 as usize) == 0
                            && occ.get_bit(Square::C8 as usize) == 0
                            && occ.get_bit(Square::B8 as usize) == 0
                            && self.piece_mapping[Square::A8 as usize] == Piece::Rook
                            && self.occupancies[Side::Black].get_bit(Square::A8 as usize) == 1
                    } else {
                        return false;
                    }
                };
                if !castle_ok {
                    return false;
                }
                if self.history.is_cache_valid() {
                    let c = self.history.current_cache();
                    let threats = c.all_threats;
                    let pinned = c.pinned;
                    let rook_sq = if us == Side::White {
                        if to == Square::G1 {
                            Square::H1 as usize
                        } else {
                            Square::A1 as usize
                        }
                    } else if to == Square::G8 {
                        Square::H8 as usize
                    } else {
                        Square::A8 as usize
                    };
                    if (pinned & (1u64 << rook_sq)) != 0 {
                        return false;
                    }
                    let transit = if to > from {
                        Square::from(from as usize + 1)
                    } else {
                        Square::from(from as usize - 1)
                    };
                    if (threats & (1u64 << (ksq as usize))) != 0 {
                        return false;
                    }
                    if (threats & (1u64 << (transit as usize))) != 0 {
                        return false;
                    }
                    return (threats & (1u64 << (to as usize))) == 0;
                }
                if self.is_square_attacked(ksq, them) {
                    return false;
                }
                let transit = if to > from {
                    Square::from(from as usize + 1)
                } else {
                    Square::from(from as usize - 1)
                };
                if self.is_square_attacked(transit, them) {
                    return false;
                }
                return !self.is_square_attacked(to, them);
            }
            if self.history.is_cache_valid() {
                let c = self.history.current_cache();
                return (c.all_threats & (1u64 << (to as usize))) == 0;
            }
            let occ = Bitboard(self.occupancy().0 ^ (1u64 << (from as usize)));
            return !self.is_square_attacked_with_occ(to, them, occ);
        }

        if m.move_type == MoveType::EnPassant {
            if self.history.is_cache_valid() {
                let c = self.history.current_cache();
                if (c.pinned & (1u64 << (from as usize))) != 0
                    && (LINE_BB[ksq as usize][from as usize] & (1u64 << (to as usize))) == 0
                {
                    return false;
                }
                let cap_sq = Square::from_rank_file(from.rank(), to.file());
                let occ_after =
                    self.occupancy().0 ^ (1u64 << (from as usize)) ^ (1u64 << (cap_sq as usize))
                        | (1u64 << (to as usize));
                let enemy_bq =
                    self.get_pieces(them, Piece::Bishop).0 | self.get_pieces(them, Piece::Queen).0;
                let enemy_rq =
                    self.get_pieces(them, Piece::Rook).0 | self.get_pieces(them, Piece::Queen).0;
                let mut sliders = enemy_bq | enemy_rq;
                while sliders != 0 {
                    let psq = sliders.trailing_zeros() as usize;
                    sliders &= sliders - 1;
                    if LINE_BB[ksq as usize][psq] == 0 {
                        continue;
                    }
                    let between = BETWEEN_BB[ksq as usize][psq] & occ_after;
                    if between != 0 {
                        continue;
                    }
                    let is_diag =
                        get_bishop_attacks_from_table(ksq, Bitboard(0)).0 & (1u64 << psq) != 0;
                    if is_diag {
                        if (enemy_bq & (1u64 << psq)) != 0 {
                            return false;
                        }
                    } else if (enemy_rq & (1u64 << psq)) != 0 {
                        return false;
                    }
                }
                let enemy_pawns =
                    self.get_pieces(them, Piece::Pawn).0 & !(1u64 << (cap_sq as usize));
                if (enemy_pawns & pawn_attacks()[us as usize][ksq as usize]) != 0 {
                    return false;
                }
                if (self.get_pieces(them, Piece::Knight).0 & knight_attacks()[ksq as usize]) != 0 {
                    return false;
                }
                return true;
            }
            let cap_sq = Square::from_rank_file(from.rank(), to.file());
            let occ = Bitboard(
                self.occupancy().0 ^ (1u64 << (from as usize)) ^ (1u64 << (cap_sq as usize))
                    | (1u64 << (to as usize)),
            );
            let enemy_pawns = self.get_pieces(them, Piece::Pawn).0 & !(1u64 << (cap_sq as usize));
            if (enemy_pawns & pawn_attacks()[us as usize][ksq as usize]) != 0 {
                return false;
            }
            if (self.get_pieces(them, Piece::Knight).0 & knight_attacks()[ksq as usize]) != 0 {
                return false;
            }
            let enemy_bq =
                self.get_pieces(them, Piece::Bishop).0 | self.get_pieces(them, Piece::Queen).0;
            if (get_bishop_attacks_from_table(ksq, occ).0 & enemy_bq) != 0 {
                return false;
            }
            let enemy_rq =
                self.get_pieces(them, Piece::Rook).0 | self.get_pieces(them, Piece::Queen).0;
            if (get_rook_attacks_from_table(ksq, occ).0 & enemy_rq) != 0 {
                return false;
            }
            return true;
        }

        if checkers == 0 {
            if (pinned & (1u64 << from as usize)) == 0 {
                return true;
            }
            return (LINE_BB[ksq as usize][from as usize] & (1u64 << to as usize)) != 0;
        }

        let num_checkers = checkers.count_ones();
        if num_checkers > 1 {
            return false;
        }

        if (pinned & (1u64 << from as usize)) != 0
            && (LINE_BB[ksq as usize][from as usize] & (1u64 << to as usize)) == 0
        {
            return false;
        }

        let checker_sq = checkers.trailing_zeros() as usize;
        if to as usize == checker_sq {
            return true;
        }

        (BETWEEN_BB[ksq as usize][checker_sq] & (1u64 << to as usize)) != 0
    }
}
