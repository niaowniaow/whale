use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks,
    pawn_attacks,
};
use crate::board::history::History;
use crate::common::castle::Castle;
use crate::common::constants::{PIECES, SIDES, SQUARES};
use crate::common::game_phase::{add_phase, get_clipped_phase, remove_phase};
use crate::common::move_type::MoveType;
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

const fn generate_ray_tables() -> ([[u64; SQUARES]; SQUARES], [[u64; SQUARES]; SQUARES]) {
    let mut between = [[0u64; SQUARES]; SQUARES];
    let mut line = [[0u64; SQUARES]; SQUARES];
    let mut sq1 = 0;
    while sq1 < 64 {
        let r1 = sq1 as i32 / 8;
        let f1 = sq1 as i32 % 8;
        let mut sq2 = 0;
        while sq2 < 64 {
            if sq1 != sq2 {
                let r2 = sq2 as i32 / 8;
                let f2 = sq2 as i32 % 8;
                let dr = if r2 > r1 {
                    1
                } else if r2 < r1 {
                    -1
                } else {
                    0
                };
                let df = if f2 > f1 {
                    1
                } else if f2 < f1 {
                    -1
                } else {
                    0
                };
                let dy = if r2 >= r1 { r2 - r1 } else { r1 - r2 };
                let dx = if f2 >= f1 { f2 - f1 } else { f1 - f2 };
                if dr == 0 || df == 0 || dy == dx {
                    let mut curr_r = r1 + dr;
                    let mut curr_f = f1 + df;
                    let mut b_mask = 0u64;
                    while curr_r != r2 || curr_f != f2 {
                        b_mask |= 1u64 << (curr_r * 8 + curr_f);
                        curr_r += dr;
                        curr_f += df;
                    }
                    between[sq1][sq2] = b_mask;

                    let mut l_mask = 0u64;
                    let mut r = r1;
                    let mut f = f1;
                    while r >= 0 && r < 8 && f >= 0 && f < 8 {
                        l_mask |= 1u64 << (r * 8 + f);
                        r += dr;
                        f += df;
                    }
                    r = r1 - dr;
                    f = f1 - df;
                    while r >= 0 && r < 8 && f >= 0 && f < 8 {
                        l_mask |= 1u64 << (r * 8 + f);
                        r -= dr;
                        f -= df;
                    }
                    line[sq1][sq2] = l_mask;
                }
            }
            sq2 += 1;
        }
        sq1 += 1;
    }
    (between, line)
}

pub static RAY_TABLES: ([[u64; SQUARES]; SQUARES], [[u64; SQUARES]; SQUARES]) =
    generate_ray_tables();
pub static BETWEEN_BB: &[[u64; SQUARES]; SQUARES] = &RAY_TABLES.0;
pub static LINE_BB: &[[u64; SQUARES]; SQUARES] = &RAY_TABLES.1;

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
        let occ = self.occupancy();
        let mut pawns = self.get_pieces(them, Piece::Pawn);
        let mut pawn_threats = 0u64;
        while !pawns.is_empty() {
            let sq = pawns.get_lsb();
            pawns.clear_lsb();
            pawn_threats |= pawn_attacks()[them as usize][sq as usize];
        }

        let mut knights = self.get_pieces(them, Piece::Knight);
        let mut knight_threats = 0u64;
        while !knights.is_empty() {
            let sq = knights.get_lsb();
            knights.clear_lsb();
            knight_threats |= knight_attacks()[sq as usize];
        }

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

    pub fn is_legal(&self, m: Move) -> bool {
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
                // Castling must originate from the king's home square with the
                // right still available, an empty path and the rook on its
                // home square (cf. Reckless board.rs is_legal castling branch).
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
            let occ = Bitboard(self.occupancy().0 ^ (1u64 << (from as usize)));
            return !self.is_square_attacked_with_occ(to, them, occ);
        }

        if m.move_type == MoveType::EnPassant {
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

        let checkers = self.checkers(us).0;
        let pinned = self.pinned_pieces(us).0;

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

        // A pinned piece may still resolve a single check by moving along the
        // pin ray (to capture the checker or interpose), cf. Stockfish
        // position.cpp legal() and Reckless board.rs is_legal().
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

            board.ensure_accumulators_fresh();
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

            board.ensure_accumulators_fresh();
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

    #[test]
    fn test_is_legal_matches_make_unmake() {
        use crate::common::move_list::MoveList;

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "8/8/8/8/k2Pp2R/8/8/3K4 b - d3 0 1",
        ];

        for fen in fens {
            let mut board = BoardState::parse_fen(fen);
            let mut moves = MoveList::new();
            board.generate_moves(&mut moves);

            for m_entry in moves.iter() {
                let m = m_entry.mv;
                let fast_legal = board.is_legal(m);

                board.make_move(m);
                let slow_legal = !board.is_in_check(board.side_to_move.other());
                board.unmake_move(m);

                assert_eq!(
                    fast_legal, slow_legal,
                    "Legality mismatch for move {:?} in FEN: {}",
                    m, fen
                );
            }
        }
    }

    #[test]
    fn has_non_pawn_material_detects_major_pieces() {
        use crate::common::helpers::STARTING_FEN;

        let board = BoardState::parse_fen(STARTING_FEN);
        assert!(board.has_non_pawn_material(Side::White));
        assert!(board.has_non_pawn_material(Side::Black));

        let mut bare = BoardState::new();
        bare.add_piece(Square::E1, Side::White, Piece::King, false);
        bare.add_piece(Square::E8, Side::Black, Piece::King, false);
        assert!(!bare.has_non_pawn_material(Side::White));
        assert!(!bare.has_non_pawn_material(Side::Black));

        bare.add_piece(Square::E2, Side::White, Piece::Pawn, false);
        assert!(!bare.has_non_pawn_material(Side::White));

        bare.add_piece(Square::D1, Side::White, Piece::Queen, false);
        assert!(bare.has_non_pawn_material(Side::White));
    }

    #[test]
    fn passed_pawn_detection() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        assert!(board.is_passed_pawn(Square::E4 as usize, Side::White));
        assert_eq!(
            board
                .get_passed_pawns(Side::White)
                .get_bit(Square::E4 as usize),
            1
        );

        // Black pawn on D5 attacks E4, so it is no longer passed.
        board.add_piece(Square::D5, Side::Black, Piece::Pawn, false);
        assert!(!board.is_passed_pawn(Square::E4 as usize, Side::White));
        assert!(board.get_passed_pawns(Side::White).is_empty());
    }

    #[test]
    fn is_aligned_covers_ranks_files_diagonals() {
        assert!(BoardState::is_aligned(Square::A1, Square::H1));
        assert!(BoardState::is_aligned(Square::E1, Square::E8));
        assert!(BoardState::is_aligned(Square::A1, Square::H8));
        assert!(BoardState::is_aligned(Square::E4, Square::E4));
        assert!(!BoardState::is_aligned(Square::A1, Square::H7));
        assert!(!BoardState::is_aligned(Square::E2, Square::F4));
    }

    #[test]
    fn square_attack_queries() {
        use crate::common::helpers::STARTING_FEN;

        let board = BoardState::parse_fen(STARTING_FEN);
        // White pawn on D2 attacks C3 and E3.
        assert!(board.is_square_attacked(Square::E3, Side::White));
        assert!(board.is_square_attacked(Square::C3, Side::White));
        assert!(!board.is_square_attacked(Square::E4, Side::White));
        assert!(!board.is_square_attacked(Square::E4, Side::Black));

        let occ = board.occupancy();
        assert!(board.is_square_attacked_with_occ(Square::E3, Side::White, occ));
        assert!(!board.is_square_attacked_with_occ(Square::E4, Side::White, occ));

        // Knight, king and slider branches with a custom occupancy.
        let mut custom = BoardState::new();
        custom.add_piece(Square::E1, Side::White, Piece::King, false);
        custom.add_piece(Square::D3, Side::Black, Piece::Knight, false);
        custom.add_piece(Square::D2, Side::Black, Piece::King, false);
        let custom_occ = custom.occupancy();
        assert!(custom.is_square_attacked_with_occ(Square::E1, Side::Black, custom_occ));

        let mut slider = BoardState::new();
        slider.add_piece(Square::E1, Side::White, Piece::King, false);
        slider.add_piece(Square::H4, Side::Black, Piece::Bishop, false);
        slider.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        let slider_occ = slider.occupancy();
        assert!(slider.is_square_attacked_with_occ(Square::E1, Side::Black, slider_occ));
    }

    #[test]
    fn pinned_pieces_and_checkers() {
        use crate::common::helpers::STARTING_FEN;

        let start = BoardState::parse_fen(STARTING_FEN);
        assert!(start.pinned_pieces(Side::White).is_empty());
        assert!(start.checkers(Side::White).is_empty());
        assert!(BoardState::new().pinned_pieces(Side::White).is_empty());
        assert!(BoardState::new().checkers(Side::White).is_empty());

        // White queen on E2 is pinned to the king by a black rook on E8.
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E2, Side::White, Piece::Queen, false);
        board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        assert_eq!(
            board.pinned_pieces(Side::White).0,
            1u64 << Square::E2 as usize
        );
        assert!(board.pinned_pieces(Side::Black).is_empty());

        // Open E file: the same rook gives check.
        let mut check = BoardState::new();
        check.add_piece(Square::E1, Side::White, Piece::King, false);
        check.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        assert!(check.is_in_check(Side::White));
        assert_eq!(check.checkers(Side::White).0, 1u64 << Square::E8 as usize);
    }

    #[test]
    fn check_squares_and_threat_by_lesser() {
        use crate::common::helpers::STARTING_FEN;

        assert_eq!(BoardState::new().check_squares(Side::White), [0; 6]);

        let start = BoardState::parse_fen(STARTING_FEN);
        let squares = start.check_squares(Side::White);
        assert_ne!(squares[Piece::Knight as usize], 0);
        assert_eq!(squares[Piece::King as usize], 0);
        assert_eq!(
            squares[Piece::Queen as usize],
            squares[Piece::Bishop as usize] | squares[Piece::Rook as usize]
        );

        let threats = start.threat_by_lesser(Side::White);
        assert_eq!(threats[Piece::Pawn as usize], 0);
        assert_eq!(threats[Piece::King as usize], 0);
        assert_ne!(threats[Piece::Knight as usize], 0);
        assert_eq!(
            threats[Piece::Queen as usize] & threats[Piece::Rook as usize],
            threats[Piece::Rook as usize]
        );
        assert_eq!(
            threats[Piece::Rook as usize] & threats[Piece::Knight as usize],
            threats[Piece::Knight as usize]
        );
    }

    #[test]
    fn clipped_phase_tracks_material() {
        use crate::common::helpers::STARTING_FEN;

        assert_eq!(BoardState::new().clipped_phase(), 0);
        let start = BoardState::parse_fen(STARTING_FEN);
        assert!(start.clipped_phase() > 0);
    }

    #[test]
    fn pseudo_legal_accepts_generated_moves_and_rejects_obvious_illegal() {
        use crate::common::helpers::STARTING_FEN;
        use crate::common::move_list::MoveList;
        use crate::common::move_type::MoveType;

        let board = BoardState::parse_fen(STARTING_FEN);
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        assert!(!moves.is_empty());
        for entry in moves.iter() {
            assert!(board.is_pseudo_legal(entry.mv), "{:?}", entry.mv);
        }

        assert!(!board.is_pseudo_legal(Move::NO_MOVE));
        // Source == target.
        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E2, MoveType::Quiet)));
        // From an empty square.
        assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));
        // Opponent piece (black pawn on A7, white to move).
        assert!(!board.is_pseudo_legal(Move::new(Square::A7, Square::A6, MoveType::Quiet)));
        // Capturing our own piece (queen takes own pawn).
        assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D2, MoveType::Capture)));
        // Pawn flagged as castle.
        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::Castle)));
        // B1-B3 is not a knight move.
        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::B3, MoveType::Quiet)));
        // B1-C3 lands on an empty square, so the capture flag mismatches.
        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Capture)));
        // Double push without the flag.
        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::Quiet)));
        // Double push with the flag from the starting square.
        assert!(board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::DoublePush)));
        // Single push.
        assert!(board.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));
        // Promotion flag away from the promotion rank.
        assert!(!board.is_pseudo_legal(Move::new(
            Square::E2,
            Square::E3,
            MoveType::QueenPromotion
        )));
    }
}
