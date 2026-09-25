use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks,
    pawn_attacks,
};
use crate::board::history::{History, StateCache};
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

    pub fn update_node_cache(&mut self) {
        let stm = self.side_to_move;
        let them = stm.other();
        let checkers = self.checkers(stm).0;
        let pinned = self.pinned_pieces(stm).0;
        let pinned_them = self.pinned_pieces(them).0;
        let pinners = self.compute_pinners(stm).0;
        let all_threats = self.compute_all_threats(them).0;
        let check_squares = self.check_squares(them);
        let threats_us = self.threat_by_lesser(stm);
        let threats_them = self.threat_by_lesser(them);
        self.history.store_cache(StateCache {
            checkers,
            pinned,
            pinners,
            pinned_them,
            all_threats,
            check_squares,
            threats_us,
            threats_them,
        });
    }

    pub fn refresh_cache(&mut self) {
        self.update_node_cache();
    }

    pub fn ensure_cache_fresh(&mut self) {
        if !self.history.is_cache_valid() {
            self.update_node_cache();
        }
    }

    pub fn cached_checkers(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().checkers;
        }
        self.checkers(self.side_to_move).0
    }

    pub fn cached_pinned(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().pinned;
        }
        self.pinned_pieces(self.side_to_move).0
    }

    pub fn cached_pinners(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().pinners;
        }
        self.compute_pinners(self.side_to_move).0
    }

    pub fn cached_all_threats(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().all_threats;
        }
        self.compute_all_threats(self.side_to_move.other()).0
    }

    pub fn cached_check_squares(&self) -> [u64; 6] {
        if self.history.is_cache_valid() {
            return self.history.current_cache().check_squares;
        }
        self.check_squares(self.side_to_move.other())
    }

    pub fn cached_threats_us(&self) -> [u64; 6] {
        if self.history.is_cache_valid() {
            return self.history.current_cache().threats_us;
        }
        self.threat_by_lesser(self.side_to_move)
    }

    pub fn cached_threats_them(&self) -> [u64; 6] {
        if self.history.is_cache_valid() {
            return self.history.current_cache().threats_them;
        }
        self.threat_by_lesser(self.side_to_move.other())
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

            let stm = board.side_to_move;
            let node_checkers = board.checkers(stm).0;
            let node_pinned = board.pinned_pieces(stm).0;
            assert_eq!(board.is_in_check(stm), node_checkers != 0);

            for m_entry in moves.iter() {
                let m = m_entry.mv;
                let fast_legal = board.is_legal(m);
                let cached_legal = board.is_legal_with(m, node_checkers, node_pinned);

                board.make_move(m);
                let slow_legal = !board.is_in_check(board.side_to_move.other());
                board.unmake_move(m);

                assert_eq!(
                    fast_legal, slow_legal,
                    "Legality mismatch for move {:?} in FEN: {}",
                    m, fen
                );
                assert_eq!(
                    cached_legal, slow_legal,
                    "Cached-legality mismatch for move {:?} in FEN: {}",
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

        assert!(board.is_square_attacked(Square::E3, Side::White));
        assert!(board.is_square_attacked(Square::C3, Side::White));
        assert!(!board.is_square_attacked(Square::E4, Side::White));
        assert!(!board.is_square_attacked(Square::E4, Side::Black));

        let occ = board.occupancy();
        assert!(board.is_square_attacked_with_occ(Square::E3, Side::White, occ));
        assert!(!board.is_square_attacked_with_occ(Square::E4, Side::White, occ));

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

        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E2, Side::White, Piece::Queen, false);
        board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        assert_eq!(
            board.pinned_pieces(Side::White).0,
            1u64 << Square::E2 as usize
        );
        assert!(board.pinned_pieces(Side::Black).is_empty());

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

        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E2, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::A7, Square::A6, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D2, MoveType::Capture)));

        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::Castle)));

        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::B3, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Capture)));

        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::Quiet)));

        assert!(board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::DoublePush)));

        assert!(board.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(
            Square::E2,
            Square::E3,
            MoveType::QueenPromotion
        )));
    }

    #[test]
    fn pseudo_legal_rejects_bad_squares() {
        use crate::common::helpers::STARTING_FEN;

        let board = BoardState::parse_fen(STARTING_FEN);
        assert!(!board.is_pseudo_legal(Move::NO_MOVE));
        assert!(!board.is_pseudo_legal(Move::new(Square::NoSquare, Square::E4, MoveType::Quiet)));
        assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::NoSquare, MoveType::Quiet)));
    }

    #[test]
    fn pawn_pseudo_legal_push_and_double_push_branches() {
        use crate::common::helpers::STARTING_FEN;

        let start = BoardState::parse_fen(STARTING_FEN);
        assert!(start.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

        assert!(!start.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Capture)));

        let mut blocked = BoardState::new();
        blocked.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        blocked.add_piece(Square::E1, Side::White, Piece::King, false);
        blocked.add_piece(Square::E8, Side::Black, Piece::King, false);
        blocked.add_piece(Square::E3, Side::White, Piece::Pawn, false);
        blocked.add_piece(Square::E2, Side::White, Piece::Pawn, false);
        assert!(!blocked.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));
        assert!(!blocked.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::DoublePush)));

        let mut occupied = BoardState::new();
        occupied.add_piece(Square::E2, Side::White, Piece::Pawn, false);
        occupied.add_piece(Square::E3, Side::Black, Piece::Pawn, false);
        assert!(!occupied.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

        let mut target_busy = BoardState::new();
        target_busy.add_piece(Square::E2, Side::White, Piece::Pawn, false);
        target_busy.add_piece(Square::E4, Side::Black, Piece::Knight, false);
        assert!(!target_busy.is_pseudo_legal(Move::new(
            Square::E2,
            Square::E4,
            MoveType::DoublePush
        )));

        let mut mistimed = BoardState::new();
        mistimed.add_piece(Square::E3, Side::White, Piece::Pawn, false);
        assert!(!mistimed.is_pseudo_legal(Move::new(Square::E3, Square::E5, MoveType::DoublePush)));

        let mut black = BoardState::new();
        black.add_piece(Square::E7, Side::Black, Piece::Pawn, false);
        black.side_to_move = Side::Black;
        assert!(black.is_pseudo_legal(Move::new(Square::E7, Square::E6, MoveType::Quiet)));
        assert!(black.is_pseudo_legal(Move::new(Square::E7, Square::E5, MoveType::DoublePush)));

        let mut broken = BoardState::new();
        broken.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        broken.piece_mapping[Square::E4 as usize] = Piece::None;
        assert!(!broken.is_pseudo_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));
    }

    #[test]
    fn pawn_pseudo_legal_captures_promotions_and_en_passant() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        board.add_piece(Square::D5, Side::Black, Piece::Pawn, false);
        assert!(board.is_pseudo_legal(Move::new(Square::E4, Square::D5, MoveType::Capture)));
        assert!(board.is_pseudo_legal(Move::new(Square::E4, Square::D5, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::E6, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::F5, MoveType::Capture)));

        let mut promo = BoardState::new();
        promo.add_piece(Square::E7, Side::White, Piece::Pawn, false);
        assert!(promo.is_pseudo_legal(Move::new(Square::E7, Square::E8, MoveType::QueenPromotion)));
        assert!(!promo.is_pseudo_legal(Move::new(Square::E7, Square::E8, MoveType::Quiet)));
        assert!(!promo.is_pseudo_legal(Move::new(
            Square::E7,
            Square::E8,
            MoveType::QueenPromotionCapture
        )));
        let mut black_promo = BoardState::new();
        black_promo.add_piece(Square::E2, Side::Black, Piece::Pawn, false);
        black_promo.side_to_move = Side::Black;
        assert!(black_promo.is_pseudo_legal(Move::new(
            Square::E2,
            Square::E1,
            MoveType::QueenPromotion
        )));
        assert!(!black_promo.is_pseudo_legal(Move::new(Square::E2, Square::E1, MoveType::Quiet)));

        let mut promo_cap = BoardState::new();
        promo_cap.add_piece(Square::F7, Side::White, Piece::Pawn, false);
        promo_cap.add_piece(Square::G8, Side::Black, Piece::Rook, false);
        assert!(promo_cap.is_pseudo_legal(Move::new(
            Square::F7,
            Square::G8,
            MoveType::QueenPromotionCapture
        )));

        let ep = BoardState::parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
        assert!(ep.is_pseudo_legal(Move::new(Square::E5, Square::D6, MoveType::EnPassant)));
        let mut no_ep = BoardState::new();
        no_ep.add_piece(Square::E5, Side::White, Piece::Pawn, false);
        assert!(!no_ep.is_pseudo_legal(Move::new(Square::E5, Square::D6, MoveType::EnPassant)));
    }

    #[test]
    fn knight_pseudo_legal_branches() {
        let mut board = BoardState::new();
        board.add_piece(Square::B1, Side::White, Piece::Knight, false);
        board.add_piece(Square::C3, Side::Black, Piece::Pawn, false);
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E8, Side::Black, Piece::King, false);
        assert!(board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Capture)));

        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::B3, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Castle)));
        assert!(!board.is_pseudo_legal(Move::new(
            Square::B1,
            Square::A3,
            MoveType::KnightPromotion
        )));
    }

    #[test]
    fn slider_pseudo_legal_branches() {
        let mut board = BoardState::new();
        board.add_piece(Square::C1, Side::White, Piece::Bishop, false);
        board.add_piece(Square::D2, Side::White, Piece::Pawn, false);
        board.add_piece(Square::A1, Side::White, Piece::Rook, false);
        board.add_piece(Square::A2, Side::White, Piece::Pawn, false);
        board.add_piece(Square::D1, Side::White, Piece::Queen, false);
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E8, Side::Black, Piece::King, false);

        assert!(!board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Quiet)));
        assert!(!board.is_pseudo_legal(Move::new(Square::A1, Square::A4, MoveType::Quiet)));
        assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D3, MoveType::Quiet)));

        assert!(!board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Castle)));
        assert!(!board.is_pseudo_legal(Move::new(Square::A1, Square::A3, MoveType::RookPromotion)));
        assert!(!board.is_pseudo_legal(Move::new(
            Square::D1,
            Square::D2,
            MoveType::QueenPromotion
        )));

        board.remove_piece(Square::D2, false);
        board.add_piece(Square::E3, Side::Black, Piece::Pawn, false);
        assert!(board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Capture)));
        assert!(!board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Quiet)));

        board.remove_piece(Square::A2, false);
        board.add_piece(Square::A4, Side::Black, Piece::Rook, false);
        assert!(board.is_pseudo_legal(Move::new(Square::A1, Square::A4, MoveType::Capture)));
        assert!(board.is_pseudo_legal(Move::new(Square::A1, Square::A2, MoveType::Quiet)));

        board.remove_piece(Square::D2, false);
        board.add_piece(Square::D3, Side::Black, Piece::Knight, false);
        assert!(board.is_pseudo_legal(Move::new(Square::D1, Square::D3, MoveType::Capture)));
        assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D4, MoveType::Quiet)));
    }

    #[test]
    fn king_pseudo_legal_castle_branches() {
        use crate::common::helpers::STARTING_FEN;

        let mut open = BoardState::new();
        open.add_piece(Square::E1, Side::White, Piece::King, false);
        open.add_piece(Square::E8, Side::Black, Piece::King, false);
        assert!(!open.is_pseudo_legal(Move::new(Square::E1, Square::E2, MoveType::QueenPromotion)));

        assert!(!open.is_pseudo_legal(Move::new(Square::E1, Square::E3, MoveType::Quiet)));

        assert!(!open.is_pseudo_legal(Move::new(Square::E1, Square::E2, MoveType::Capture)));
        assert!(open.is_pseudo_legal(Move::new(Square::E1, Square::E2, MoveType::Quiet)));

        let mut off = BoardState::new();
        off.add_piece(Square::E2, Side::White, Piece::King, false);
        off.add_piece(Square::E8, Side::Black, Piece::King, false);
        off.castle = Castle::WHITE_SHORT;
        assert!(!off.is_pseudo_legal(Move::new(Square::E2, Square::G2, MoveType::Castle)));
        let home = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        assert!(!home.is_pseudo_legal(Move::new(Square::E1, Square::F1, MoveType::Castle)));

        let no_rights = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1");
        assert!(!no_rights.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        let start = BoardState::parse_fen(STARTING_FEN);
        assert!(!start.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(!start.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));

        let attacked =
            BoardState::parse_fen("r3k2r/pppppppp/8/8/2b5/8/PPPP1PPP/R3K2R w KQkq - 0 1");
        assert!(!attacked.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(attacked.is_pseudo_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));

        assert!(home.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(home.is_pseudo_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));
        let black = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
        assert!(black.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
        assert!(black.is_pseudo_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));
    }

    #[test]
    fn is_legal_rejects_null_moves_and_kingless_boards() {
        use crate::common::move_list::MoveList;

        let board = BoardState::starting_position();
        assert!(!board.is_legal(Move::NO_MOVE));
        assert!(!board.is_legal(Move::new(Square::NoSquare, Square::E4, MoveType::Quiet)));
        assert!(!board.is_legal(Move::new(Square::E2, Square::NoSquare, MoveType::Quiet)));

        let mut kingless = BoardState::new();
        kingless.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        assert!(!kingless.is_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));

        let mut moves = MoveList::new();
        kingless.generate_moves(&mut moves);
        for entry in moves.iter() {
            assert!(!kingless.is_legal(entry.mv));
        }
    }

    #[test]
    fn is_legal_castling_needs_rook_rights_and_quiet_path() {
        let white = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        assert!(white.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(white.is_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));
        let black = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
        assert!(black.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
        assert!(black.is_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));

        let mut rookless = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w K - 0 1");
        rookless.remove_piece(Square::H1, false);
        let castle = Move::new(Square::E1, Square::G1, MoveType::Castle);
        assert!(rookless.is_pseudo_legal(castle));
        assert!(!rookless.is_legal(castle));

        let mut off_home = BoardState::new();
        off_home.add_piece(Square::E2, Side::White, Piece::King, false);
        off_home.add_piece(Square::E8, Side::Black, Piece::King, false);
        off_home.castle = Castle::WHITE_SHORT;
        assert!(!off_home.is_legal(Move::new(Square::E2, Square::G2, MoveType::Castle)));
        assert!(!white.is_legal(Move::new(Square::E1, Square::F1, MoveType::Castle)));
        let no_rights = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1");
        assert!(!no_rights.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(!BoardState::starting_position().is_legal(Move::new(
            Square::E1,
            Square::G1,
            MoveType::Castle
        )));
    }

    #[test]
    fn is_legal_castling_respects_checks() {
        let in_check = BoardState::parse_fen("4r1k1/8/8/8/8/8/8/R3K2R w K - 0 1");
        assert!(in_check.is_in_check(Side::White));
        assert!(!in_check.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));

        let transit = BoardState::parse_fen("r3k2r/pppppppp/8/8/2b5/8/PPPP1PPP/R3K2R w KQkq - 0 1");
        assert!(!transit.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(transit.is_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));

        let dest = BoardState::parse_fen("4k3/8/8/8/8/8/7b/R3K2R w K - 0 1");
        assert!(!dest.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
        assert!(dest.is_legal(Move::new(Square::E1, Square::F1, MoveType::Quiet)));
    }

    #[test]
    fn is_legal_king_steps_out_of_check() {
        let board = BoardState::parse_fen("4r1k1/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert!(board.is_in_check(Side::White));

        assert!(board.is_legal(Move::new(Square::E1, Square::F1, MoveType::Quiet)));

        assert!(!board.is_legal(Move::new(Square::E1, Square::E2, MoveType::Quiet)));
    }

    #[test]
    fn is_legal_en_passant_pin_and_precheck_branches() {
        let free = BoardState::parse_fen("4k3/8/8/3Pp3/4K3/8/8/8 w - e6 0 1");
        let mv = Move::new(Square::D5, Square::E6, MoveType::EnPassant);
        assert!(free.is_pseudo_legal(mv));
        assert!(free.is_legal(mv));

        let rook = BoardState::parse_fen("7k/8/8/r2Pp2K/8/8/8/8 w - e6 0 1");
        let rm = Move::new(Square::D5, Square::E6, MoveType::EnPassant);
        assert!(rook.is_pseudo_legal(rm));
        assert!(!rook.is_legal(rm));

        let bishop = BoardState::parse_fen("4k3/1b6/8/3Pp3/4K3/8/8/8 w - e6 0 1");
        assert!(!bishop.is_legal(Move::new(Square::D5, Square::E6, MoveType::EnPassant)));

        let knight = BoardState::parse_fen("4k3/8/3n4/3Pp3/4K3/8/8/8 w - e6 0 1");
        assert!(knight.is_in_check(Side::White));
        assert!(!knight.is_legal(Move::new(Square::D5, Square::E6, MoveType::EnPassant)));

        let pawn = BoardState::parse_fen("4k3/8/8/3Ppp2/4K3/8/8/8 w - e6 0 1");
        assert!(!pawn.is_legal(Move::new(Square::D5, Square::E6, MoveType::EnPassant)));
    }

    #[test]
    fn is_legal_single_and_double_check() {
        let dbl = BoardState::parse_fen("4rk2/8/8/8/1b6/8/8/4K1N1 w - - 0 1");
        assert!(dbl.is_in_check(Side::White));
        assert!(!dbl.is_legal(Move::new(Square::G1, Square::F3, MoveType::Quiet)));
        assert!(dbl.is_legal(Move::new(Square::E1, Square::F1, MoveType::Quiet)));

        let pin = BoardState::parse_fen("4r1k1/8/8/8/8/8/4Q3/4K3 w - - 0 1");
        assert!(!pin.is_in_check(Side::White));
        assert!(pin.is_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));
        assert!(!pin.is_legal(Move::new(Square::E2, Square::D3, MoveType::Quiet)));
        assert!(pin.is_legal(Move::new(Square::E2, Square::E8, MoveType::Capture)));

        let block = BoardState::parse_fen("4k3/8/8/8/1b6/8/P1P5/4K3 w - - 0 1");
        assert!(block.is_legal(Move::new(Square::C2, Square::C3, MoveType::Quiet)));
        assert!(!block.is_legal(Move::new(Square::A2, Square::A3, MoveType::Quiet)));
    }

    #[test]
    fn square_attacked_covers_every_piece_type() {
        let start = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        assert!(start.is_square_attacked(Square::E3, Side::White));

        let knight = BoardState::parse_fen("4k3/8/8/8/8/3n4/8/4K3 w - - 0 1");
        assert!(knight.is_square_attacked(Square::E1, Side::Black));
        let king = BoardState::parse_fen("4k3/8/8/8/8/8/3k4/4K3 w - - 0 1");
        assert!(king.is_square_attacked(Square::E1, Side::Black));
        let queen = BoardState::parse_fen("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1");
        assert!(queen.is_square_attacked(Square::E1, Side::Black));

        let bare = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!(!bare.is_square_attacked(Square::E4, Side::Black));
        assert!(!bare.is_square_attacked(Square::E4, Side::White));
    }

    #[test]
    fn checkers_cover_all_piece_types() {
        let pawn = BoardState::parse_fen("4k3/8/8/3p4/4K3/8/8/8 w - - 0 1");
        assert_eq!(pawn.checkers(Side::White).0, 1u64 << Square::D5 as usize);

        let knight = BoardState::parse_fen("4k3/8/3n4/8/4K3/8/8/8 w - - 0 1");
        assert_eq!(knight.checkers(Side::White).0, 1u64 << Square::D6 as usize);

        let bishop = BoardState::parse_fen("4k3/1b6/8/8/4K3/8/8/8 w - - 0 1");
        assert_eq!(bishop.checkers(Side::White).0, 1u64 << Square::B7 as usize);

        let queen = BoardState::parse_fen("4q3/8/8/8/4K3/8/8/8 w - - 0 1");
        assert_eq!(queen.checkers(Side::White).0, 1u64 << Square::E8 as usize);
    }

    #[test]
    fn pinned_pieces_with_two_blockers_is_empty() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E2, Side::White, Piece::Pawn, false);
        board.add_piece(Square::E3, Side::White, Piece::Pawn, false);
        board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
        assert!(board.pinned_pieces(Side::White).is_empty());

        let mut diag = BoardState::new();
        diag.add_piece(Square::E1, Side::White, Piece::King, false);
        diag.add_piece(Square::D2, Side::White, Piece::Bishop, false);
        diag.add_piece(Square::B4, Side::Black, Piece::Bishop, false);
        assert_eq!(
            diag.pinned_pieces(Side::White).0,
            1u64 << Square::D2 as usize
        );
    }

    #[test]
    fn pseudo_legal_black_castle_edges() {
        let mut off = BoardState::new();
        off.add_piece(Square::E7, Side::Black, Piece::King, false);
        off.add_piece(Square::E1, Side::White, Piece::King, false);
        off.side_to_move = Side::Black;
        off.castle = Castle::BLACK_SHORT;
        assert!(!off.is_pseudo_legal(Move::new(Square::E7, Square::G8, MoveType::Castle)));

        let black = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
        assert!(!black.is_pseudo_legal(Move::new(Square::E8, Square::F8, MoveType::Castle)));

        let transit = BoardState::parse_fen("r3k2r/8/8/2B5/8/8/8/4K3 b kq - 0 1");
        assert!(!transit.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
        assert!(transit.is_pseudo_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));

        let dest = BoardState::parse_fen("4k3/8/8/8/8/8/B7/4K3 b k - 0 1");
        assert!(!dest.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
    }

    #[test]
    fn is_legal_black_castling_attack_and_check_branches() {
        let transit = BoardState::parse_fen("r3k2r/8/8/2B5/8/8/8/4K3 b kq - 0 1");
        assert!(!transit.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
        assert!(transit.is_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));

        let dest = BoardState::parse_fen("4k3/8/8/8/8/8/B7/4K3 b k - 0 1");
        assert!(!dest.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));

        let in_check = BoardState::parse_fen("4k3/4Q3/8/8/8/8/8/4K3 b k - 0 1");
        assert!(in_check.is_in_check(Side::Black));
        assert!(!in_check.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));

        let mut off_home = BoardState::new();
        off_home.add_piece(Square::E7, Side::Black, Piece::King, false);
        off_home.add_piece(Square::E1, Side::White, Piece::King, false);
        off_home.side_to_move = Side::Black;
        off_home.castle = Castle::BLACK_SHORT;
        assert!(!off_home.is_legal(Move::new(Square::E7, Square::G8, MoveType::Castle)));
        let home = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
        assert!(!home.is_legal(Move::new(Square::E8, Square::F8, MoveType::Castle)));

        let mut rookless =
            BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b kq - 0 1");
        rookless.remove_piece(Square::A8, false);
        let castle = Move::new(Square::E8, Square::C8, MoveType::Castle);
        assert!(rookless.is_pseudo_legal(castle));
        assert!(!rookless.is_legal(castle));
    }

    #[test]
    fn pin_edge_cases_adjacent_pinner_and_enemy_blocker() {
        let adj = BoardState::parse_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
        assert!(adj.pinned_pieces(Side::White).is_empty());
        assert_eq!(adj.checkers(Side::White).0, 1u64 << Square::E2 as usize);

        let enemy = BoardState::parse_fen("k3r3/8/8/8/8/8/4p3/4K3 w - - 0 1");
        assert!(!enemy.is_in_check(Side::White));
        assert!(enemy.pinned_pieces(Side::White).is_empty());
    }

    #[test]
    fn queen_attacks_and_checkers_on_files_and_diagonals() {
        let file = BoardState::parse_fen("k3q3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert!(file.is_square_attacked(Square::E1, Side::Black));

        let diag = BoardState::parse_fen("4k3/1q6/8/8/4K3/8/8/8 w - - 0 1");
        assert_eq!(diag.checkers(Side::White).0, 1u64 << Square::B7 as usize);
        assert!(diag.is_in_check(Side::White));
    }

    #[test]
    fn passed_pawns_with_multiple_pawns_per_side() {
        let mut board = BoardState::new();
        board.add_piece(Square::E1, Side::White, Piece::King, false);
        board.add_piece(Square::E8, Side::Black, Piece::King, false);
        board.add_piece(Square::A4, Side::White, Piece::Pawn, false);
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        board.add_piece(Square::B5, Side::Black, Piece::Pawn, false);
        board.add_piece(Square::H5, Side::Black, Piece::Pawn, false);

        assert!(!board.is_passed_pawn(Square::A4 as usize, Side::White));
        assert!(board.is_passed_pawn(Square::E4 as usize, Side::White));
        assert_eq!(
            board.get_passed_pawns(Side::White).0,
            1u64 << Square::E4 as usize
        );

        assert!(!board.is_passed_pawn(Square::B5 as usize, Side::Black));
        assert!(board.is_passed_pawn(Square::H5 as usize, Side::Black));
        assert_eq!(
            board.get_passed_pawns(Side::Black).0,
            1u64 << Square::H5 as usize
        );
    }

    #[test]
    fn is_legal_pinned_piece_in_single_knight_check() {
        let board = BoardState::parse_fen("k3r3/8/8/8/8/3n4/2P1Q3/4K3 w - - 0 1");
        assert!(board.is_in_check(Side::White));

        assert!(!board.is_legal(Move::new(Square::E2, Square::D3, MoveType::Capture)));

        assert!(!board.is_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

        assert!(board.is_legal(Move::new(Square::C2, Square::D3, MoveType::Capture)));

        assert!(!board.is_legal(Move::new(Square::C2, Square::C3, MoveType::Quiet)));
    }

    #[test]
    fn remove_piece_on_empty_square_returns_none() {
        let mut board = BoardState::new();
        assert_eq!(board.remove_piece(Square::E4, false), Piece::None);
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        assert_eq!(board.remove_piece(Square::E4, false), Piece::Pawn);
        assert_eq!(board.remove_piece(Square::E4, false), Piece::None);
    }

    #[test]
    fn get_piece_on_covers_pawn_values_and_minor_material() {
        let mut board = BoardState::new();
        board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
        board.add_piece(Square::E5, Side::Black, Piece::Pawn, false);
        board.add_piece(Square::C3, Side::White, Piece::Knight, false);
        assert_eq!(board.get_piece_on(Square::E4), Piece::Pawn as i32);
        assert_eq!(board.get_piece_on(Square::E5), 6 + Piece::Pawn as i32);
        assert!(board.has_non_pawn_material(Side::White));
        assert!(!board.has_non_pawn_material(Side::Black));
    }

    #[test]
    fn pawn_pseudo_legal_black_en_passant_and_blocked_double_push() {
        let ep = BoardState::parse_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
        assert!(ep.is_pseudo_legal(Move::new(Square::D4, Square::E3, MoveType::EnPassant)));
        assert!(!ep.is_pseudo_legal(Move::new(Square::D4, Square::C3, MoveType::EnPassant)));

        let mut blocked = BoardState::new();
        blocked.add_piece(Square::E7, Side::Black, Piece::Pawn, false);
        blocked.add_piece(Square::E6, Side::White, Piece::Pawn, false);
        blocked.side_to_move = Side::Black;
        assert!(!blocked.is_pseudo_legal(Move::new(Square::E7, Square::E5, MoveType::DoublePush)));
        assert!(!blocked.is_pseudo_legal(Move::new(Square::E7, Square::E6, MoveType::Quiet)));
    }

    #[test]
    fn threat_by_lesser_on_bare_and_queen_boards() {
        let bare = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        assert_eq!(bare.threat_by_lesser(Side::White), [0; 6]);

        let queen = BoardState::parse_fen("4k3/8/8/3q4/8/8/8/4K3 w - - 0 1");
        let threats = queen.threat_by_lesser(Side::White);
        assert_eq!(threats[Piece::Pawn as usize], 0);
        assert_eq!(threats[Piece::King as usize], 0);
        assert_ne!(threats[Piece::Rook as usize], 0);
        assert_ne!(threats[Piece::Queen as usize], 0);
        assert_eq!(
            threats[Piece::Queen as usize] & threats[Piece::Rook as usize],
            threats[Piece::Rook as usize]
        );

        assert_ne!(
            threats[Piece::Queen as usize] & (1u64 << Square::D1 as usize),
            0
        );
        assert_ne!(
            threats[Piece::Rook as usize] & (1u64 << Square::A2 as usize),
            0
        );
    }
}
