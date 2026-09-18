use crate::bitboard::Bitboard;
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_queen_attacks_from_table, get_rook_attacks_from_table,
    king_attacks, knight_attacks, pawn_attacks,
};
use crate::board::state::BoardState;
use crate::common::castle::Castle;
use crate::common::move_list::{MoveList, ScoredMove};
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;
use crate::search::iterative_deepening;
use crate::search::search_state::SearchState;
use std::sync::atomic::AtomicBool;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveGenType {
    Captures,
    Quiets,
}

impl BoardState {
    pub fn find_best_move(
        &mut self,
        depth: u8,
        cancellation_token: &AtomicBool,
        debug_mode: &mut bool,
        search_state: &mut SearchState,
        num_threads: usize,
    ) -> Move {
        // Only shortcut on an unconditional tablebase win at halfmove 0:
        // with halfmove > 0 the 50-move rule may convert the win, and
        // losses/draws are always left to normal search. probe/root_move
        // signatures are owned by another team and stay untouched.
        if self.half_move_clock == 0
            && let Some(tb_move) = crate::syzygy::root_move(self)
        {
            search_state.best_move = tb_move;
            search_state.score = crate::syzygy::TB_WIN;
            return tb_move;
        }
        iterative_deepening::search(
            self,
            depth,
            cancellation_token,
            debug_mode,
            search_state,
            num_threads,
        );
        search_state.best_move
    }

    pub fn generate_moves(&self, move_list: &mut MoveList) {
        move_list.clear();
        self.generate_captures_internal(move_list);
        self.generate_quiets_internal(move_list);
    }

    pub fn generate_captures(&self, move_list: &mut MoveList) {
        move_list.clear();
        self.generate_captures_internal(move_list);
    }

    pub fn generate_quiets(&self, move_list: &mut MoveList) {
        move_list.clear();
        self.generate_quiets_internal(move_list);
    }

    fn generate_captures_internal(&self, move_list: &mut MoveList) {
        self.generate_moves_selective(move_list, MoveGenType::Captures);
    }

    fn generate_quiets_internal(&self, move_list: &mut MoveList) {
        self.generate_moves_selective(move_list, MoveGenType::Quiets);
    }

    fn generate_moves_selective(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        self.generate_pawn_moves(move_list, gen_type);
        self.generate_bishop_moves(move_list, gen_type);
        self.generate_knight_moves(move_list, gen_type);
        self.generate_rook_moves(move_list, gen_type);
        self.generate_queen_moves(move_list, gen_type);
        self.generate_king_moves(move_list, gen_type);
    }

    fn generate_pawn_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        let mut bitboard = self.get_pieces(self.side_to_move, Piece::Pawn);
        while bitboard.is_not_empty() {
            let source = bitboard.get_lsb() as usize;
            match gen_type {
                MoveGenType::Quiets => {
                    self.generate_pawn_pushes(source, move_list, gen_type);
                }
                MoveGenType::Captures => {
                    self.generate_en_passants(source, move_list, gen_type);
                    self.generate_pawn_attacks(source, move_list, gen_type);
                    // Quiet queen promotions belong to the captures stage;
                    // add_pawn_move filters out everything else here.
                    self.generate_pawn_pushes(source, move_list, gen_type);
                }
            }
            bitboard.clear_lsb();
        }
    }

    fn generate_pawn_pushes(&self, source: usize, move_list: &mut MoveList, gen_type: MoveGenType) {
        let both_occ = self.occupancy();

        if self.side_to_move == Side::Black {
            if source > 55 {
                return;
            }
            let one_sq = source + 8;
            if both_occ.get_bit(one_sq) != 0 {
                return;
            }
            self.add_pawn_move(source, one_sq, false, false, move_list, gen_type);

            if source >= Square::A7 as usize && source <= Square::H7 as usize {
                let two_sq = one_sq + 8;
                if both_occ.get_bit(two_sq) != 0 {
                    return;
                }
                self.add_pawn_move(source, two_sq, false, true, move_list, gen_type);
            }
        } else {
            if source < 8 {
                return;
            }
            let one_sq = source - 8;
            if both_occ.get_bit(one_sq) != 0 {
                return;
            }
            self.add_pawn_move(source, one_sq, false, false, move_list, gen_type);

            if source >= Square::A2 as usize && source <= Square::H2 as usize {
                let two_sq = one_sq - 8;
                if both_occ.get_bit(two_sq) != 0 {
                    return;
                }
                self.add_pawn_move(source, two_sq, false, true, move_list, gen_type);
            }
        }
    }

    fn generate_en_passants(&self, source: usize, move_list: &mut MoveList, gen_type: MoveGenType) {
        if self.en_passant_square == Square::NoSquare {
            return;
        }
        let ep_bit = 1u64 << (self.en_passant_square as usize);
        let attacks = Bitboard(pawn_attacks()[self.side_to_move as usize][source] & ep_bit);
        if attacks.is_not_empty() {
            let target = attacks.get_lsb() as usize;
            self.add_pawn_move(source, target, true, false, move_list, gen_type);
        }
    }

    fn generate_pawn_attacks(
        &self,
        source: usize,
        move_list: &mut MoveList,
        gen_type: MoveGenType,
    ) {
        let enemy_occ = self.occupancies[self.side_to_move.other()];
        let mut attacks = enemy_occ & pawn_attacks()[self.side_to_move as usize][source];

        while attacks.is_not_empty() {
            let target = attacks.get_lsb() as usize;
            self.add_pawn_move(source, target, false, false, move_list, gen_type);
            attacks.clear_lsb();
        }
    }

    fn generate_bishop_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        let mut bitboard = self.get_pieces(self.side_to_move, Piece::Bishop);
        while bitboard.is_not_empty() {
            let source = bitboard.get_lsb() as usize;
            let attacks = get_bishop_attacks_from_table(Square::from(source), self.occupancy());
            self.add_attacks(source, attacks, move_list, gen_type);
            bitboard.clear_lsb();
        }
    }

    fn generate_knight_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        let mut bitboard = self.get_pieces(self.side_to_move, Piece::Knight);
        while bitboard.is_not_empty() {
            let source = bitboard.get_lsb() as usize;
            let attacks = Bitboard(knight_attacks()[source]);
            self.add_attacks(source, attacks, move_list, gen_type);
            bitboard.clear_lsb();
        }
    }

    fn generate_rook_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        let mut bitboard = self.get_pieces(self.side_to_move, Piece::Rook);
        while bitboard.is_not_empty() {
            let source = bitboard.get_lsb() as usize;
            let attacks = get_rook_attacks_from_table(Square::from(source), self.occupancy());
            self.add_attacks(source, attacks, move_list, gen_type);
            bitboard.clear_lsb();
        }
    }

    fn generate_queen_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        let mut bitboard = self.get_pieces(self.side_to_move, Piece::Queen);
        while bitboard.is_not_empty() {
            let source = bitboard.get_lsb() as usize;
            let attacks = get_queen_attacks_from_table(Square::from(source), self.occupancy());
            self.add_attacks(source, attacks, move_list, gen_type);
            bitboard.clear_lsb();
        }
    }

    fn generate_king_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        let king_bb = self.get_pieces(self.side_to_move, Piece::King);
        if king_bb.is_empty() {
            return;
        }

        let source = king_bb.get_lsb() as usize;
        let attacks = Bitboard(king_attacks()[source]);

        self.add_attacks(source, attacks, move_list, gen_type);
        self.generate_castle_moves(move_list, gen_type);
    }

    fn generate_castle_moves(&self, move_list: &mut MoveList, gen_type: MoveGenType) {
        if gen_type == MoveGenType::Captures {
            return;
        }
        let occ = self.occupancy();

        if self.side_to_move == Side::White {
            if self.castle.contains(Castle::WHITE_SHORT)
                && occ.get_bit(Square::F1 as usize) == 0
                && occ.get_bit(Square::G1 as usize) == 0
                && !self.is_square_attacked(Square::E1, Side::Black)
                && !self.is_square_attacked(Square::F1, Side::Black)
                && !self.is_square_attacked(Square::G1, Side::Black)
            {
                move_list.push(ScoredMove::new(Square::E1, Square::G1, MoveType::Castle));
            }
            if self.castle.contains(Castle::WHITE_LONG)
                && occ.get_bit(Square::D1 as usize) == 0
                && occ.get_bit(Square::C1 as usize) == 0
                && occ.get_bit(Square::B1 as usize) == 0
                && !self.is_square_attacked(Square::E1, Side::Black)
                && !self.is_square_attacked(Square::D1, Side::Black)
                && !self.is_square_attacked(Square::C1, Side::Black)
            {
                move_list.push(ScoredMove::new(Square::E1, Square::C1, MoveType::Castle));
            }
        } else {
            if self.castle.contains(Castle::BLACK_SHORT)
                && occ.get_bit(Square::F8 as usize) == 0
                && occ.get_bit(Square::G8 as usize) == 0
                && !self.is_square_attacked(Square::E8, Side::White)
                && !self.is_square_attacked(Square::F8, Side::White)
                && !self.is_square_attacked(Square::G8, Side::White)
            {
                move_list.push(ScoredMove::new(Square::E8, Square::G8, MoveType::Castle));
            }
            if self.castle.contains(Castle::BLACK_LONG)
                && occ.get_bit(Square::D8 as usize) == 0
                && occ.get_bit(Square::C8 as usize) == 0
                && occ.get_bit(Square::B8 as usize) == 0
                && !self.is_square_attacked(Square::E8, Side::White)
                && !self.is_square_attacked(Square::D8, Side::White)
                && !self.is_square_attacked(Square::C8, Side::White)
            {
                move_list.push(ScoredMove::new(Square::E8, Square::C8, MoveType::Castle));
            }
        }
    }

    fn add_attacks(
        &self,
        source: usize,
        mut attacks: Bitboard,
        move_list: &mut MoveList,
        gen_type: MoveGenType,
    ) {
        while attacks.is_not_empty() {
            let target = attacks.get_lsb() as usize;
            if self.occupancies[self.side_to_move].get_bit(target) == 1 {
                attacks.clear_lsb();
                continue;
            }
            let is_capture = self.is_square_capture(target);
            if gen_type == MoveGenType::Captures && !is_capture {
                attacks.clear_lsb();
                continue;
            }
            if gen_type == MoveGenType::Quiets && is_capture {
                attacks.clear_lsb();
                continue;
            }
            self.add_move_to_moves_list(source, target, move_list);
            attacks.clear_lsb();
        }
    }

    fn add_move_to_moves_list(&self, source: usize, target: usize, move_list: &mut MoveList) {
        let move_type = if self.is_square_capture(target) {
            MoveType::Capture
        } else {
            MoveType::Quiet
        };
        move_list.push(ScoredMove::new(
            Square::from(source),
            Square::from(target),
            move_type,
        ));
    }

    fn add_pawn_move(
        &self,
        source: usize,
        target: usize,
        enpassant: bool,
        double_push: bool,
        move_list: &mut MoveList,
        gen_type: MoveGenType,
    ) {
        let on_rank1 = target >= Square::A1 as usize && target <= Square::H1 as usize;
        let on_rank8 = target >= Square::A8 as usize && target <= Square::H8 as usize;

        if on_rank1 || on_rank8 {
            let capture = self.is_square_capture(target);
            let src = Square::from(source);
            let tgt = Square::from(target);
            const PROMOS: [(MoveType, MoveType, bool); 4] = [
                (
                    MoveType::KnightPromotion,
                    MoveType::KnightPromotionCapture,
                    false,
                ),
                (
                    MoveType::BishopPromotion,
                    MoveType::BishopPromotionCapture,
                    false,
                ),
                (
                    MoveType::RookPromotion,
                    MoveType::RookPromotionCapture,
                    false,
                ),
                (
                    MoveType::QueenPromotion,
                    MoveType::QueenPromotionCapture,
                    true,
                ),
            ];
            for (quiet_mt, cap_mt, is_queen) in PROMOS {
                match gen_type {
                    // The captures stage owns every capture plus quiet
                    // queen promotions (L1 movegen boundary).
                    MoveGenType::Captures => {
                        if !capture && !is_queen {
                            continue;
                        }
                    }
                    // The quiets stage owns quiet underpromotions only.
                    MoveGenType::Quiets => {
                        if capture || is_queen {
                            continue;
                        }
                    }
                }
                move_list.push(ScoredMove::new(
                    src,
                    tgt,
                    if capture { cap_mt } else { quiet_mt },
                ));
            }
        } else if enpassant || double_push {
            if enpassant {
                if gen_type == MoveGenType::Quiets {
                    return;
                }
            } else {
                if gen_type == MoveGenType::Captures {
                    return;
                }
            }
            move_list.push(ScoredMove::new(
                Square::from(source),
                Square::from(target),
                if enpassant {
                    MoveType::EnPassant
                } else {
                    MoveType::DoublePush
                },
            ));
        } else {
            let is_capture = self.is_square_capture(target);
            if gen_type == MoveGenType::Captures && !is_capture {
                return;
            }
            if gen_type == MoveGenType::Quiets && is_capture {
                return;
            }
            self.add_move_to_moves_list(source, target, move_list);
        }
    }

    fn is_square_capture(&self, target: usize) -> bool {
        self.occupancies[self.side_to_move.other()].get_bit(target) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::{ADVANCED_MOVE_FEN, KIWI_PETE_FEN, STARTING_FEN};

    #[test]
    fn should_generate_correct_move_counts() {
        let starting = BoardState::parse_fen(STARTING_FEN);
        let mut move_list = MoveList::new();
        starting.generate_moves(&mut move_list);
        assert_eq!(move_list.len(), 20, "Starting position move count");

        let kiwi = BoardState::parse_fen(KIWI_PETE_FEN);
        let mut move_list2 = MoveList::new();
        kiwi.generate_moves(&mut move_list2);
        assert_eq!(move_list2.len(), 48, "KiwiPete move count");

        let advanced = BoardState::parse_fen(ADVANCED_MOVE_FEN);
        let mut move_list3 = MoveList::new();
        advanced.generate_moves(&mut move_list3);
        assert_eq!(move_list3.len(), 42, "AdvancedMove move count");
    }

    // If the move count ever goes wrong one of these tests usually helps catch edge cases missed
    #[test]
    fn starting_position_has_no_castle_moves() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let castle_count = move_list.iter().filter(|m| m.mv.is_castle()).count();
        assert_eq!(castle_count, 0, "No castling from starting position");
    }

    #[test]
    fn kiwi_pete_has_castle_moves() {
        let board = BoardState::parse_fen(KIWI_PETE_FEN);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let castle_count = move_list.iter().filter(|m| m.mv.is_castle()).count();
        assert_eq!(
            castle_count, 2,
            "KiwiPete should have exactly 2 castling options"
        );
    }

    #[test]
    fn advanced_fen_has_promotion_moves() {
        let board = BoardState::parse_fen(ADVANCED_MOVE_FEN);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let promo_count = move_list.iter().filter(|m| m.mv.is_promotion()).count();
        assert_eq!(
            promo_count, 12,
            "AdvancedMove FEN should have exactly 12 promotions"
        );
    }

    #[test]
    fn advanced_fen_has_en_passant_move() {
        let board = BoardState::parse_fen(ADVANCED_MOVE_FEN);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let ep_count = move_list
            .iter()
            .filter(|m| m.mv.move_type == MoveType::EnPassant)
            .count();
        assert_eq!(
            ep_count, 1,
            "AdvancedMove FEN should have exactly 1 en passant"
        );
    }

    #[test]
    fn captures_and_quiets_partition_all_moves() {
        for fen in [STARTING_FEN, KIWI_PETE_FEN, ADVANCED_MOVE_FEN] {
            let board = BoardState::parse_fen(fen);
            let mut all = MoveList::new();
            board.generate_moves(&mut all);
            let mut captures = MoveList::new();
            board.generate_captures(&mut captures);
            let mut quiets = MoveList::new();
            board.generate_quiets(&mut quiets);
            assert_eq!(
                captures.len() + quiets.len(),
                all.len(),
                "staged generation must partition {fen}"
            );
            assert!(
                captures
                    .iter()
                    .all(|m| m.mv.is_capture() || m.mv.move_type == MoveType::QueenPromotion)
            );
            assert!(quiets.iter().all(|m| !m.mv.is_capture()));
        }
        let start = BoardState::parse_fen(STARTING_FEN);
        let mut captures = MoveList::new();
        start.generate_captures(&mut captures);
        assert!(captures.is_empty());
        let mut quiets = MoveList::new();
        start.generate_quiets(&mut quiets);
        assert_eq!(quiets.len(), 20);
    }

    #[test]
    fn blocked_pawn_pushes_generate_no_moves() {
        // White pawn on E2 is stopped by the black pawn on E3.
        let board = BoardState::parse_fen("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1");
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        assert!(
            moves.iter().all(|m| m.mv.source != Square::E2),
            "blocked E2 pawn must not move"
        );
        // Black pawn on E2 is stopped by the white king on E1.
        let black = BoardState::parse_fen("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1");
        let mut black_moves = MoveList::new();
        black.generate_moves(&mut black_moves);
        assert!(
            black_moves.iter().all(|m| m.mv.source != Square::E2),
            "blocked black E2 pawn must not move"
        );
    }

    #[test]
    fn kingless_board_generates_pawn_moves_but_no_castles() {
        let board = BoardState::parse_fen("8/8/8/8/8/8/PP6/8 w - - 0 1");
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        assert!(!moves.is_empty());
        assert!(
            moves.iter().all(|m| !m.mv.is_castle()),
            "no king means no castling"
        );
    }

    #[test]
    fn short_castle_blocked_by_attack_but_long_available() {
        // Black bishop on C4 attacks F1 through the empty D3/E2 squares.
        let board = BoardState::parse_fen("r3k2r/pppppppp/8/8/2b5/8/PPPP1PPP/R3K2R w KQkq - 0 1");
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        let castles: Vec<Move> = moves
            .iter()
            .filter(|m| m.mv.is_castle())
            .map(|m| m.mv)
            .collect();
        assert!(
            castles.iter().all(|m| m.target != Square::G1),
            "short castle through attacked F1 must be absent"
        );
        assert!(
            castles.iter().any(|m| m.target == Square::C1),
            "long castle must still be generated"
        );
    }

    #[test]
    fn en_passant_and_double_push_staging() {
        let board = BoardState::parse_fen(ADVANCED_MOVE_FEN);
        let mut captures = MoveList::new();
        board.generate_captures(&mut captures);
        assert_eq!(
            captures
                .iter()
                .filter(|m| m.mv.move_type == MoveType::EnPassant)
                .count(),
            1
        );
        let mut quiets = MoveList::new();
        board.generate_quiets(&mut quiets);
        assert_eq!(
            quiets
                .iter()
                .filter(|m| m.mv.move_type == MoveType::EnPassant)
                .count(),
            0
        );

        let start = BoardState::parse_fen(STARTING_FEN);
        let mut start_captures = MoveList::new();
        start.generate_captures(&mut start_captures);
        assert_eq!(
            start_captures
                .iter()
                .filter(|m| m.mv.move_type == MoveType::DoublePush)
                .count(),
            0
        );
        let mut start_quiets = MoveList::new();
        start.generate_quiets(&mut start_quiets);
        assert_eq!(
            start_quiets
                .iter()
                .filter(|m| m.mv.move_type == MoveType::DoublePush)
                .count(),
            8
        );
    }

    #[test]
    fn slider_capture_quiet_filtering() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/p7/8/R3K3 w - - 0 1");
        let mut captures = MoveList::new();
        board.generate_captures(&mut captures);
        assert!(
            captures
                .iter()
                .any(|m| m.mv.source == Square::A1 && m.mv.target == Square::A3)
        );
        assert!(
            captures
                .iter()
                .all(|m| !(m.mv.source == Square::A1 && m.mv.target == Square::A2))
        );
        let mut quiets = MoveList::new();
        board.generate_quiets(&mut quiets);
        assert!(
            quiets
                .iter()
                .any(|m| m.mv.source == Square::A1 && m.mv.target == Square::A2)
        );
        assert!(
            quiets
                .iter()
                .all(|m| !(m.mv.source == Square::A1 && m.mv.target == Square::A3))
        );
    }

    #[test]
    fn black_pawn_single_and_double_push() {
        let board = BoardState::parse_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1");
        let mut quiets = MoveList::new();
        board.generate_quiets(&mut quiets);
        assert!(
            quiets
                .iter()
                .any(|m| m.mv.source == Square::E7 && m.mv.target == Square::E6)
        );
        assert!(
            quiets
                .iter()
                .any(|m| m.mv.source == Square::E7 && m.mv.target == Square::E5)
        );
    }
}
