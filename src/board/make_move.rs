use crate::bitboard::Bitboard;
use crate::bitboard::attacks::{FILE_A, FILE_H};
use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, knight_attacks, pawn_attacks,
};
use crate::board::state::{BoardState, CASTLING_CONSTANTS};
use crate::common::castle::Castle;
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;
use crate::common::zobrist;

#[inline(always)]
fn is_light_square(sq: usize) -> bool {
    let file = sq % 8;
    let rank_from_white = 7 - sq / 8;
    (file + rank_from_white) % 2 == 1
}

impl BoardState {
    pub fn make_move(&mut self, m: Move) {
        let current_idx = self.history.index;
        let next_idx = current_idx + 1;
        if crate::eval::nnue::v16::maintenance_active() {
            self.history.sfnn16[next_idx] = self.history.sfnn16[current_idx].clone();
        }

        let captured_piece = Piece::None;
        let original_board_hash = self.board_hash;
        let original_en_passant_square = self.en_passant_square;
        let original_castling_rights = self.castle;
        let original_half_move_clock = self.half_move_clock;

        self.board_hash ^=
            zobrist::zobrist_table()[self.get_piece_on(m.source) as usize][m.source as usize];
        let moved_piece = self.remove_piece(m.source, true);
        if moved_piece == Piece::Pawn || m.is_capture() {
            self.half_move_clock = 0;
        } else {
            self.half_move_clock += 1;
        }

        let mut final_moved_piece = moved_piece;
        let mut final_captured_piece = captured_piece;

        if m.is_capture() {
            final_captured_piece = self.handle_capture(m);
        }

        if m.is_promotion() {
            final_moved_piece = m.move_type.promotion_piece();
        }

        if m.is_castle() {
            self.handle_castle(m);
        }

        self.add_piece(m.target, self.side_to_move, final_moved_piece, true);
        self.board_hash ^=
            zobrist::zobrist_table()[self.get_piece_on(m.target) as usize][m.target as usize];

        self.record_pending_updates(next_idx);
        self.update_castling_rights(m);
        self.update_en_passant(m);
        self.flip_side_to_move();

        self.history.save(
            final_captured_piece,
            original_en_passant_square,
            original_castling_rights,
            original_board_hash,
            original_half_move_clock,
        );
        self.history.plies_from_null = self.history.plies_from_null.saturating_add(1);
        self.move_count += 1;
    }

    fn handle_castle(&mut self, m: Move) {
        match m.target {
            Square::C1 => self.move_rook_from(Square::A1, Square::D1, self.side_to_move),
            Square::G1 => self.move_rook_from(Square::H1, Square::F1, self.side_to_move),
            Square::C8 => self.move_rook_from(Square::A8, Square::D8, self.side_to_move),
            Square::G8 => self.move_rook_from(Square::H8, Square::F8, self.side_to_move),
            _ => {}
        }
    }

    fn handle_capture(&mut self, m: Move) -> Piece {
        let target_square = if m.move_type.is_en_passant() {
            self.en_passant_square_for(m)
        } else {
            m.target
        };

        let piece_idx = self.get_piece_on(target_square);
        if piece_idx != -1 {
            self.board_hash ^= zobrist::zobrist_table()[piece_idx as usize][target_square as usize];
        }
        self.half_move_clock = 0;

        self.remove_piece(target_square, true)
    }

    fn flip_side_to_move(&mut self) {
        self.board_hash = zobrist::flip_side_to_move_hashes(self, self.board_hash);
        self.side_to_move = self.side_to_move.other();
    }

    fn update_en_passant(&mut self, m: Move) {
        self.board_hash = zobrist::hash_en_passant(self, self.board_hash);

        // TODO: this needs to be rethought for proper impl (FEN, and legal en passsnt represent EP square differently)
        // https://www.talkchess.com/forum/viewtopic.php?t=33397
        self.en_passant_square = if m.move_type.is_double_push() {
            let t = m.target as usize;
            let adjacent = ((1u64 << (t - 1)) & !FILE_H) | ((1u64 << (t + 1)) & !FILE_A);
            if (self.get_pieces(self.side_to_move.other(), Piece::Pawn) & adjacent).is_not_empty() {
                self.en_passant_square_for(m)
            } else {
                Square::NoSquare
            }
        } else {
            Square::NoSquare
        };
        self.board_hash = zobrist::hash_en_passant(self, self.board_hash);

        // Drop phantom EP squares: keep the square only if at least one
        // enemy pawn can legally capture en passant (no horizontal pin),
        // cf. Stockfish position.cpp do_move() and Reckless validate_en_passant().
        if self.en_passant_square != Square::NoSquare && !self.has_legal_ep_capture(m) {
            // The hash currently contains the tentative EP key: xor it back
            // out first (hash_en_passant on NoSquare afterwards is a no-op).
            self.board_hash = zobrist::hash_en_passant(self, self.board_hash);
            self.en_passant_square = Square::NoSquare;
        }
    }

    /// True if the opponent (to move after the pending side flip) has at
    /// least one legal en passant capture following double push `m`.
    /// Must be called before the side-to-move flip, i.e. `side_to_move`
    /// is still the pushing side.
    fn has_legal_ep_capture(&self, m: Move) -> bool {
        let ep = self.en_passant_square as usize;
        let mover = self.side_to_move;
        let capturer = mover.other();
        let mut takers =
            self.get_pieces(capturer, Piece::Pawn).0 & pawn_attacks()[mover as usize][ep];
        if takers == 0 {
            return false;
        }
        let king_bb = self.get_pieces(capturer, Piece::King);
        if king_bb.is_empty() {
            return true;
        }
        let ksq = Square::from(king_bb.get_lsb() as usize);
        let cap_sq = m.target as usize;
        let enemy_pawns = self.get_pieces(mover, Piece::Pawn).0 & !(1u64 << cap_sq);
        let enemy_knights = self.get_pieces(mover, Piece::Knight).0;
        let enemy_bq =
            self.get_pieces(mover, Piece::Bishop).0 | self.get_pieces(mover, Piece::Queen).0;
        let enemy_rq =
            self.get_pieces(mover, Piece::Rook).0 | self.get_pieces(mover, Piece::Queen).0;
        while takers != 0 {
            let from = takers.trailing_zeros() as usize;
            takers &= takers - 1;
            let occ =
                Bitboard((self.occupancy().0 ^ (1u64 << from) ^ (1u64 << cap_sq)) | (1u64 << ep));
            if (enemy_pawns & pawn_attacks()[capturer as usize][ksq as usize]) != 0 {
                continue;
            }
            if (enemy_knights & knight_attacks()[ksq as usize]) != 0 {
                continue;
            }
            if (get_bishop_attacks_from_table(ksq, occ).0 & enemy_bq) != 0 {
                continue;
            }
            if (get_rook_attacks_from_table(ksq, occ).0 & enemy_rq) != 0 {
                continue;
            }
            return true;
        }
        false
    }

    fn update_castling_rights(&mut self, m: Move) {
        self.board_hash = zobrist::hash_castling_rights(self, self.board_hash);
        self.castle &= Castle::from_bits_retain(CASTLING_CONSTANTS[m.source as usize]);
        self.castle &= Castle::from_bits_retain(CASTLING_CONSTANTS[m.target as usize]);
        self.board_hash = zobrist::hash_castling_rights(self, self.board_hash);
    }

    fn move_rook_from(&mut self, source: Square, target: Square, side: Side) {
        let rook_index = self.get_piece_on(source);
        self.remove_piece(source, true);
        self.add_piece(target, side, Piece::Rook, true);

        self.board_hash ^= zobrist::zobrist_table()[rook_index as usize][source as usize];
        self.board_hash ^= zobrist::zobrist_table()[rook_index as usize][target as usize];
    }

    pub fn unmake_move(&mut self, m: Move) {
        let history = self.history.restore();

        let moved_piece = self.remove_piece(m.target, false);
        self.side_to_move = self.side_to_move.other();

        if history.captured_piece != Piece::None {
            if m.move_type.is_en_passant() {
                self.add_piece(
                    self.en_passant_square_for(m),
                    self.side_to_move.other(),
                    Piece::Pawn,
                    false,
                );
            } else {
                self.add_piece(
                    m.target,
                    self.side_to_move.other(),
                    history.captured_piece,
                    false,
                );
            }
        }

        if m.is_castle() {
            match m.target {
                Square::C1 => {
                    self.remove_piece(Square::D1, false);
                    self.add_piece(Square::A1, self.side_to_move, Piece::Rook, false);
                }
                Square::G1 => {
                    self.remove_piece(Square::F1, false);
                    self.add_piece(Square::H1, self.side_to_move, Piece::Rook, false);
                }
                Square::C8 => {
                    self.remove_piece(Square::D8, false);
                    self.add_piece(Square::A8, self.side_to_move, Piece::Rook, false);
                }
                Square::G8 => {
                    self.remove_piece(Square::F8, false);
                    self.add_piece(Square::H8, self.side_to_move, Piece::Rook, false);
                }
                _ => {}
            }
        }

        self.add_piece(
            m.source,
            self.side_to_move,
            if m.is_promotion() {
                Piece::Pawn
            } else {
                moved_piece
            },
            false,
        );
        self.half_move_clock = history.half_move_clock;
        self.board_hash = history.board_hash;
        self.castle = history.castling_rights;
        self.en_passant_square = history.en_passant_square;
        self.move_count -= 1;
    }

    fn en_passant_square_for(&self, m: Move) -> Square {
        let offset = if self.side_to_move == Side::Black {
            -8
        } else {
            8
        };
        Square::from((m.target as i32 + offset) as usize)
    }

    /// Insufficient mating material, cf. Reckless board.rs draw_by_material():
    /// KK, K+minor vs K, KNvKN, and KBvKB with same-colour bishops.
    /// KNNvK is deliberately NOT an automatic draw, nor is KBvKN or
    /// KBBvK (mate remains possible with help).
    fn is_insufficient_material(&self) -> bool {
        if (self.pieces[Piece::Pawn] | self.pieces[Piece::Rook] | self.pieces[Piece::Queen])
            .is_not_empty()
        {
            return false;
        }
        let piece_count = self.occupancy().count_ones();
        if piece_count == 2 {
            // KK.
            return true;
        }
        if piece_count == 3 {
            // K+minor vs K (no pawns/majors left, so the third piece is B/N).
            return true;
        }
        if piece_count != 4 {
            return false;
        }
        // Exactly two minor pieces besides the kings.
        let w_minors = (self.get_pieces(Side::White, Piece::Bishop)
            | self.get_pieces(Side::White, Piece::Knight))
        .count_ones();
        if w_minors != 1 {
            // KNNvK / KBBvK: mate is still possible.
            return false;
        }
        let w_bishop = self.get_pieces(Side::White, Piece::Bishop).is_not_empty();
        let b_bishop = self.get_pieces(Side::Black, Piece::Bishop).is_not_empty();
        if w_bishop != b_bishop {
            // KBvKN: mate is still possible.
            return false;
        }
        if !w_bishop {
            // KNvKN.
            return true;
        }
        // KBvKB: draw only when both bishops share the same square colour.
        let bishops = self.pieces[Piece::Bishop].0;
        let b1 = bishops.trailing_zeros() as usize;
        let rest = bishops & bishops.wrapping_sub(1);
        let b2 = rest.trailing_zeros() as usize;
        is_light_square(b1) == is_light_square(b2)
    }

    /// True when the side to move is in check with no legal move.
    /// Checkmate takes precedence over the fifty-move draw
    /// (cf. Stockfish Position::is_draw).
    fn is_checkmated(&self) -> bool {
        if !self.is_in_check(self.side_to_move) {
            return false;
        }
        !self.has_any_legal_move()
    }

    fn has_any_legal_move(&self) -> bool {
        let mut moves = MoveList::new();
        self.generate_moves(&mut moves);
        for m in moves.iter() {
            if self.is_legal(m.mv) {
                return true;
            }
        }
        false
    }

    /// Repetition lookback stops at the last reversible move AND at the
    /// last null move (cf. Stockfish `min(rule50, pliesFromNull)`).
    fn repetition_window(&self) -> usize {
        (self.half_move_clock as usize).min(self.history.plies_from_null)
    }

    pub fn is_draw(&self) -> bool {
        if self.is_insufficient_material() {
            return true;
        }

        if self.half_move_clock >= 100 {
            return !self.is_checkmated();
        }
        if self.half_move_clock <= 7 {
            return false;
        }
        self.history.has_hash_appeared_twice(
            self.board_hash,
            self.history.index.saturating_sub(self.repetition_window()),
        )
    }

    pub fn is_draw_in_search(&self, ply: u16) -> bool {
        if self.is_insufficient_material() {
            return true;
        }

        if self.half_move_clock >= 100 {
            return !self.is_checkmated();
        }

        if self.half_move_clock <= 3 {
            return false;
        }

        let start = self.history.index.saturating_sub(self.repetition_window());
        let mut matches = 0;
        for i in (start..self.history.index).rev() {
            if self.history.entries[i].board_hash == self.board_hash {
                matches += 1;

                if matches == 2 {
                    return true;
                }

                if matches == 1 && i > self.history.index.saturating_sub(ply as usize) {
                    return true;
                }
            }
        }

        false
    }

    pub fn make_null_move(&mut self) {
        let current_idx = self.history.index;
        let next_idx = current_idx + 1;
        self.history.dirty_updates[next_idx] = crate::board::history::DirtyUpdate::default();
        self.history.computed[next_idx] = false;
        if crate::eval::nnue::v16::maintenance_active() {
            self.history.sfnn16[next_idx] = self.history.sfnn16[current_idx].clone();
        }

        self.history.save(
            Piece::None,
            self.en_passant_square,
            self.castle,
            self.board_hash,
            self.half_move_clock,
        );
        self.history.plies_from_null = 0;
        self.update_en_passant(Move::NO_MOVE);
        self.flip_side_to_move();
    }

    pub fn undo_null_move(&mut self) {
        let history = self.history.restore();
        self.flip_side_to_move();
        self.half_move_clock = history.half_move_clock;
        self.board_hash = history.board_hash;
        self.castle = history.castling_rights;
        self.en_passant_square = history.en_passant_square;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;
    use crate::common::helpers::STARTING_FEN;
    use crate::common::move_list::MoveList;
    use crate::common::move_type::MoveType;

    #[test]
    fn test_should_make_and_undo_null_move_correctly() {
        let mut board_state = BoardState::parse_fen(STARTING_FEN);
        let original_state_pieces = board_state.pieces;
        let original_state_side = board_state.side_to_move;
        let original_board_hash = board_state.board_hash;

        board_state.make_null_move();

        assert_eq!(board_state.pieces, original_state_pieces);
        assert_ne!(board_state.side_to_move, original_state_side);
        assert_ne!(board_state.board_hash, original_board_hash);

        board_state.undo_null_move();

        assert_eq!(board_state.pieces, original_state_pieces);
        assert_eq!(board_state.side_to_move, original_state_side);
        assert_eq!(board_state.board_hash, original_board_hash);
    }

    #[test]
    fn test_zobrist_hashing_restore() {
        let cases = vec![
            // Quiet, Captures & Promotions
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "e2e4",
            ),
            (
                "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "e4d5",
            ),
            (
                "rnbqkbnr/ppp2ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 1",
                "d5e6",
            ),
            (
                "rnbqkbnr/ppppp1P1/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1",
                "g7h8q",
            ),
            // En Passant
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
                "d7d5",
            ),
            (
                "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
                "e5d6",
            ),
            (
                "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
                "e5e6",
            ),
            // Castling Rights
            ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"),
            ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8"),
            ("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8c8"),
        ];

        for (fen, move_str) in cases {
            let mut board_state = BoardState::parse_fen(fen);
            let mut move_list = MoveList::new();
            board_state.generate_moves(&mut move_list);

            let parsed_move = Move::parse_long_algebraic(move_str).unwrap();
            let mut found_move = Move::NO_MOVE;
            for m in move_list.iter() {
                if m.mv.source == parsed_move.source
                    && m.mv.target == parsed_move.target
                    && (parsed_move.move_type == MoveType::Quiet
                        || ((m.mv.move_type.value() & !8) == parsed_move.move_type.value()))
                {
                    found_move = m.mv;
                    break;
                }
            }
            assert_ne!(
                found_move,
                Move::NO_MOVE,
                "Move {} not found in FEN {}",
                move_str,
                fen
            );

            let original_hash = board_state.board_hash;

            board_state.make_move(found_move);
            assert_eq!(
                zobrist::get_board_hash(&board_state),
                board_state.board_hash,
                "Incremental hash mismatch after make_move for {} in {}",
                move_str,
                fen
            );

            board_state.unmake_move(found_move);
            assert_eq!(
                original_hash, board_state.board_hash,
                "Hash not restored after unmake_move for {} in {}",
                move_str, fen
            );
            assert_eq!(
                zobrist::get_board_hash(&board_state),
                board_state.board_hash,
                "State hash mismatch after unmake_move for {} in {}",
                move_str,
                fen
            );
        }
    }

    #[test]
    fn test_is_draw_fifty_move_rule() {
        let moves_str = "d2d4 g8f6 g1f3 g7g6 c1f4 d7d6 b1d2 f6h5 f4g5 f7f6 g5e3 e7e5 d4d5 f8e7 e3h6 c7c6 e2e4 b8d7 d5c6 b7c6 f1c4 \
                       c8b7 d2b3 a7a5 e1g1 a5a4 b3d2 d6d5 c4d3 d7c5 a1b1 h5f4 h6f4 e5f4 d1e2 d8d7 f1e1 c5d3 e2d3 e8g8 b1d1 a8e8 \
                       d3d4 c6c5 d4d3 d7e6 e4d5 e6d5 d3a3 b7c6 h2h3 g8g7 g1h2 f8f7 a3c3 e8d8 c3c4 d5c4 d2c4 h7h6 d1d8 e7d8 f3d2 f7e7 \
                       e1d1 c6d5 c4d6 d5a2 d2e4 e7e5 e4c3 a2g8 c3a4 d8e7 d6c8 e7f8 d1d7 g8f7 a4c3 g6g5 c3b5 g7g8 d7d2 e5d5 d2d5 f7d5 \
                       c8d6 g8h7 g2g3 h7g6 g3g4 d5c6 h2g1 h6h5 g4h5 g6h5 g1h2 c6d7 h2g2 h5h4 b2b3 f8e7 c2c4 d7h3 g2f3 f6f5 f3e2 h4g4 d6f7 \
                       e7f6 f7h6 g4h5 h6f7 g5g4 f7d6 h5g6 d6b7 f6e7 b5c3 h3g2 c3d5 g2f3 e2d2 f3d5 c4d5 g4g3 f2g3 f4g3 d2e2 e7h4 e2f3 g6f6 b7c5 \
                       f6e5 c5d3 e5d5 d3f4 d5d4 f4e2 d4e5 b3b4 g3g2 b4b5 h4d8 f3g2 e5e4 e2g3 e4f4 g3h5 f4g4 h5g3 f5f4 g3e4 f4f3 g2f1 \
                       g4f5 e4d6 f5f4 d6c4 f4e4 b5b6 e4d5 b6b7 d8c7 c4e3 d5c6 f1f2 c6b7 f2f3 b7c6 f3e4 c6c5 e3d5 c7a5 d5e7 a5b4 \
                       e7g8 b4d2 g8e7 d2b4 e7g8 b4d2 g8f6 d2e1 f6e8 e1c3 e8c7 c3f6 c7e8 f6b2 e8c7 b2f6 c7e6 c5d6 e6d4 d6c5 d4e6 c5c6 \
                       e6d4 c6d6 d4b5 d6c5 b5c7 f6b2 c7a6 c5c4 a6c7 c4c5 c7e6 c5d6 e6g5 b2a1 g5f7 d6c5 f7d8 c5c4 d8f7 c4c5 f7h6 a1c3 \
                       h6f5 c3f6 f5h6 f6c3 h6f5 c3f6 f5e3 f6g7 e3d5 c5d6 d5b4 g7f8 b4d5 f8g7 d5b4 g7f8 b4d5 f8h6 d5b6 h6g7 b6c8 d6c5 c8e7 \
                       g7b2 e7g8 c5d6 g8h6 d6c6 h6g8 c6d7 g8h6 d7e6 h6f5 b2a1 f5h6 e6d6 h6f5 d6c5 f5h4 a1f6 h4f3 f6c3 f3e5 c5d6 e5f3 c3f6 \
                       f3d4 f6e5";
        let moves: Vec<&str> = moves_str.split_whitespace().collect();
        let mut board_state = BoardState::default();
        for move_str in moves {
            let mut move_list = MoveList::new();
            board_state.generate_moves(&mut move_list);
            let parsed_move = Move::parse_long_algebraic(move_str)
                .unwrap_or_else(|| panic!("Failed to parse move: '{}'", move_str));
            let mut found_move = Move::NO_MOVE;
            for m in move_list.iter() {
                if m.mv.source == parsed_move.source && m.mv.target == parsed_move.target {
                    found_move = m.mv;
                    break;
                }
            }
            board_state.make_move(found_move);
        }
        assert!(!board_state.is_draw());
        let fifty_move = Move::new(Square::D4, Square::C2, MoveType::Quiet);
        board_state.make_move(fifty_move);
        assert!(board_state.is_draw());
    }

    #[test]
    fn test_threefold_with_intervening_pawn_moves() {
        let moves_str = "d2d4 e7e6 g1f3 g8f6 c1f4 c7c5 c2c3 c5d4 c3d4 d8b6 d1c2 b8c6 e2e3 c6b4 c2b3 b4d5 b3b6 d5b6 b1c3 f6d5 f4e5 d5c3 b2c3 f7f6 e5g3 b6d5 a1c1 f8a3 c1c2 e8g8 e3e4 d5e7 f1d3 d7d5 e1g1 d5e4 d3e4 f6f5 e4d3 f5f4 g3h4 a3d6 f3e5 e7f5 h4g5 h7h6 g5f4 f5d4 c3d4 f8f4 f1c1 f4f8 d3e4 f8d8 e4g6 d8f8 g6f7 g8h7 f7g6 h7g8 a2a4 a7a5 g6f7 g8h7 f7g6 h7g8 g6f7 g8h7 f7g6 h7g8";
        let moves: Vec<&str> = moves_str.split_whitespace().collect();
        let mut board = BoardState::default();
        for (i, move_str) in moves.iter().enumerate() {
            let mut move_list = MoveList::new();
            board.generate_moves(&mut move_list);
            let parsed = Move::parse_long_algebraic(move_str).unwrap();
            let mut found = Move::NO_MOVE;
            for m in move_list.iter() {
                if m.mv.source == parsed.source && m.mv.target == parsed.target {
                    found = m.mv;
                    break;
                }
            }
            assert_ne!(found, Move::NO_MOVE, "Move {} ({}) not found", i, move_str);
            board.make_move(found);
        }
        assert!(
            board.is_draw(),
            "Should detect threefold repetition after the full sequence"
        );
    }

    #[test]
    fn test_is_draw_threefold_repetition() {
        let mut board = BoardState::default();

        let nf3 = Move::new(Square::G1, Square::F3, MoveType::Quiet);
        let nf6 = Move::new(Square::G8, Square::F6, MoveType::Quiet);
        let ng1 = Move::new(Square::F3, Square::G1, MoveType::Quiet);
        let ng8 = Move::new(Square::F6, Square::G8, MoveType::Quiet);

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);
        assert!(!board.is_draw());

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);
        assert!(board.is_draw());
    }

    #[test]
    fn test_should_not_detect_threefold_repetition_when_moves_are_different() {
        let mut board = BoardState::default();

        let nf3 = Move::new(Square::G1, Square::F3, MoveType::Quiet);
        let nf6 = Move::new(Square::G8, Square::F6, MoveType::Quiet);
        let ne5 = Move::new(Square::F3, Square::E5, MoveType::Quiet);
        let ne4 = Move::new(Square::F6, Square::E4, MoveType::Quiet);
        let back_nf3 = Move::new(Square::E5, Square::F3, MoveType::Quiet);
        let back_nf6 = Move::new(Square::E4, Square::F6, MoveType::Quiet);

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ne5);
        board.make_move(ne4);
        board.make_move(back_nf3);
        board.make_move(back_nf6);

        assert!(!board.is_draw());
    }

    #[test]
    fn test_reset_repetition_count_after_pawn_move() {
        let mut board = BoardState::default();
        let nf3 = Move::new(Square::G1, Square::F3, MoveType::Quiet);
        let nf6 = Move::new(Square::G8, Square::F6, MoveType::Quiet);
        let ng1 = Move::new(Square::F3, Square::G1, MoveType::Quiet);
        let ng8 = Move::new(Square::F6, Square::G8, MoveType::Quiet);
        let e4 = Move::new(Square::E2, Square::E4, MoveType::DoublePush);

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);
        assert!(!board.is_draw());

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(e4);
        board.make_move(ng8);
        assert!(!board.is_draw());

        board.make_move(ng1);
        board.make_move(nf6);
        board.make_move(nf3);
        board.make_move(ng8);
        assert!(!board.is_draw());
    }

    #[test]
    fn test_reset_repetition_count_after_capture() {
        let mut board = BoardState::default();
        let e4 = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        let d5 = Move::new(Square::D7, Square::D5, MoveType::DoublePush);
        let dxe4 = Move::new(Square::D5, Square::E4, MoveType::Capture);

        let nf3 = Move::new(Square::G1, Square::F3, MoveType::Quiet);
        let nf6 = Move::new(Square::G8, Square::F6, MoveType::Quiet);
        let ng1 = Move::new(Square::F3, Square::G1, MoveType::Quiet);
        let ng8 = Move::new(Square::F6, Square::G8, MoveType::Quiet);

        board.make_move(e4);
        board.make_move(d5);

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);
        assert!(!board.is_draw());

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        assert!(!board.is_draw());

        board.make_move(dxe4);

        board.make_move(nf3);
        board.make_move(ng8);
        board.make_move(ng1);
        board.make_move(nf6);
        assert!(!board.is_draw());

        board.make_move(nf3);
        board.make_move(ng8);
        board.make_move(ng1);
        board.make_move(nf6);
        assert!(board.is_draw());
    }

    #[test]
    fn test_is_draw_insufficient_material() {
        // KvK
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/4K2k w - - 0 1");
        assert!(board.is_draw());

        // KvKB
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/4K1Bk w - - 0 1");
        assert!(board.is_draw());

        // KvKN
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/4K1Nk w - - 0 1");
        assert!(board.is_draw());

        // KBvK
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/4K1bk w - - 0 1");
        assert!(board.is_draw());

        // KNvK
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/4K1nk w - - 0 1");
        assert!(board.is_draw());

        // KvKBB (not a forced draw)
        let board = BoardState::parse_fen("8/8/8/8/8/8/8/4KBBk w - - 0 1");
        assert!(!board.is_draw());

        // KvP
        let board = BoardState::parse_fen("8/8/8/8/8/8/4P3/4K2k w - - 0 1");
        assert!(!board.is_draw());

        // KvR
        let board = BoardState::parse_fen("8/8/8/8/8/8/4R3/4K2k w - - 0 1");
        assert!(!board.is_draw());

        // KvQ
        let board = BoardState::parse_fen("8/8/8/8/8/8/4Q3/4K2k w - - 0 1");
        assert!(!board.is_draw());
    }

    #[test]
    fn should_not_reset_half_move_clock_after_castle() {
        let mut board = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let original_half_move_clock = board.half_move_clock;
        board.make_move(Move::new(Square::E1, Square::G1, MoveType::Castle));
        assert_eq!(board.half_move_clock, original_half_move_clock + 1);
    }

    #[test]
    fn should_reset_half_move_clock_after_en_passant() {
        let mut board =
            BoardState::parse_fen("rnbqkbnr/pppp1ppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2");

        // Quiet Moves which won't update draw killer
        board.make_move(Move {
            source: Square::B7,
            target: Square::C6,
            move_type: MoveType::Quiet,
        });
        board.make_move(Move {
            source: Square::B1,
            target: Square::C3,
            move_type: MoveType::Quiet,
        });
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let double_push = move_list
            .iter()
            .copied()
            .find(|m| m.mv.source == Square::F7 && m.mv.target == Square::F5)
            .expect("f7f5 double push must exist");
        board.make_move(double_push.mv);
        assert_eq!(board.en_passant_square, Square::F6);
        assert_eq!(board.half_move_clock, 0);
    }

    #[test]
    fn test_is_draw_in_search_twofold_repetition() {
        let mut board = BoardState::default();

        let nf3 = Move::new(Square::G1, Square::F3, MoveType::Quiet);
        let nf6 = Move::new(Square::G8, Square::F6, MoveType::Quiet);
        let ng1 = Move::new(Square::F3, Square::G1, MoveType::Quiet);
        let ng8 = Move::new(Square::F6, Square::G8, MoveType::Quiet);

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);

        // At this point, the initial position has occurred twice.
        // It is a 2-fold repetition at/before the root (ply = 0), so it should not be a draw.
        assert!(!board.is_draw_in_search(0));
        assert!(!board.is_draw_in_search(4));

        board.make_move(nf3);
        board.make_move(nf6);
        board.make_move(ng1);
        board.make_move(ng8);

        // Threefold repetition (3rd occurrence) is always a draw.
        assert!(board.is_draw_in_search(0));
        assert!(board.is_draw_in_search(8));

        // Test cycle strictly after the root
        let mut board = BoardState::default();
        let e3 = Move::new(Square::E2, Square::E3, MoveType::Quiet);
        let nh6 = Move::new(Square::G8, Square::H6, MoveType::Quiet);
        let ng8 = Move::new(Square::H6, Square::G8, MoveType::Quiet);

        board.make_move(e3); // index = 1

        board.make_move(nh6); // index = 2
        board.make_move(nf3); // index = 3
        board.make_move(ng8); // index = 4
        board.make_move(ng1); // index = 5

        // The position after e3 occurred at index 1 and index 5.
        // If the search root is at index 0 (ply = 5), the first occurrence (index 1) is after the root.
        assert!(board.is_draw_in_search(5));

        // If the search root is at index 1 (ply = 4), the first occurrence (index 1) is at the root, not after it.
        assert!(!board.is_draw_in_search(4));
    }

    #[test]
    fn insufficient_material_edge_cases() {
        // KBvKB on the same square colour is a draw (both dark: A1 and H2).
        let same = BoardState::parse_fen("4k3/8/8/8/8/8/7b/B3K3 w - - 0 1");
        assert!(same.is_draw());
        // Opposite colours (A1 dark, A2 light) can still mate.
        let opposite = BoardState::parse_fen("4k3/8/8/8/8/8/b7/B3K3 w - - 0 1");
        assert!(!opposite.is_draw());
        // KNvKN is a draw.
        let knights = BoardState::parse_fen("4k3/8/8/8/8/8/7n/4KN2 w - - 0 1");
        assert!(knights.is_draw());
        // KBvKN can still mate.
        let mixed = BoardState::parse_fen("4k3/8/8/8/8/8/7n/B3K3 w - - 0 1");
        assert!(!mixed.is_draw());
        // KNNvK can still mate (two minors on one side).
        let two_knights = BoardState::parse_fen("4k3/8/8/8/8/8/8/4KNN1 w - - 0 1");
        assert!(!two_knights.is_draw());
        // Five minor pieces: above the four-piece limit.
        let five = BoardState::parse_fen("4k3/8/8/8/8/8/7n/4KNN1 w - - 0 1");
        assert!(!five.is_draw());
    }

    #[test]
    fn fifty_move_draw_yields_to_checkmate() {
        // Scholar's mate: 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6 4.Qxf7#.
        let mut board = BoardState::default();
        for (from, to, kind) in [
            (Square::E2, Square::E4, MoveType::DoublePush),
            (Square::E7, Square::E5, MoveType::DoublePush),
            (Square::D1, Square::H5, MoveType::Quiet),
            (Square::B8, Square::C6, MoveType::Quiet),
            (Square::F1, Square::C4, MoveType::Quiet),
            (Square::G8, Square::F6, MoveType::Quiet),
            (Square::H5, Square::F7, MoveType::Capture),
        ] {
            board.make_move(Move::new(from, to, kind));
        }
        assert!(board.is_in_check(crate::common::side::Side::Black));
        board.half_move_clock = 100;
        assert!(!board.is_draw());
        assert!(!board.is_draw_in_search(0));
        // A fifty-move stalemate is still a draw.
        let mut stale = BoardState::parse_fen("7k/5Q2/8/8/8/8/8/6K1 b - - 0 1");
        stale.half_move_clock = 100;
        assert!(stale.is_draw());
    }

    #[test]
    fn null_move_resets_repetition_window_and_restores() {
        let mut board = BoardState::default();
        let nf3 = Move::new(Square::G1, Square::F3, MoveType::Quiet);
        let nf6 = Move::new(Square::G8, Square::F6, MoveType::Quiet);
        let ng1 = Move::new(Square::F3, Square::G1, MoveType::Quiet);
        let ng8 = Move::new(Square::F6, Square::G8, MoveType::Quiet);
        for _ in 0..2 {
            board.make_move(nf3);
            board.make_move(nf6);
            board.make_move(ng1);
            board.make_move(ng8);
        }
        assert!(board.is_draw());
        board.make_null_move();
        assert_eq!(board.en_passant_square, Square::NoSquare);
        assert!(!board.is_draw());
        board.undo_null_move();
        assert!(board.is_draw());
    }

    #[test]
    fn double_push_sets_ep_only_with_adjacent_pawn() {
        let mut board = BoardState::default();
        board.make_move(Move::new(Square::E2, Square::E4, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::NoSquare);
    }

    #[test]
    fn black_double_push_sets_ep_with_adjacent_pawn() {
        let mut board =
            BoardState::parse_fen("rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        board.make_move(Move::new(Square::D7, Square::D5, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::D6);
    }

    #[test]
    fn phantom_ep_cleared_when_rook_would_check() {
        // After ...e7-e5, dxe6 would uncover the A5 rook on the white king.
        let mut board = BoardState::parse_fen("7k/4p3/8/r2P3K/8/8/8/8 b - - 0 1");
        board.make_move(Move::new(Square::E7, Square::E5, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::NoSquare);
    }

    #[test]
    fn phantom_ep_cleared_when_bishop_would_check() {
        // After ...e7-e5, dxe6 would uncover the B7 bishop on the white king.
        let mut board = BoardState::parse_fen("4k3/1b2p3/8/3P4/4K3/8/8/8 b - - 0 1");
        board.make_move(Move::new(Square::E7, Square::E5, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::NoSquare);
    }

    #[test]
    fn phantom_ep_cleared_when_knight_checks_king() {
        // The D6 knight already checks the king, so no EP capture is legal.
        let mut board = BoardState::parse_fen("4k3/4p3/3n4/3P4/4K3/8/8/8 b - - 0 1");
        board.make_move(Move::new(Square::E7, Square::E5, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::NoSquare);
    }

    #[test]
    fn phantom_ep_cleared_when_pawn_checks_king() {
        // The D5 pawn already checks the king; ...f7-f5 helps nothing.
        let mut board = BoardState::parse_fen("4k3/5p2/8/3Pp3/4K3/8/8/8 b - - 0 1");
        board.make_move(Move::new(Square::F7, Square::F5, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::NoSquare);
    }

    #[test]
    fn kingless_double_push_keeps_ep() {
        let mut board = BoardState::parse_fen("8/4p3/8/3P4/8/8/8/8 b - - 0 1");
        board.make_move(Move::new(Square::E7, Square::E5, MoveType::DoublePush));
        assert_eq!(board.en_passant_square, Square::E6);
    }

    #[test]
    fn moving_rook_clears_only_its_castling_right() {
        let mut board = BoardState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        board.make_move(Move::new(Square::A1, Square::A2, MoveType::Quiet));
        assert!(!board.castle.contains(Castle::WHITE_LONG));
        assert!(board.castle.contains(Castle::WHITE_SHORT));
        assert!(board.castle.contains(Castle::BLACK_SHORT));
    }

    #[test]
    fn capturing_rook_clears_opponent_right() {
        let mut board = BoardState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        board.make_move(Move::new(Square::A1, Square::A8, MoveType::Capture));
        assert!(!board.castle.contains(Castle::WHITE_LONG));
        assert!(!board.castle.contains(Castle::BLACK_LONG));
        assert!(board.castle.contains(Castle::WHITE_SHORT));
    }

    #[test]
    fn king_move_clears_both_rights() {
        let mut board = BoardState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        board.make_move(Move::new(Square::E1, Square::E2, MoveType::Quiet));
        assert!(!board.castle.contains(Castle::WHITE_SHORT));
        assert!(!board.castle.contains(Castle::WHITE_LONG));
        assert!(board.castle.contains(Castle::BLACK_SHORT));
    }

    #[test]
    fn en_passant_capture_removes_pawn_and_unmakes() {
        let mut board =
            BoardState::parse_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1");
        let hash_before = board.board_hash;
        let mv = Move::new(Square::E5, Square::D6, MoveType::EnPassant);
        board.make_move(mv);
        assert_eq!(
            board
                .get_pieces(Side::White, Piece::Pawn)
                .get_bit(Square::D6 as usize),
            1
        );
        assert_eq!(
            board
                .get_pieces(Side::Black, Piece::Pawn)
                .get_bit(Square::D5 as usize),
            0
        );
        assert_eq!(board.half_move_clock, 0);
        board.unmake_move(mv);
        assert_eq!(board.board_hash, hash_before);
        assert_eq!(
            board
                .get_pieces(Side::Black, Piece::Pawn)
                .get_bit(Square::D5 as usize),
            1
        );
        assert_eq!(board.en_passant_square, Square::D6);
    }

    #[test]
    fn promotion_capture_replaces_piece_and_unmakes() {
        let mut board = BoardState::parse_fen("5r1k/6P1/8/8/8/8/8/4K3 w - - 0 1");
        let mv = Move::new(Square::G7, Square::F8, MoveType::QueenPromotionCapture);
        board.make_move(mv);
        assert_eq!(board.piece_mapping[Square::F8 as usize], Piece::Queen);
        assert!(board.get_pieces(Side::Black, Piece::Rook).is_empty());
        board.unmake_move(mv);
        assert_eq!(board.piece_mapping[Square::G7 as usize], Piece::Pawn);
        assert_eq!(
            board
                .get_pieces(Side::Black, Piece::Rook)
                .get_bit(Square::F8 as usize),
            1
        );
    }

    #[test]
    fn capture_on_empty_square_and_odd_castle_type_do_not_panic() {
        let mut board = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
        let hash_before = board.board_hash;
        // Capture flag on an empty target: no victim to remove.
        let phantom = Move::new(Square::E2, Square::E4, MoveType::Capture);
        board.make_move(phantom);
        assert_eq!(board.piece_mapping[Square::E4 as usize], Piece::Pawn);
        board.unmake_move(phantom);
        assert_eq!(board.board_hash, hash_before);
        // Castle type with a non-castle target hits the no-op rook branch.
        let odd = Move::new(Square::E2, Square::E3, MoveType::Castle);
        board.make_move(odd);
        board.unmake_move(odd);
        assert_eq!(board.board_hash, hash_before);
    }

    #[test]
    fn is_draw_in_search_early_exits() {
        // Bare kings: insufficient material at any ply.
        let bare = BoardState::parse_fen("8/8/8/8/8/8/8/4K2k w - - 0 1");
        assert!(bare.is_draw_in_search(0));
        // Fresh position: nothing to repeat yet.
        let fresh = BoardState::starting_position();
        assert!(!fresh.is_draw());
        assert!(!fresh.is_draw_in_search(0));
        // Fifty quiet moves without mate is a draw.
        let mut fifty = BoardState::starting_position();
        fifty.half_move_clock = 100;
        assert!(fifty.is_draw());
        assert!(fifty.is_draw_in_search(0));
    }
}
