use crate::common::castle::Castle;
use crate::common::piece::Piece;
use crate::common::square::Square;
use crate::eval::nnue::accumulator::Accumulators;
use crate::eval::nnue::v16::Sfnn16Accs;

pub const HISTORY_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoardHistory {
    pub captured_piece: Piece,
    pub en_passant_square: Square,
    pub castling_rights: Castle,
    pub board_hash: u64,
    pub half_move_clock: u8,
}

impl Default for BoardHistory {
    fn default() -> Self {
        Self {
            captured_piece: Piece::None,
            en_passant_square: Square::NoSquare,
            castling_rights: Castle::NONE,
            board_hash: 0,
            half_move_clock: 0,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DirtyUpdate {
    pub adds_w: [usize; 2],
    pub dels_w: [usize; 2],
    pub adds_b: [usize; 2],
    pub dels_b: [usize; 2],
    pub n_adds: u8,
    pub n_dels: u8,
}

#[derive(Debug, Clone)]
pub struct History {
    pub entries: Box<[BoardHistory]>,
    pub accumulators: Box<[Accumulators]>,
    pub sfnn16: Box<[Sfnn16Accs]>,
    pub sfnn16_computed: Box<[[bool; 2]]>,
    pub sfnn16_pending: Box<[crate::eval::nnue::v16::SfnnPending]>,
    pub dirty_updates: Box<[DirtyUpdate]>,
    pub computed: Box<[bool]>,
    pub index: usize,
    /// Plies since the last null move (cf. Stockfish `pliesFromNull`).
    /// Repetition windows must stop at a null move, so callers use
    /// `min(half_move_clock, plies_from_null)` as the lookback distance.
    pub plies_from_null: usize,
    plies_from_null_stack: Box<[usize]>,
}

impl History {
    pub fn new() -> Self {
        let mut computed = vec![false; HISTORY_SIZE].into_boxed_slice();
        computed[0] = true;
        let mut sfnn16_computed = vec![[false, false]; HISTORY_SIZE].into_boxed_slice();
        sfnn16_computed[0] = [true, true];
        Self {
            entries: vec![BoardHistory::default(); HISTORY_SIZE].into_boxed_slice(),
            accumulators: vec![Accumulators::default(); HISTORY_SIZE].into_boxed_slice(),
            sfnn16: vec![Sfnn16Accs::empty(); HISTORY_SIZE].into_boxed_slice(),
            sfnn16_computed,
            sfnn16_pending: vec![
                crate::eval::nnue::v16::SfnnPending::default();
                HISTORY_SIZE
            ]
            .into_boxed_slice(),
            dirty_updates: vec![DirtyUpdate::default(); HISTORY_SIZE].into_boxed_slice(),
            computed,
            index: 0,
            plies_from_null: 0,
            plies_from_null_stack: vec![0usize; HISTORY_SIZE].into_boxed_slice(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn save(
        &mut self,
        captured_piece: Piece,
        en_passant: Square,
        original_castling_rights: Castle,
        board_hash: u64,
        half_move_clock: u8,
    ) {
        if self.index < HISTORY_SIZE {
            self.entries[self.index] = BoardHistory {
                captured_piece,
                en_passant_square: en_passant,
                castling_rights: original_castling_rights,
                board_hash,
                half_move_clock,
            };
            self.plies_from_null_stack[self.index] = self.plies_from_null;
            self.index += 1;
        } else {
            panic!("History stack overflow");
        }
    }

    pub fn restore(&mut self) -> BoardHistory {
        if self.index > 0 {
            self.index -= 1;
            self.plies_from_null = self.plies_from_null_stack[self.index];
            self.entries[self.index]
        } else {
            panic!("History stack underflow");
        }
    }

    pub fn clear(&mut self) {
        for i in 0..self.index {
            self.entries[i] = BoardHistory::default();
        }
        self.computed.fill(false);
        self.computed[0] = true;
        self.sfnn16_computed.fill([false, false]);
        self.sfnn16_computed[0] = [true, true];
        self.index = 0;
        self.plies_from_null = 0;
    }

    pub fn has_hash_appeared_twice(&self, board_hash: u64, starting_index: usize) -> bool {
        let mut count = 0;

        for i in (starting_index..self.index).rev() {
            if self.entries[i].board_hash == board_hash {
                count += 1;
            }

            if count == 2 {
                return true;
            }
        }
        false
    }

    pub fn is_empty(&self) -> bool {
        self.index == 0
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_history_save_restore() {
        let mut history = History::new();

        history.save(Piece::Pawn, Square::E3, Castle::WHITE_SHORT, 123456789, 42);

        assert!(!history.is_empty());
        assert_eq!(history.index, 1);

        let restored = history.restore();
        assert_eq!(restored.captured_piece, Piece::Pawn);
        assert_eq!(restored.en_passant_square, Square::E3);
        assert_eq!(restored.castling_rights, Castle::WHITE_SHORT);
        assert_eq!(restored.board_hash, 123456789);
        assert_eq!(restored.half_move_clock, 42);
        assert!(history.is_empty());
    }

    #[test]
    fn test_history_clear() {
        let mut history = History::new();
        history.save(Piece::None, Square::NoSquare, Castle::NONE, 0, 0);
        assert!(!history.is_empty());
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.index, 0);
    }

    #[test]
    fn test_has_hash_appeared_twice() {
        let mut history = History::new();
        let hash = 0xDEADBEEF;

        history.save(Piece::None, Square::NoSquare, Castle::NONE, hash, 0);
        history.save(Piece::None, Square::NoSquare, Castle::NONE, 0x123, 0);
        history.save(Piece::None, Square::NoSquare, Castle::NONE, hash, 0);

        assert!(history.has_hash_appeared_twice(hash, 0));
        assert!(!history.has_hash_appeared_twice(0x123, 0));
        assert!(!history.has_hash_appeared_twice(hash, 1));
    }

    #[test]
    fn test_should_save_and_restore_board_history() {
        use crate::board::state::BoardState;
        use crate::common::helpers::STARTING_FEN;
        use crate::common::move_type::MoveType;
        use crate::common::moves::Move;

        let mut board_state = BoardState::parse_fen(STARTING_FEN);
        let original_state_pieces = board_state.pieces;
        let original_state_side = board_state.side_to_move;
        let original_board_hash = board_state.board_hash;

        let move_e2e4 = Move::new(Square::E2, Square::E4, MoveType::Quiet);

        board_state.make_move(move_e2e4);

        assert_ne!(board_state.pieces, original_state_pieces);
        assert_ne!(board_state.side_to_move, original_state_side);
        assert_ne!(board_state.board_hash, original_board_hash);

        board_state.unmake_move(move_e2e4);

        assert_eq!(board_state.pieces, original_state_pieces);
        assert_eq!(board_state.side_to_move, original_state_side);
        assert_eq!(board_state.board_hash, original_board_hash);
    }

    #[test]
    fn board_history_default_values() {
        let entry = BoardHistory::default();
        assert_eq!(entry.captured_piece, Piece::None);
        assert_eq!(entry.en_passant_square, Square::NoSquare);
        assert_eq!(entry.castling_rights, Castle::NONE);
        assert_eq!(entry.board_hash, 0);
        assert_eq!(entry.half_move_clock, 0);
    }

    #[test]
    fn dirty_update_default_is_zeroed() {
        assert_eq!(
            DirtyUpdate::default(),
            DirtyUpdate {
                adds_w: [0; 2],
                dels_w: [0; 2],
                adds_b: [0; 2],
                dels_b: [0; 2],
                n_adds: 0,
                n_dels: 0,
            }
        );
    }

    #[test]
    fn history_default_starts_empty() {
        let history = History::default();
        assert!(history.is_empty());
        assert_eq!(history.index, 0);
        assert_eq!(history.plies_from_null, 0);
    }

    #[test]
    fn has_hash_appeared_twice_edge_ranges() {
        let history = History::new();
        assert!(!history.has_hash_appeared_twice(0x123, 0));

        let mut history = History::new();
        history.save(Piece::None, Square::NoSquare, Castle::NONE, 0xABC, 0);
        assert!(!history.has_hash_appeared_twice(0xABC, 0));
        // Empty range: start == len.
        assert!(!history.has_hash_appeared_twice(0xABC, 1));
        // Start beyond len is an empty range, not a panic.
        assert!(!history.has_hash_appeared_twice(0xABC, 99));
    }

    #[test]
    fn clear_resets_plies_and_stays_reusable() {
        let mut history = History::new();
        history.plies_from_null = 7;
        history.save(Piece::Pawn, Square::E3, Castle::WHITE_SHORT, 1, 1);
        history.save(Piece::Knight, Square::E4, Castle::BLACK_SHORT, 2, 2);
        history.clear();
        assert!(history.is_empty());
        assert_eq!(history.index, 0);
        assert_eq!(history.plies_from_null, 0);
        assert!(history.computed[0]);
        history.save(Piece::None, Square::NoSquare, Castle::NONE, 3, 0);
        assert_eq!(history.index, 1);
        assert_eq!(history.restore().board_hash, 3);
    }

    #[test]
    fn save_restore_roundtrips_plies_from_null() {
        let mut history = History::new();
        history.plies_from_null = 3;
        history.save(Piece::None, Square::NoSquare, Castle::NONE, 10, 0);
        history.plies_from_null = 9;
        let restored = history.restore();
        assert_eq!(restored.board_hash, 10);
        assert_eq!(history.plies_from_null, 3);
    }

    #[test]
    #[should_panic(expected = "History stack overflow")]
    fn save_panics_on_overflow() {
        let mut history = History::new();
        history.index = HISTORY_SIZE;
        history.save(Piece::None, Square::NoSquare, Castle::NONE, 0, 0);
    }

    #[test]
    #[should_panic(expected = "History stack underflow")]
    fn restore_panics_on_underflow() {
        History::new().restore();
    }
}
