use crate::board::state::BoardState;
use crate::common::moves::Move;
use crate::common::piece::Piece;

pub const CORRECTION_HISTORY_SIZE: usize = 16384;
pub const CORRECTION_LIMIT: i16 = 250;

pub struct CorrectionHistory {
    pawn_table: Box<[[i32; CORRECTION_HISTORY_SIZE]; 2]>,
    minor_table: Box<[[i32; CORRECTION_HISTORY_SIZE]; 2]>,
    non_pawn_table: Box<[[i32; CORRECTION_HISTORY_SIZE]; 2]>,
    continuation_table: Box<[[i32; 64]; 12]>,
}

impl CorrectionHistory {
    pub fn new() -> Self {
        Self {
            pawn_table: Box::new([[0; CORRECTION_HISTORY_SIZE]; 2]),
            minor_table: Box::new([[0; CORRECTION_HISTORY_SIZE]; 2]),
            non_pawn_table: Box::new([[0; CORRECTION_HISTORY_SIZE]; 2]),
            continuation_table: Box::new([[0; 64]; 12]),
        }
    }

    pub fn clear(&mut self) {
        *self.pawn_table = [[0; CORRECTION_HISTORY_SIZE]; 2];
        *self.minor_table = [[0; CORRECTION_HISTORY_SIZE]; 2];
        *self.non_pawn_table = [[0; CORRECTION_HISTORY_SIZE]; 2];
        *self.continuation_table = [[0; 64]; 12];
    }

    pub fn get_correction(&self, board_state: &BoardState, previous_move: Option<Move>) -> i16 {
        let stm = board_state.side_to_move as usize;
        let pawn_hash = crate::common::zobrist::get_pawn_hash(board_state);
        let pawn_idx = (pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let minor_idx = ((pawn_hash >> 4) & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let non_pawn_hash = board_state.board_hash ^ pawn_hash;
        let non_pawn_idx = (non_pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;

        let pcv = self.pawn_table[stm][pawn_idx];
        let micv = self.minor_table[stm][minor_idx];
        let non_pawn = self.non_pawn_table[stm][non_pawn_idx];

        let mut cntcv = 0;
        if let Some(prev) = previous_move {
            let pc = board_state.piece_mapping[prev.target as usize];
            if pc != Piece::None {
                let prev_side = board_state.side_to_move.other();
                let idx = prev_side as usize * 6 + pc as usize;
                if idx < 12 {
                    cntcv = self.continuation_table[idx][prev.target as usize];
                }
            }
        }

        let cv = 15341 * pcv + 10569 * micv + 12906 * non_pawn + 8761 * cntcv;
        (cv / 131072).clamp(-CORRECTION_LIMIT as i32, CORRECTION_LIMIT as i32) as i16
    }

    pub fn update(&mut self, board_state: &BoardState, previous_move: Option<Move>, bonus: i32) {
        let stm = board_state.side_to_move as usize;
        let pawn_hash = crate::common::zobrist::get_pawn_hash(board_state);
        let pawn_idx = (pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let minor_idx = ((pawn_hash >> 4) & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let non_pawn_hash = board_state.board_hash ^ pawn_hash;
        let non_pawn_idx = (non_pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;

        Self::apply_bonus(&mut self.pawn_table[stm][pawn_idx], bonus);
        Self::apply_bonus(&mut self.minor_table[stm][minor_idx], bonus * 150 / 128);
        Self::apply_bonus(
            &mut self.non_pawn_table[stm][non_pawn_idx],
            bonus * 186 / 128,
        );

        if let Some(prev) = previous_move {
            let pc = board_state.piece_mapping[prev.target as usize];
            if pc != Piece::None {
                let prev_side = board_state.side_to_move.other();
                let idx = prev_side as usize * 6 + pc as usize;
                if idx < 12 {
                    Self::apply_bonus(
                        &mut self.continuation_table[idx][prev.target as usize],
                        bonus * 130 / 128,
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn apply_bonus(entry: &mut i32, bonus: i32) {
        let clamped = bonus.clamp(-1024, 1024);
        *entry += clamped - (*entry * clamped.abs()) / 1024;
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::move_type::MoveType;
    use crate::common::moves::Move;
    use crate::common::square::Square;

    #[test]
    fn initial_correction_is_zero() {
        let board = BoardState::default();
        let corr = CorrectionHistory::new();
        assert_eq!(corr.get_correction(&board, None), 0);
    }

    #[test]
    fn update_and_clear() {
        let board = BoardState::default();
        let mut corr = CorrectionHistory::new();
        corr.update(&board, None, 100);
        let val = corr.get_correction(&board, None);
        assert_ne!(val, 0);
        corr.clear();
        assert_eq!(corr.get_correction(&board, None), 0);
    }

    #[test]
    fn previous_move_continuation_path() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let mv = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        board.make_move(mv);
        let mut corr = CorrectionHistory::new();
        assert_eq!(corr.get_correction(&board, Some(mv)), 0);
        corr.update(&board, Some(mv), 300);
        assert_ne!(corr.get_correction(&board, Some(mv)), 0);

        let empty = Move::new(Square::A3, Square::A4, MoveType::Quiet);
        assert_eq!(
            corr.get_correction(&board, Some(empty)),
            corr.get_correction(&board, None)
        );
        corr.update(&board, Some(empty), 300);
        let _ = CorrectionHistory::default();
    }

    #[test]
    fn bonus_clamps_and_stays_bounded() {
        let board = BoardState::default();
        let mut corr = CorrectionHistory::new();
        corr.update(&board, None, 5_000);
        corr.update(&board, None, -5_000);
        let v = corr.get_correction(&board, None);
        assert!(v.abs() <= CORRECTION_LIMIT);
        for _ in 0..200 {
            corr.update(&board, None, 1_024);
        }
        assert!(corr.get_correction(&board, None).abs() <= CORRECTION_LIMIT);
    }
}
