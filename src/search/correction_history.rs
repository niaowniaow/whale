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

        let pawn_corr = self.pawn_table[stm][pawn_idx];
        let minor_corr = self.minor_table[stm][minor_idx];
        let non_pawn_corr = self.non_pawn_table[stm][non_pawn_idx];

        let mut cont_corr = 0;
        if let Some(prev) = previous_move {
            let pc = board_state.piece_mapping[prev.target as usize];
            if pc != Piece::None {
                let prev_side = board_state.side_to_move.other();
                let idx = prev_side as usize * 6 + pc as usize;
                if idx < 12 {
                    cont_corr = self.continuation_table[idx][prev.target as usize];
                }
            }
        }

        ((pawn_corr + minor_corr + non_pawn_corr + cont_corr) / 256)
            .clamp(-CORRECTION_LIMIT as i32, CORRECTION_LIMIT as i32) as i16
    }

    pub fn update(
        &mut self,
        board_state: &BoardState,
        previous_move: Option<Move>,
        diff: i32,
        weight: i32,
    ) {
        let stm = board_state.side_to_move as usize;
        let pawn_hash = crate::common::zobrist::get_pawn_hash(board_state);
        let pawn_idx = (pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let minor_idx = ((pawn_hash >> 4) & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let non_pawn_hash = board_state.board_hash ^ pawn_hash;
        let non_pawn_idx = (non_pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;

        Self::apply_ema(&mut self.pawn_table[stm][pawn_idx], diff, weight);

        let minor_weight = weight * 150 / 128;
        Self::apply_ema(&mut self.minor_table[stm][minor_idx], diff, minor_weight);

        let non_pawn_weight = weight * 100 / 128;
        Self::apply_ema(
            &mut self.non_pawn_table[stm][non_pawn_idx],
            diff,
            non_pawn_weight,
        );

        if let Some(prev) = previous_move {
            let pc = board_state.piece_mapping[prev.target as usize];
            if pc != Piece::None {
                let prev_side = board_state.side_to_move.other();
                let idx = prev_side as usize * 6 + pc as usize;
                if idx < 12 {
                    let cont_weight = weight * 130 / 128;
                    Self::apply_ema(
                        &mut self.continuation_table[idx][prev.target as usize],
                        diff,
                        cont_weight,
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn apply_ema(entry: &mut i32, diff: i32, weight: i32) {
        *entry = (*entry * (256 - weight) + (diff / 2) * weight * 256) / 256;
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
        corr.update(&board, None, 100, 16);
        let val = corr.get_correction(&board, None);
        assert_ne!(val, 0);
        corr.clear();
        assert_eq!(corr.get_correction(&board, None), 0);
    }
}
