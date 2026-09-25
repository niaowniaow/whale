use crate::board::state::BoardState;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::square::Square;
pub const CORRECTION_HISTORY_SIZE: usize = 16384;
pub const CORRECTION_LIMIT: i16 = 250;
pub const CORR_CONT_COUNT: usize = 3;
pub const CORR_CONT_OFFSETS: [usize; 3] = [2, 4, 6];
pub const CORR_HISTORY_GRAVITY: i32 = 1024;
pub struct CorrectionHistory {
    pawn_table: Box<[[i32; CORRECTION_HISTORY_SIZE]; 2]>,
    minor_table: Box<[[i32; CORRECTION_HISTORY_SIZE]; 2]>,
    non_pawn_table: Box<[[i32; CORRECTION_HISTORY_SIZE]; 2]>,
    continuation_table: Box<[[i32; 64]; 12]>,
    continuation_multi: Box<[[[i32; 64]; 12]; 3]>,
}
impl CorrectionHistory {
    pub fn new() -> Self {
        Self {
            pawn_table: Box::new([[0; CORRECTION_HISTORY_SIZE]; 2]),
            minor_table: Box::new([[0; CORRECTION_HISTORY_SIZE]; 2]),
            non_pawn_table: Box::new([[0; CORRECTION_HISTORY_SIZE]; 2]),
            continuation_table: Box::new([[0; 64]; 12]),
            continuation_multi: Box::new([[[0; 64]; 12]; 3]),
        }
    }
    pub fn clear(&mut self) {
        *self.pawn_table = [[0; CORRECTION_HISTORY_SIZE]; 2];
        *self.minor_table = [[0; CORRECTION_HISTORY_SIZE]; 2];
        *self.non_pawn_table = [[0; CORRECTION_HISTORY_SIZE]; 2];
        *self.continuation_table = [[0; 64]; 12];
        *self.continuation_multi = [[[0; 64]; 12]; 3];
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
            let target = prev.target as usize;
            if target < 64 {
                let pc = board_state.piece_mapping[target];
                if pc != Piece::None {
                    let prev_side = board_state.side_to_move.other();
                    let idx = prev_side as usize * 6 + pc as usize;
                    if idx < 12 {
                        cntcv = self.continuation_table[idx][target];
                    }
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
            let target = prev.target as usize;
            if target < 64 {
                let pc = board_state.piece_mapping[target];
                if pc != Piece::None {
                    let prev_side = board_state.side_to_move.other();
                    let idx = prev_side as usize * 6 + pc as usize;
                    if idx < 12 {
                        Self::apply_bonus(
                            &mut self.continuation_table[idx][target],
                            bonus * 130 / 128,
                        );
                    }
                }
            }
        }
    }
    #[inline(always)]
    fn apply_bonus(entry: &mut i32, bonus: i32) {
        let clamped = bonus.clamp(-1024, 1024);
        *entry += clamped - (*entry * clamped.abs()) / 1024;
    }
    #[inline(always)]
    pub fn corr_cont_index_for_offset(offset: usize) -> usize {
        match offset {
            2 => 0,
            4 => 1,
            6 => 2,
            _ => 0,
        }
    }
    #[inline(always)]
    pub fn continuation_ptr(&mut self, offset_idx: usize, idx: usize) -> *mut [i32; 64] {
        let oi = if offset_idx < CORR_CONT_COUNT {
            offset_idx
        } else {
            0
        };
        let ii = if idx < 12 { idx } else { 0 };
        &mut self.continuation_multi[oi][ii] as *mut [i32; 64]
    }
    #[inline(always)]
    pub fn continuation_entry(&self, offset_idx: usize, idx: usize, target: usize) -> i32 {
        if offset_idx >= CORR_CONT_COUNT || idx >= 12 || target >= 64 {
            return 0;
        }
        self.continuation_multi[offset_idx][idx][target]
    }
    #[inline(always)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn get_via_ptr(ptr: *const [i32; 64], target: usize) -> i32 {
        if target >= 64 {
            return 0;
        }
        unsafe { (*ptr)[target] }
    }
    #[inline(always)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn update_via_ptr(ptr: *mut [i32; 64], target: usize, bonus: i32) {
        if target >= 64 {
            return;
        }
        unsafe {
            let slot = &mut (*ptr)[target];
            let clamped = bonus.clamp(-1024, 1024);
            *slot += clamped - (*slot * clamped.abs()) / 1024;
        }
    }
    pub fn get_correction_with_stack(
        &self,
        board_state: &BoardState,
        stack: &[(Piece, Square)],
    ) -> i16 {
        let stm = board_state.side_to_move as usize;
        let pawn_hash = crate::common::zobrist::get_pawn_hash(board_state);
        let pawn_idx = (pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let minor_idx = ((pawn_hash >> 4) & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let non_pawn_hash = board_state.board_hash ^ pawn_hash;
        let non_pawn_idx = (non_pawn_hash & (CORRECTION_HISTORY_SIZE as u64 - 1)) as usize;
        let pcv = self.pawn_table[stm][pawn_idx];
        let micv = self.minor_table[stm][minor_idx];
        let non_pawn = self.non_pawn_table[stm][non_pawn_idx];
        let mut cnt0 = 0;
        let mut cnt2 = 0;
        let mut cnt4 = 0;
        let mut cnt6 = 0;
        let n = stack.len();
        if n > 0 {
            let pc0 = stack[0].0;
            let sq0 = stack[0].1 as usize;
            if pc0 != Piece::None && sq0 < 64 {
                let prev_side = board_state.side_to_move.other();
                let idx0 = prev_side as usize * 6 + pc0 as usize;
                if idx0 < 12 {
                    cnt0 = self.continuation_table[idx0][sq0];
                }
            }
        }
        if n > 1 {
            let pc1 = stack[1].0;
            let sq1 = stack[1].1 as usize;
            if pc1 != Piece::None && sq1 < 64 {
                let idx1 = stm * 6 + pc1 as usize;
                if idx1 < 12 {
                    cnt2 = self.continuation_multi[0][idx1][sq1];
                }
            }
        }
        if n > 3 {
            let pc3 = stack[3].0;
            let sq3 = stack[3].1 as usize;
            if pc3 != Piece::None && sq3 < 64 {
                let idx3 = stm * 6 + pc3 as usize;
                if idx3 < 12 {
                    cnt4 = self.continuation_multi[1][idx3][sq3];
                }
            }
        }
        if n > 5 {
            let pc5 = stack[5].0;
            let sq5 = stack[5].1 as usize;
            if pc5 != Piece::None && sq5 < 64 {
                let idx5 = stm * 6 + pc5 as usize;
                if idx5 < 12 {
                    cnt6 = self.continuation_multi[2][idx5][sq5];
                }
            }
        }
        let cv = 15341 * pcv
            + 10569 * micv
            + 12906 * non_pawn
            + 8761 * cnt0
            + 7885 * cnt2
            + 7885 * cnt4
            + 6307 * cnt6;
        (cv / 131072).clamp(-CORRECTION_LIMIT as i32, CORRECTION_LIMIT as i32) as i16
    }
    pub fn update_with_stack(
        &mut self,
        board_state: &BoardState,
        stack: &[(Piece, Square)],
        bonus: i32,
    ) {
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
        let n = stack.len();
        if n > 0 {
            let pc0 = stack[0].0;
            let sq0 = stack[0].1 as usize;
            if pc0 != Piece::None && sq0 < 64 {
                let prev_side = board_state.side_to_move.other();
                let idx0 = prev_side as usize * 6 + pc0 as usize;
                if idx0 < 12 {
                    Self::apply_bonus(&mut self.continuation_table[idx0][sq0], bonus * 130 / 128);
                }
            }
        }
        if n > 1 {
            let pc1 = stack[1].0;
            let sq1 = stack[1].1 as usize;
            if pc1 != Piece::None && sq1 < 64 {
                let idx1 = stm * 6 + pc1 as usize;
                if idx1 < 12 {
                    Self::apply_bonus(
                        &mut self.continuation_multi[0][idx1][sq1],
                        bonus * 130 / 128,
                    );
                }
            }
        }
        if n > 3 {
            let pc3 = stack[3].0;
            let sq3 = stack[3].1 as usize;
            if pc3 != Piece::None && sq3 < 64 {
                let idx3 = stm * 6 + pc3 as usize;
                if idx3 < 12 {
                    Self::apply_bonus(&mut self.continuation_multi[1][idx3][sq3], bonus * 70 / 128);
                }
            }
        }
        if n > 5 {
            let pc5 = stack[5].0;
            let sq5 = stack[5].1 as usize;
            if pc5 != Piece::None && sq5 < 64 {
                let idx5 = stm * 6 + pc5 as usize;
                if idx5 < 12 {
                    Self::apply_bonus(&mut self.continuation_multi[2][idx5][sq5], bonus * 35 / 128);
                }
            }
        }
    }
    #[inline(always)]
    pub fn standard_bonus_131(depth: u8) -> i32 {
        (131 * depth as i32 - 31).clamp(-1024, 1024)
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
