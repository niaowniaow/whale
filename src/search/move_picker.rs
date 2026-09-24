use crate::board::state::BoardState;
use crate::common::move_list::{MoveList, ScoredMove};
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::square::Square;
use crate::eval::move_ordering::{self, MoveOrdering};
use crate::search::bmo::BanditArm;
pub const PICKER_TT_LIMIT: i32 = 8192;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPhase {
    PvMove,
    TtMove,
    GenerateCaptures,
    GoodCaptures,
    GenerateQuiets,
    GoodQuiets,
    BadCaptures,
    BadQuiets,
    Done,
}
pub struct MovePicker {
    pub phase: SearchPhase,
    pub arm: BanditArm,
    pv_move: Option<Move>,
    tt_best: Option<Move>,
    previous_move: Option<Move>,
    excluded_move: Option<Move>,
    good_captures_count: usize,
    current_index: usize,
    saved_quiet_index: usize,
    ply: usize,
    is_qsearch: bool,
    cont_stack: [(Piece, Square); 6],
    cont_stack_len: usize,
    tt_history: i16,
}
impl MovePicker {
    pub fn new(
        pv_move: Option<Move>,
        tt_best: Option<Move>,
        previous_move: Option<Move>,
        ply: usize,
        excluded_move: Option<Move>,
        arm: BanditArm,
    ) -> Self {
        Self {
            phase: SearchPhase::PvMove,
            arm,
            pv_move,
            tt_best,
            previous_move,
            excluded_move,
            good_captures_count: 0,
            current_index: 0,
            saved_quiet_index: 0,
            ply,
            is_qsearch: false,
            cont_stack: [(Piece::None, Square::NoSquare); 6],
            cont_stack_len: 0,
            tt_history: 0,
        }
    }
    pub fn new_qsearch(ply: usize) -> Self {
        Self {
            phase: SearchPhase::PvMove,
            arm: BanditArm::CapturesFirst,
            pv_move: None,
            tt_best: None,
            previous_move: None,
            excluded_move: None,
            good_captures_count: 0,
            current_index: 0,
            saved_quiet_index: 0,
            ply,
            is_qsearch: true,
            cont_stack: [(Piece::None, Square::NoSquare); 6],
            cont_stack_len: 0,
            tt_history: 0,
        }
    }
    pub fn set_continuation_stack(&mut self, stack: &[(Piece, Square)]) {
        let n = stack.len().min(6);
        let mut i = 0;
        while i < n {
            self.cont_stack[i] = stack[i];
            i += 1;
        }
        self.cont_stack_len = n;
    }
    pub fn continuation_stack(&self) -> &[(Piece, Square)] {
        &self.cont_stack[..self.cont_stack_len]
    }
    pub fn continuation_stack_len(&self) -> usize {
        self.cont_stack_len
    }
    pub fn continuation_offset_ptr(
        &mut self,
        ordering: &mut MoveOrdering,
        offset_idx: usize,
    ) -> *mut [i16; 64] {
        if self.cont_stack_len == 0 {
            return std::ptr::null_mut();
        }
        let want_offset = if offset_idx < 4 {
            crate::eval::move_ordering::CONT_OFFSETS[offset_idx]
        } else {
            1
        };
        let pos = want_offset.min(self.cont_stack_len) - 1;
        let pc = self.cont_stack[pos].0 as usize;
        let sq = self.cont_stack[pos].1 as usize;
        ordering.continuation_ptr(offset_idx, pc, sq)
    }
    pub fn set_tt_history(&mut self, v: i16) {
        self.tt_history = v;
    }
    pub fn tt_history(&self) -> i16 {
        self.tt_history
    }
    pub fn update_tt_history(&mut self, best_is_tt: bool) {
        let bonus = if best_is_tt { 918 } else { -747 };
        let b = bonus.clamp(-PICKER_TT_LIMIT, PICKER_TT_LIMIT);
        let v = self.tt_history as i32;
        let nv = v + b - v * b.abs() / PICKER_TT_LIMIT;
        self.tt_history = nv.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    pub fn update_tt_history_multicut(&mut self, depth: u8) {
        let bonus = -421 - 110 * depth as i32;
        let b = bonus.clamp(-PICKER_TT_LIMIT, PICKER_TT_LIMIT);
        let v = self.tt_history as i32;
        let nv = v + b - v * b.abs() / PICKER_TT_LIMIT;
        self.tt_history = nv.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    pub fn tt_singular_double_margin(&self) -> i32 {
        -1175 * self.tt_history as i32 / 114178
    }
    pub fn tt_lmr_adjustment(&self) -> i32 {
        self.tt_history as i32 * 439 / 4096
    }
    pub fn singular_margin_with_tt(&self, base: i16, depth: u8) -> i16 {
        let bonus_part = 131 * depth as i32;
        let adj = bonus_part + self.tt_singular_double_margin();
        base.saturating_add(adj.clamp(-500, 500) as i16)
    }
    pub fn lmr_reduction_with_tt(&self, base: i32) -> i32 {
        base - self.tt_lmr_adjustment()
    }
    #[inline(always)]
    pub fn standard_bonus(depth: u8) -> i32 {
        (131 * depth as i32 - 31).clamp(0, 1487)
    }
    #[inline(always)]
    pub fn standard_malus(depth: u8) -> i32 {
        (968 * depth as i32 - 235).clamp(0, 2244)
    }
    pub fn next(
        &mut self,
        board_state: &mut BoardState,
        move_ordering: &MoveOrdering,
        captures: &mut MoveList,
        quiets: &mut MoveList,
        nt: &crate::board::node_threats::NodeThreats,
    ) -> Option<Move> {
        loop {
            let m = self.next_internal(board_state, move_ordering, captures, quiets, nt);
            if let Some(mv) = m
                && Some(mv) == self.excluded_move
            {
                continue;
            }
            return m;
        }
    }
    fn next_internal(
        &mut self,
        board_state: &mut BoardState,
        move_ordering: &MoveOrdering,
        captures: &mut MoveList,
        quiets: &mut MoveList,
        nt: &crate::board::node_threats::NodeThreats,
    ) -> Option<Move> {
        loop {
            match self.phase {
                SearchPhase::PvMove => {
                    self.phase = SearchPhase::TtMove;
                    if let Some(mv) = self.pv_move
                        && mv != Move::NO_MOVE
                    {
                        return Some(mv);
                    }
                }
                SearchPhase::TtMove => {
                    self.phase = match self.arm {
                        BanditArm::CapturesFirst => SearchPhase::GenerateCaptures,
                        BanditArm::QuietsFirst => SearchPhase::GenerateQuiets,
                    };
                    if let Some(mv) = self.tt_best
                        && mv != Move::NO_MOVE
                        && Some(mv) != self.pv_move
                    {
                        return Some(mv);
                    }
                }
                SearchPhase::GenerateCaptures => {
                    captures.clear();
                    self.current_index = 0;
                    board_state.generate_captures(captures);
                    move_ordering::populate_capture_scores(captures, board_state, move_ordering);
                    let mut left = 0;
                    let mut right = captures.len() as i32 - 1;
                    while left <= right {
                        let moved_piece =
                            board_state.get_piece_on(captures[left as usize].mv.source);
                        let target = captures[left as usize].mv.target as usize;
                        let target_piece =
                            if captures[left as usize].mv.move_type == MoveType::EnPassant {
                                Piece::Pawn
                            } else if target < crate::common::constants::SQUARES {
                                board_state.piece_mapping[target]
                            } else {
                                Piece::None
                            };
                        let hist = if moved_piece >= 0
                            && (moved_piece as usize) < crate::common::constants::PIECES * 2
                            && target < crate::common::constants::SQUARES
                            && target_piece != Piece::None
                            && (target_piece as usize) < crate::common::constants::PIECES
                        {
                            move_ordering.capture_history[moved_piece as usize][target]
                                [target_piece as usize] as i32
                        } else {
                            0
                        };
                        let cap_val = target_piece.see_value() as i32;
                        let threshold = -((7 * cap_val + hist) / 18) as i16;
                        if board_state.see_ge(captures[left as usize].mv, threshold) {
                            left += 1;
                        } else {
                            captures.swap(left as usize, right as usize);
                            right -= 1;
                        }
                    }
                    self.good_captures_count = left as usize;
                    self.phase = SearchPhase::GoodCaptures;
                }
                SearchPhase::GoodCaptures => {
                    let count = self.good_captures_count.min(captures.len());
                    if let Some(mv) = get_next_valid_move(
                        &mut captures[..count],
                        &mut self.current_index,
                        self.pv_move,
                        self.tt_best,
                    ) {
                        return Some(mv);
                    } else if self.is_qsearch {
                        self.phase = SearchPhase::Done;
                    } else if self.arm == BanditArm::CapturesFirst {
                        self.phase = SearchPhase::GenerateQuiets;
                    } else {
                        self.phase = SearchPhase::BadCaptures;
                        self.current_index = self.good_captures_count.min(captures.len());
                    }
                }
                SearchPhase::GenerateQuiets => {
                    quiets.clear();
                    self.current_index = 0;
                    board_state.generate_quiets(quiets);
                    if self.cont_stack_len > 0 {
                        let pawn_key = crate::common::zobrist::get_pawn_hash(board_state);
                        move_ordering.populate_quiet_scores_with_stack(
                            quiets,
                            board_state,
                            self.ply,
                            &self.cont_stack[..self.cont_stack_len],
                            pawn_key,
                            &nt.threats_us,
                            &nt.check_squares,
                        );
                    } else {
                        move_ordering.populate_quiet_scores(
                            quiets,
                            board_state,
                            self.ply,
                            self.previous_move,
                            &nt.threats_us,
                            &nt.check_squares,
                        );
                    }
                    self.phase = SearchPhase::GoodQuiets;
                }
                SearchPhase::GoodQuiets => {
                    if let Some(mv) = get_next_valid_quiet_move(
                        quiets,
                        &mut self.current_index,
                        self.pv_move,
                        self.tt_best,
                        -14_000,
                    ) {
                        return Some(mv);
                    } else {
                        self.saved_quiet_index = self.current_index;
                        if self.arm == BanditArm::CapturesFirst {
                            self.phase = SearchPhase::BadCaptures;
                            self.current_index = self.good_captures_count.min(captures.len());
                        } else {
                            self.phase = SearchPhase::GenerateCaptures;
                        }
                    }
                }
                SearchPhase::BadCaptures => {
                    if let Some(mv) = get_next_valid_move(
                        captures,
                        &mut self.current_index,
                        self.pv_move,
                        self.tt_best,
                    ) {
                        return Some(mv);
                    } else {
                        self.phase = SearchPhase::BadQuiets;
                        self.current_index = self.saved_quiet_index;
                    }
                }
                SearchPhase::BadQuiets => {
                    if let Some(mv) = get_next_valid_move(
                        quiets,
                        &mut self.current_index,
                        self.pv_move,
                        self.tt_best,
                    ) {
                        return Some(mv);
                    } else {
                        self.phase = SearchPhase::Done;
                    }
                }
                SearchPhase::Done => return None,
            }
        }
    }
}
fn get_next_valid_move(
    moves: &mut [ScoredMove],
    current_index: &mut usize,
    pv_move: Option<Move>,
    tt_best: Option<Move>,
) -> Option<Move> {
    let limit = moves.len();
    while *current_index < limit {
        MoveOrdering::sort_next_best_move(moves, *current_index);
        let mv = moves[*current_index].mv;
        *current_index += 1;
        if Some(mv) != pv_move && Some(mv) != tt_best {
            return Some(mv);
        }
    }
    None
}
fn get_next_valid_quiet_move(
    moves: &mut [ScoredMove],
    current_index: &mut usize,
    pv_move: Option<Move>,
    tt_best: Option<Move>,
    threshold: i32,
) -> Option<Move> {
    let limit = moves.len();
    while *current_index < limit {
        MoveOrdering::sort_next_best_move(moves, *current_index);
        if moves[*current_index].score <= threshold {
            return None;
        }
        let mv = moves[*current_index].mv;
        *current_index += 1;
        if Some(mv) != pv_move && Some(mv) != tt_best {
            return Some(mv);
        }
    }
    None
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;
    #[test]
    fn test_move_picker_qsearch_only_good_captures() {
        let mut board = BoardState::parse_fen("k7/8/8/5n2/1p1p4/2B5/3R4/K7 w - - 0 1");
        let mut picker = MovePicker::new_qsearch(0);
        let mut captures = MoveList::new();
        let mut quiets = MoveList::new();
        let mut good_captures = Vec::new();
        let move_ordering = MoveOrdering::new();
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        while let Some(mv) =
            picker.next(&mut board, &move_ordering, &mut captures, &mut quiets, &nt)
        {
            good_captures.push(mv);
        }
        assert!(!good_captures.is_empty());
        let has_good_capture = good_captures
            .iter()
            .any(|m| m.source == Square::C3 && m.target == Square::B4);
        let has_bad_capture = good_captures
            .iter()
            .any(|m| m.source == Square::D2 && m.target == Square::D4);
        let has_quiets = good_captures.iter().any(|m| m.move_type == MoveType::Quiet);
        assert!(has_good_capture);
        assert!(!has_bad_capture);
        assert!(!has_quiets);
    }
    #[test]
    fn test_move_picker_normal_search_all_phases() {
        let mut board = BoardState::parse_fen("k7/8/8/5n2/1p1p4/2B5/3R4/K7 w - - 0 1");
        let mut picker = MovePicker::new(None, None, None, 0, None, BanditArm::CapturesFirst);
        let mut captures = MoveList::new();
        let mut quiets = MoveList::new();
        let mut returned_moves = Vec::new();
        let move_ordering = MoveOrdering::new();
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        while let Some(mv) =
            picker.next(&mut board, &move_ordering, &mut captures, &mut quiets, &nt)
        {
            returned_moves.push(mv);
        }
        let good_capture_idx = returned_moves
            .iter()
            .position(|m| m.source == Square::C3 && m.target == Square::B4)
            .unwrap();
        let bad_capture_idx = returned_moves
            .iter()
            .position(|m| m.source == Square::D2 && m.target == Square::D4)
            .unwrap();
        assert!(good_capture_idx < bad_capture_idx);
        let quiet_indices: Vec<usize> = returned_moves
            .iter()
            .enumerate()
            .filter(|(_, m)| m.move_type == MoveType::Quiet)
            .map(|(i, _)| i)
            .collect();
        assert!(!quiet_indices.is_empty());
        for &qi in &quiet_indices {
            assert!(
                good_capture_idx < qi,
                "Good capture should be searched before quiet moves"
            );
            assert!(
                qi < bad_capture_idx,
                "Bad capture should be searched after quiet moves"
            );
        }
    }
    #[test]
    fn test_move_picker_quiets_first_arm() {
        let mut board = BoardState::parse_fen("k7/8/8/5n2/1p1p4/2B5/3R4/K7 w - - 0 1");
        let mut picker = MovePicker::new(None, None, None, 0, None, BanditArm::QuietsFirst);
        let mut captures = MoveList::new();
        let mut quiets = MoveList::new();
        let mut returned_moves = Vec::new();
        let move_ordering = MoveOrdering::new();
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        while let Some(mv) =
            picker.next(&mut board, &move_ordering, &mut captures, &mut quiets, &nt)
        {
            returned_moves.push(mv);
        }
        let good_capture_idx = returned_moves
            .iter()
            .position(|m| m.source == Square::C3 && m.target == Square::B4)
            .unwrap();
        let quiet_indices: Vec<usize> = returned_moves
            .iter()
            .enumerate()
            .filter(|(_, m)| m.move_type == MoveType::Quiet)
            .map(|(i, _)| i)
            .collect();
        assert!(!quiet_indices.is_empty());
        assert!(quiet_indices[0] < good_capture_idx);
    }
    #[test]
    fn test_bad_quiets_searched_after_bad_captures() {
        let mut board = BoardState::parse_fen("k7/8/8/5n2/1p1p4/2B5/3R4/K7 w - - 0 1");
        let mut picker = MovePicker::new(None, None, None, 0, None, BanditArm::CapturesFirst);
        let mut captures = MoveList::new();
        let mut quiets = MoveList::new();
        let mut move_ordering = MoveOrdering::new();
        let quiet_to_penalize = Move {
            source: Square::A1,
            target: Square::B1,
            move_type: MoveType::Quiet,
        };
        move_ordering.update_quiet_history(board.side_to_move, quiet_to_penalize, -16384);
        let mut returned_moves = Vec::new();
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        while let Some(mv) =
            picker.next(&mut board, &move_ordering, &mut captures, &mut quiets, &nt)
        {
            returned_moves.push(mv);
        }
        let bad_capture_idx = returned_moves
            .iter()
            .position(|m| m.source == Square::D2 && m.target == Square::D4)
            .unwrap();
        let penalized_quiet_idx = returned_moves
            .iter()
            .position(|m| m.source == Square::A1 && m.target == Square::B1)
            .unwrap();
        assert!(bad_capture_idx < penalized_quiet_idx);
    }
}
