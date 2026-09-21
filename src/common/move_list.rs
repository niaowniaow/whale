use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::square::Square;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScoredMove {
    pub mv: Move,
    pub score: i32,
}

pub const NO_SCORED_MOVE: ScoredMove = ScoredMove {
    mv: Move::NO_MOVE,
    score: 0,
};

impl ScoredMove {
    pub fn new(source: Square, target: Square, move_type: MoveType) -> Self {
        Self {
            mv: Move::new(source, target, move_type),
            score: 0,
        }
    }
}

pub const MAX_MOVES: usize = 218;

#[derive(Clone, Copy, Debug)]
pub struct MoveList {
    pub moves: [ScoredMove; MAX_MOVES],
    pub count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [NO_SCORED_MOVE; MAX_MOVES],
            count: 0,
        }
    }

    pub fn push(&mut self, m: ScoredMove) {
        if self.count >= MAX_MOVES {
            debug_assert!(self.count < MAX_MOVES);
            return;
        }
        self.moves[self.count] = m;
        self.count += 1;
    }

    pub fn clear(&mut self) {
        self.count = 0;
    }
}

impl Deref for MoveList {
    type Target = [ScoredMove];

    fn deref(&self) -> &Self::Target {
        &self.moves[..self.count]
    }
}

impl DerefMut for MoveList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.moves[..self.count]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::square::Square;

    fn sample_move() -> ScoredMove {
        ScoredMove::new(Square::E2, Square::E4, MoveType::Quiet)
    }

    #[test]
    fn test_scored_move_new_defaults_score_to_zero() {
        let sm = ScoredMove::new(Square::E2, Square::E4, MoveType::Quiet);
        assert_eq!(sm.mv.source, Square::E2);
        assert_eq!(sm.mv.target, Square::E4);
        assert_eq!(sm.mv.move_type, MoveType::Quiet);
        assert_eq!(sm.score, 0);
    }

    #[test]
    fn test_no_scored_move_constant_is_empty_quiet() {
        assert_eq!(NO_SCORED_MOVE.mv, Move::NO_MOVE);
        assert_eq!(NO_SCORED_MOVE.score, 0);
    }

    #[test]
    fn test_move_list_new_is_empty() {
        let list = MoveList::new();
        assert_eq!(list.count, 0);
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_move_list_default_matches_new() {
        let list = MoveList::default();
        assert_eq!(list.count, 0);
        assert!(list.is_empty());
    }

    #[test]
    fn test_move_list_push_increments_and_derefs() {
        let mut list = MoveList::new();
        let m1 = ScoredMove::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = ScoredMove::new(Square::D2, Square::D4, MoveType::DoublePush);
        list.push(m1);
        list.push(m2);
        assert_eq!(list.count, 2);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0], m1);
        assert_eq!(list[1], m2);
        assert_eq!(list.moves[0], m1);
    }

    #[test]
    fn test_move_list_push_overflow_is_capped() {
        let mut list = MoveList::new();
        for _ in 0..MAX_MOVES {
            list.push(sample_move());
        }
        assert_eq!(list.count, MAX_MOVES);
        assert_eq!(list.len(), MAX_MOVES);

        if !cfg!(debug_assertions) {
            list.push(sample_move());
            assert_eq!(list.count, MAX_MOVES);
            assert_eq!(list.len(), MAX_MOVES);
        }
    }

    #[test]
    fn test_move_list_clear_resets_count() {
        let mut list = MoveList::new();
        list.push(sample_move());
        list.push(sample_move());
        assert_eq!(list.count, 2);
        list.clear();
        assert_eq!(list.count, 0);
        assert!(list.is_empty());

        list.push(sample_move());
        assert_eq!(list.count, 1);
        assert_eq!(list[0], sample_move());
    }

    #[test]
    fn test_move_list_deref_mut_allows_mutation() {
        let mut list = MoveList::new();
        list.push(sample_move());
        list.push(sample_move());
        list[0].score = 150;
        list[1].score = -20;
        assert_eq!(list[0].score, 150);
        assert_eq!(list[1].score, -20);

        let scores: Vec<i32> = list.iter().map(|sm| sm.score).collect();
        assert_eq!(scores, vec![150, -20]);
    }

    #[test]
    fn test_move_list_preserves_score_on_push() {
        let mut list = MoveList::new();
        let mut sm = sample_move();
        sm.score = 777;
        list.push(sm);
        assert_eq!(list[0].score, 777);
        assert_eq!(list[0].mv, sm.mv);
    }
}
