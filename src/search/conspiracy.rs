use crate::common::moves::Move;

#[derive(Debug, Clone, Copy)]
pub struct ConspiracyEntry {
    pub mv: Move,
    pub number: u32,
    pub score: i16,
}

impl ConspiracyEntry {
    pub fn new(mv: Move, score: i16) -> Self {
        Self {
            mv,
            number: 1,
            score,
        }
    }
}

pub fn rank_by_stability(moves: &[Move], scores: &[i16]) -> Vec<ConspiracyEntry> {
    let best = scores.iter().copied().max().unwrap_or(0);
    moves
        .iter()
        .zip(scores.iter().copied())
        .map(|(&mv, s)| {
            let mut e = ConspiracyEntry::new(mv, s);
            let gap = (best as i32 - s as i32).max(0) as u32;
            e.number = 1 + gap / 50;
            e
        })
        .collect()
}

pub fn needs_resolution(ranked: &[ConspiracyEntry], tolerance_cp: i16, budget_left: u8) -> bool {
    if budget_left == 0 || ranked.len() < 2 {
        return false;
    }
    let mut scores: Vec<i16> = ranked.iter().map(|e| e.score).collect();
    scores.sort_unstable_by(|a, b| b.cmp(a));
    scores[0] - scores[1] <= tolerance_cp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;

    fn mv() -> Move {
        Move::new(Square::E2, Square::E4, MoveType::Quiet)
    }

    #[test]
    fn far_moves_need_more_conspirators() {
        let m = mv();
        let ranked = rank_by_stability(&[m, m], &[100, -200]);
        assert!(ranked[1].number > ranked[0].number);
    }

    #[test]
    fn resolution_gate() {
        let m = mv();
        let close = rank_by_stability(&[m, m], &[100, 90]);
        assert!(needs_resolution(&close, 30, 3));
        assert!(!needs_resolution(&close, 30, 0));
        let far = rank_by_stability(&[m, m], &[300, 0]);
        assert!(!needs_resolution(&far, 30, 3));
    }
}
