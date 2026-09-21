use crate::common::moves::Move;
use crate::world::position_state::PositionState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateClass {
    Tactical,
    Defensive,
    Stabilizing,
    Improving,
    Pressing,
    Attacking,
    MustTry,
    Conversion,
    Reset,
}

#[derive(Debug, Clone)]
pub struct MultipvLine {
    pub mv: Move,
    pub score: i16,
    pub pv: Vec<Move>,
    pub kind: &'static str,
}

impl MultipvLine {
    pub fn uci_string(&self) -> String {
        let promotion = self
            .mv
            .promotion_char()
            .map(|c| c.to_string())
            .unwrap_or_else(String::new);
        format!("{}{}{}", self.mv.source, self.mv.target, promotion)
    }
}

pub fn classify_candidate(mv: Move, score: i16, best_score: i16) -> &'static str {
    classify_candidate_rich(mv, score, best_score, false, PositionState::Improve)
}

pub fn classify_candidate_rich(
    mv: Move,
    score: i16,
    best_score: i16,
    is_musttry: bool,
    state: PositionState,
) -> &'static str {
    let close_to_best = score >= best_score - 25;

    if is_musttry && close_to_best {
        return "must-try";
    }

    if close_to_best {
        match state {
            PositionState::Convert | PositionState::Crush => "conversion",
            PositionState::Defend => "defensive",
            PositionState::Attack => {
                if mv.is_capture() || mv.is_promotion() {
                    "attack"
                } else {
                    "attack-prep"
                }
            }
            PositionState::Press => "press",
            PositionState::Improve => "improve",
            PositionState::Stabilize => "stabilize",
        }
    } else if mv.is_capture() || mv.is_promotion() {
        "tactical-alt"
    } else {
        "quiet-alt"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;

    #[test]
    fn classification_labels() {
        let cap = Move::new(Square::E5, Square::D6, MoveType::Capture);
        assert_eq!(classify_candidate(cap, 50, 50), "improve");
        let q = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        assert_eq!(classify_candidate(q, 50, 50), "improve");
        assert_eq!(classify_candidate(q, 0, 100), "quiet-alt");
    }

    #[test]
    fn rich_classification_handles_states() {
        let q = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        assert_eq!(
            classify_candidate_rich(q, 100, 100, true, PositionState::Attack),
            "must-try"
        );
        assert_eq!(
            classify_candidate_rich(q, 100, 100, false, PositionState::Press),
            "press"
        );
        assert_eq!(
            classify_candidate_rich(q, 100, 100, false, PositionState::Convert),
            "conversion"
        );
        assert_eq!(
            classify_candidate_rich(q, 100, 100, false, PositionState::Defend),
            "defensive"
        );
    }
}
