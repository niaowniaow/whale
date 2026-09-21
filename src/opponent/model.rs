use crate::board::node_threats::NodeThreats;
use crate::board::state::BoardState;
use crate::opponent::plans;
use crate::world::counterplay::{self, CounterplayInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpponentModel {
    pub counterplay_capacity: i32,
    pub defense_capacity: i32,
    pub escape_capacity: u32,
    pub consolidation_risk: i32,
    pub freedom: u32,
    pub pawn_breaks: u32,
}

pub fn evaluate_opponent(board: &BoardState) -> OpponentModel {
    let nt = NodeThreats::compute(board);
    let cpi_info: CounterplayInfo = counterplay::compute_cpi_fast(&nt);
    let freedom = counterplay::count_freedom(board);
    let pawn_breaks = plans::available_pawn_breaks(board);

    let k_sq = board.get_king_square_safe(board.side_to_move);
    let k_defenders = if let Some(sq) = k_sq {
        plans::defenders_of(board, sq, board.side_to_move) as i32
    } else {
        0
    };

    let defense_capacity = (k_defenders * 15 + (freedom as i32 / 4)).clamp(0, 150);

    let escape_capacity = (freedom.saturating_sub(pawn_breaks)).min(40);

    let consolidation_risk = ((defense_capacity / 2) + (escape_capacity as i32 * 2)).clamp(0, 150);

    OpponentModel {
        counterplay_capacity: cpi_info.cpi,
        defense_capacity,
        escape_capacity,
        consolidation_risk,
        freedom,
        pawn_breaks,
    }
}

impl BoardState {
    pub fn get_king_square_safe(&self, side: crate::common::side::Side) -> Option<crate::common::square::Square> {
        let bb = self.get_pieces(side, crate::common::piece::Piece::King);
        if bb.is_empty() {
            None
        } else {
            Some(crate::common::square::Square::from(bb.get_lsb() as usize))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn starting_position_opponent_model_is_sane() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let opp = evaluate_opponent(&board);

        assert!(opp.freedom > 0);
        assert_eq!(opp.pawn_breaks, 16);
        assert!(opp.defense_capacity > 0);
    }
}
