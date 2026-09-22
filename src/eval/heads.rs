use crate::board::node_threats::NodeThreats;
use crate::board::state::BoardState;
use crate::perception::compute_strategic_snapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BehaviorHeads {
    pub value_cp: i16,
    pub tactical_danger: u8,
    pub king_safety: u8,
    pub mobility: u8,
    pub weakness: u8,
    pub initiative: i8,
    pub counterplay: u8,
    pub pressure_hint: i8,
    pub volatility: u8,
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 100) as u8
}

pub fn compute_heads(
    board: &BoardState,
    score: i16,
    own_cpi: i32,
    opp_cpi: i32,
    own_freedom: u32,
    opp_freedom: u32,
    momentum: i16,
) -> BehaviorHeads {
    let nt = NodeThreats::compute(board);
    let snap = compute_strategic_snapshot(board);
    let checkers = nt.checkers.count_ones() as i32;
    let mut check_sq = 0i32;
    for bb in nt.check_squares.iter() {
        check_sq += bb.count_ones() as i32;
    }
    let mut threats = 0i32;
    for bb in nt.threats_them.iter() {
        threats += bb.count_ones() as i32;
    }
    let tactical_danger = clamp_u8(checkers * 30 + (check_sq / 4).min(40) + (threats / 4).min(30));
    let shelter = if board.side_to_move == crate::common::side::Side::White {
        snap.white_king_shelter as i32
    } else {
        snap.black_king_shelter as i32
    };
    let king_safety = clamp_u8(shelter * 11 - checkers * 20);
    let mobility = clamp_u8((own_freedom as i32 * 100 / 32).min(100));
    let weak_pawns = (snap.white_isolated_pawns as i32
        + snap.black_isolated_pawns as i32
        + snap.white_backward_pawns as i32
        + snap.black_backward_pawns as i32)
        .min(8);
    let weakness = clamp_u8(weak_pawns * 12 + (opp_cpi.min(100) * 60 / 100));
    let initiative = ((own_freedom as i32 - opp_freedom as i32) * 3
        + (opp_cpi - own_cpi).clamp(-30, 30))
    .clamp(-100, 100) as i8;
    let counterplay = clamp_u8(opp_cpi.min(200) * 100 / 200);
    let pressure_hint = ((momentum as i32).clamp(-60, 60)
        + (own_freedom as i32 - opp_freedom as i32).clamp(-20, 20))
    .clamp(-100, 100) as i8;
    let volatility = clamp_u8((momentum.abs() as i32 * 2).min(100));
    BehaviorHeads {
        value_cp: score,
        tactical_danger,
        king_safety,
        mobility,
        weakness,
        initiative,
        counterplay,
        pressure_hint,
        volatility,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    fn startpos_heads() -> BehaviorHeads {
        let board = BoardState::parse_fen(STARTING_FEN);
        compute_heads(&board, 10, 12, 12, 20, 20, 0)
    }

    #[test]
    fn startpos_heads_in_range() {
        let h = startpos_heads();
        assert_eq!(h.value_cp, 10);
        assert!(h.tactical_danger < 30);
        assert!(h.king_safety > 50);
        assert_eq!(h.initiative, 0);
        assert_eq!(h.pressure_hint, 0);
    }

    #[test]
    fn check_position_raises_danger() {
        let calm = startpos_heads();
        let board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let hot = compute_heads(&board, 300, 10, 90, 20, 6, 40);
        assert!(hot.tactical_danger > calm.tactical_danger);
        assert!(hot.king_safety < calm.king_safety);
        assert!(hot.initiative > 0);
        assert!(hot.counterplay > calm.counterplay);
        assert!(hot.volatility > calm.volatility);
    }

    #[test]
    fn constrained_side_lowers_mobility_signal() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let free = compute_heads(&board, 0, 10, 10, 20, 20, 0);
        let squeezed = compute_heads(&board, 0, 10, 10, 6, 20, 0);
        assert!(squeezed.mobility < free.mobility);
        assert!(squeezed.pressure_hint < free.pressure_hint);
    }
}
