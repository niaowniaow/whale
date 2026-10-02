use crate::board::node_threats::NodeThreats;
use crate::board::state::BoardState;
use crate::common::move_list::MoveList;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterplayInfo {
    pub cpi: i32,

    pub forcing: i32,

    pub tactical: i32,

    pub activity: i32,

    pub mobility: i32,
}

fn popcount_sum(maps: &[u64]) -> i32 {
    maps.iter().map(|b| b.count_ones() as i32).sum()
}

pub fn compute_cpi_fast(nt: &NodeThreats) -> CounterplayInfo {
    let checkers = (nt.checkers.count_ones()) as i32;
    let pinned_them = (nt.pinned_them.count_ones()) as i32;
    let threats_them = popcount_sum(&nt.threats_them);
    let check_squares = popcount_sum(&nt.check_squares);

    let forcing = (checkers * 40 + (check_squares / 4).min(40)).clamp(0, 120);

    let tactical = (pinned_them * 12 + (threats_them / 2).min(60)).clamp(0, 120);

    let activity = (check_squares / 6).clamp(0, 40);

    let mobility = ((threats_them + check_squares) / 8).clamp(0, 40);

    CounterplayInfo {
        cpi: (forcing + tactical + activity + mobility).clamp(0, 300),
        forcing,
        tactical,
        activity,
        mobility,
    }
}

pub fn count_freedom(board: &BoardState) -> u32 {
    let nt = NodeThreats::compute(board);
    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);
    let mut n = 0u32;
    for i in 0..moves.len() {
        if board.is_legal_with(moves[i].mv, nt.checkers, nt.pinned) {
            n += 1;
        }
    }
    n
}

pub fn cpi_rfp_adjust(cpi: i32) -> i16 {
    (cpi / 8).clamp(0, 60) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::{KIWI_PETE_FEN, STARTING_FEN};

    #[test]
    fn starting_position_cpi_is_low() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let nt = NodeThreats::compute(&board);
        let info = compute_cpi_fast(&nt);
        assert!(info.cpi < 40, "starting CPI {}", info.cpi);

        assert_eq!(nt.checkers.count_ones(), 0);
    }

    #[test]
    fn kiwi_pete_cpi_exceeds_starting() {
        let start = BoardState::parse_fen(STARTING_FEN);
        let kiwi = BoardState::parse_fen(KIWI_PETE_FEN);
        let c0 = compute_cpi_fast(&NodeThreats::compute(&start)).cpi;
        let c1 = compute_cpi_fast(&NodeThreats::compute(&kiwi)).cpi;
        assert!(c1 > c0, "kiwi {c1} should exceed start {c0}");
    }

    #[test]
    fn check_position_has_forcing() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let nt = NodeThreats::compute(&board);
        assert!(nt.in_check());
        let info = compute_cpi_fast(&nt);
        assert!(info.forcing > 0, "check must show forcing: {info:?}");
    }

    #[test]
    fn starting_freedom_is_twenty() {
        let board = BoardState::parse_fen(STARTING_FEN);
        assert_eq!(count_freedom(&board), 20);
    }

    #[test]
    fn rfp_adjust_monotone_and_bounded() {
        assert_eq!(cpi_rfp_adjust(0), 0);
        assert!(cpi_rfp_adjust(80) > cpi_rfp_adjust(16));
        assert!(cpi_rfp_adjust(10_000) <= 60);
    }
}
