use crate::board::state::BoardState;
use crate::common::piece::Piece;

#[cfg(test)]
use crate::common::side::Side;

#[derive(Clone, Copy, Debug)]
pub struct NodeThreats {
    pub checkers: u64,

    pub pinned: u64,

    pub pinned_them: u64,

    pub threats_us: [u64; 6],

    pub threats_them: [u64; 6],

    pub check_squares: [u64; 6],
}

impl NodeThreats {
    #[inline(always)]
    pub fn compute(board: &BoardState) -> Self {
        if board.history.is_cache_valid() {
            let c = board.history.current_cache();
            return Self {
                checkers: c.checkers,
                pinned: c.pinned,
                pinned_them: c.pinned_them,
                threats_us: c.threats_us,
                threats_them: c.threats_them,
                check_squares: c.check_squares,
            };
        }
        let stm = board.side_to_move;
        Self {
            checkers: board.checkers(stm).0,
            pinned: board.pinned_pieces(stm).0,
            pinned_them: board.pinned_pieces(stm.other()).0,
            threats_us: board.threat_by_lesser(stm),
            threats_them: board.threat_by_lesser(stm.other()),
            check_squares: board.check_squares(stm.other()),
        }
    }

    #[inline(always)]
    pub fn compute_for_qsearch(board: &BoardState) -> Self {
        if board.history.is_cache_valid() {
            let c = board.history.current_cache();
            return Self {
                checkers: c.checkers,
                pinned: c.pinned,
                pinned_them: 0,
                threats_us: c.threats_us,
                threats_them: [0; 6],
                check_squares: [0; 6],
            };
        }
        let stm = board.side_to_move;
        Self {
            checkers: board.checkers(stm).0,
            pinned: board.pinned_pieces(stm).0,
            pinned_them: 0,
            threats_us: board.threat_by_lesser(stm),
            threats_them: [0; 6],
            check_squares: [0; 6],
        }
    }

    #[inline(always)]
    pub fn compute_for_legality(board: &BoardState) -> Self {
        if board.history.is_cache_valid() {
            let c = board.history.current_cache();
            return Self {
                checkers: c.checkers,
                pinned: c.pinned,
                pinned_them: 0,
                threats_us: [0; 6],
                threats_them: [0; 6],
                check_squares: [0; 6],
            };
        }
        let stm = board.side_to_move;
        Self {
            checkers: 0,
            pinned: board.pinned_pieces(stm).0,
            pinned_them: 0,
            threats_us: [0; 6],
            threats_them: [0; 6],
            check_squares: [0; 6],
        }
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    #[inline(always)]
    pub fn checks(&self) -> u32 {
        self.checkers.count_ones()
    }

    #[inline(always)]
    pub fn queen_threat_us(&self, board: &BoardState) -> u32 {
        (self.threats_us[Piece::Queen as usize]
            & board.get_pieces(board.side_to_move, Piece::Queen).0)
            .count_ones()
    }

    #[inline(always)]
    pub fn queen_threat_them(&self, board: &BoardState) -> u32 {
        let them = board.side_to_move.other();
        (self.threats_them[Piece::Queen as usize] & board.get_pieces(them, Piece::Queen).0)
            .count_ones()
    }

    #[inline(always)]
    pub fn threatened_count(&self, board: &BoardState, piece: Piece) -> u32 {
        let idx = piece as usize;
        if idx == 0 || idx >= 5 {
            return 0;
        }
        (board.get_pieces(board.side_to_move, piece).0 & self.threats_us[idx]).count_ones()
    }

    #[inline(always)]
    pub fn cached_checkers(board: &BoardState) -> u64 {
        if board.history.is_cache_valid() {
            return board.history.current_cache().checkers;
        }
        board.checkers(board.side_to_move).0
    }

    #[inline(always)]
    pub fn cached_pinned(board: &BoardState) -> u64 {
        if board.history.is_cache_valid() {
            return board.history.current_cache().pinned;
        }
        board.pinned_pieces(board.side_to_move).0
    }

    #[inline(always)]
    pub fn cached_pinners(board: &BoardState) -> u64 {
        if board.history.is_cache_valid() {
            return board.history.current_cache().pinners;
        }
        0
    }

    #[inline(always)]
    pub fn cached_all_threats(board: &BoardState) -> u64 {
        if board.history.is_cache_valid() {
            return board.history.current_cache().all_threats;
        }
        0
    }

    #[inline(always)]
    pub fn cached_check_squares(board: &BoardState) -> [u64; 6] {
        if board.history.is_cache_valid() {
            return board.history.current_cache().check_squares;
        }
        board.check_squares(board.side_to_move.other())
    }

    #[inline(always)]
    pub fn from_cache(board: &BoardState) -> Self {
        Self::compute(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn snapshot_matches_fresh_queries() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let stm = board.side_to_move;
        let nt = NodeThreats::compute(&board);
        assert_eq!(nt.checkers, board.checkers(stm).0);
        assert_eq!(nt.pinned, board.pinned_pieces(stm).0);
        assert_eq!(nt.pinned_them, board.pinned_pieces(stm.other()).0);
        assert_eq!(nt.threats_us, board.threat_by_lesser(stm));
        assert_eq!(nt.threats_them, board.threat_by_lesser(stm.other()));
        assert_eq!(nt.check_squares, board.check_squares(stm.other()));
        assert!(!nt.in_check());
        assert_eq!(nt.checks(), 0);
    }

    #[test]
    fn qsearch_snapshot_matches_full_on_qsearch_fields() {
        let fens = [
            STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1",
            "7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1",
            "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1",
        ];
        for fen in fens {
            let board = BoardState::parse_fen(fen);
            let full = NodeThreats::compute(&board);
            let light = NodeThreats::compute_for_qsearch(&board);
            assert_eq!(light.checkers, full.checkers, "{fen}");
            assert_eq!(light.pinned, full.pinned, "{fen}");
            assert_eq!(light.threats_us, full.threats_us, "{fen}");
        }
    }

    #[test]
    fn snapshot_detects_check() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let nt = NodeThreats::compute(&board);
        assert!(nt.in_check());
        assert_eq!(nt.checks(), board.checkers(Side::Black).0.count_ones());
    }
}
