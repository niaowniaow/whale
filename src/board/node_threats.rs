use crate::board::state::BoardState;
use crate::common::piece::Piece;

#[cfg(test)]
use crate::common::side::Side;

/// Per-node threat snapshot, computed ONCE per search node and shared by
/// legality, eval (DCN), quiet move scoring, TCE and LQT.
///
/// Rationale (Stockfish `StateInfo` / Reckless cached-threats concept):
/// `checkers()`, `pinned_pieces()` and especially `threat_by_lesser()` were
/// each recomputed several times per node (legality per move, DCN eval,
/// quiet scoring, TCE/LQT features). One snapshot replaces all of them with
/// zero behavior change — every consumer reads the same values.
#[derive(Clone, Copy, Debug)]
pub struct NodeThreats {
    /// Enemy checkers bitboard (`!= 0` ⟺ side to move is in check).
    pub checkers: u64,
    /// Own pinned pieces bitboard.
    pub pinned: u64,
    /// Enemy pinned pieces bitboard (TCE pin pressure signal).
    pub pinned_them: u64,
    /// Squares attacked by enemy lesser pieces, per own piece type.
    pub threats_us: [u64; 6],
    /// Squares attacked by our lesser pieces, per enemy piece type.
    pub threats_them: [u64; 6],
    /// Squares from which our pieces give check to the enemy king.
    pub check_squares: [u64; 6],
}

impl NodeThreats {
    #[inline(always)]
    pub fn compute(board: &BoardState) -> Self {
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
    pub fn in_check(&self) -> bool {
        self.checkers != 0
    }

    /// Number of checking pieces (DCN tactical signal).
    #[inline(always)]
    pub fn checks(&self) -> u32 {
        self.checkers.count_ones()
    }

    /// Own queens standing on enemy-threatened squares (DCN tactical signal).
    #[inline(always)]
    pub fn queen_threat_us(&self, board: &BoardState) -> u32 {
        (self.threats_us[Piece::Queen as usize]
            & board.get_pieces(board.side_to_move, Piece::Queen).0)
            .count_ones()
    }

    /// Enemy queens standing on our-threatened squares (DCN tactical signal).
    #[inline(always)]
    pub fn queen_threat_them(&self, board: &BoardState) -> u32 {
        let them = board.side_to_move.other();
        (self.threats_them[Piece::Queen as usize] & board.get_pieces(them, Piece::Queen).0)
            .count_ones()
    }

    /// Own pieces of `piece` standing on enemy-threatened squares
    /// (LQT discovered-attacks signal).
    #[inline(always)]
    pub fn threatened_count(&self, board: &BoardState, piece: Piece) -> u32 {
        let idx = piece as usize;
        if idx == 0 || idx >= 5 {
            return 0;
        }
        (board.get_pieces(board.side_to_move, piece).0 & self.threats_us[idx]).count_ones()
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
    fn snapshot_detects_check() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let nt = NodeThreats::compute(&board);
        assert!(nt.in_check());
        assert_eq!(nt.checks(), board.checkers(Side::Black).0.count_ones());
    }
}
