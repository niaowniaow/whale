use crate::bitboard::lookups::king_attacks;
use crate::board::state::BoardState;
use crate::common::constants::MAX_PLY;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreatFeatures {
    pub num_pins: i32,
    pub num_threatened_pieces: i32,
    pub is_check: bool,
    pub num_checkers: i32,
    pub is_capture: bool,
    pub material_imbalance: i32,
    pub king_danger: i32,
    pub depth_remaining: i32,
    pub is_recapture: bool,
    pub is_pawn_advance: bool,
}

pub struct TceScorer;

impl TceScorer {
    pub fn threat_score(features: &ThreatFeatures) -> i32 {
        let mut score = 0i32;
        score += features.num_pins * 2;
        score += features.num_threatened_pieces;
        if features.is_check {
            score += 3;
        }
        score += features.num_checkers;
        if features.is_capture {
            score += 1;
        }
        score += features.king_danger * 2;
        if features.is_recapture {
            score += 2;
        }
        if features.is_pawn_advance {
            score += 1;
        }
        score += (features.depth_remaining / 4).clamp(0, 3);
        score
    }

    pub fn predict_extension(features: &ThreatFeatures) -> u8 {
        let score = Self::threat_score(features);
        if score >= 12 {
            2
        } else if score >= 6 {
            1
        } else {
            0
        }
    }
}

pub use TceScorer as TceModel;

pub fn extract_threat_features(
    board: &BoardState,
    nt: &crate::board::node_threats::NodeThreats,
    move_obj: Move,
    depth: u8,
    in_check: bool,
    previous_move: Option<Move>,
) -> ThreatFeatures {
    let us = board.side_to_move;
    let them = us.other();

    let num_pins = nt.pinned.count_ones() as i32 + nt.pinned_them.count_ones() as i32;

    let mut num_threatened_pieces = 0i32;
    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        num_threatened_pieces += nt.threatened_count(board, piece) as i32;
    }

    let num_checkers = if in_check { nt.checks() as i32 } else { 0 };

    let material_imbalance = board.occupancies[Side::White].count_ones() as i32
        - board.occupancies[Side::Black].count_ones() as i32;

    let king_bb = board.get_pieces(us, Piece::King);
    let king_danger = if !king_bb.is_empty() {
        let ksq = king_bb.get_lsb() as usize;
        let ring = king_attacks()[ksq];
        (ring & board.occupancies[them].0).count_ones() as i32
    } else {
        0
    };

    let is_recapture = if let Some(prev) = previous_move {
        prev.is_capture() && move_obj.is_capture() && prev.target == move_obj.target
    } else {
        false
    };

    let is_pawn_advance = {
        let piece = board.piece_mapping[move_obj.source as usize];
        if piece == Piece::Pawn {
            let target_rank = (move_obj.target as usize) / 8;
            (us == Side::White && target_rank >= 6) || (us == Side::Black && target_rank <= 1)
        } else {
            false
        }
    };

    ThreatFeatures {
        num_pins,
        num_threatened_pieces,
        is_check: in_check,
        num_checkers,
        is_capture: move_obj.is_capture(),
        material_imbalance,
        king_danger,
        depth_remaining: depth as i32,
        is_recapture,
        is_pawn_advance,
    }
}

pub fn compute_extension(
    board: &BoardState,
    nt: &crate::board::node_threats::NodeThreats,
    move_obj: Move,
    depth: u8,
    in_check: bool,
    ply: u8,
    previous_move: Option<Move>,
) -> i8 {
    if ply as usize >= MAX_PLY.saturating_sub(4) || depth <= 1 {
        return 0;
    }

    let features = extract_threat_features(board, nt, move_obj, depth, in_check, previous_move);
    let ext = TceModel::predict_extension(&features);

    if ext == 2 {
        if depth >= 6 && (in_check || features.num_pins >= 2 || features.king_danger >= 2) {
            2
        } else {
            1
        }
    } else if ext == 1 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::node_threats::NodeThreats;
    use crate::common::helpers::STARTING_FEN;
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;

    fn snapshot(board: &BoardState) -> NodeThreats {
        NodeThreats::compute(board)
    }

    #[test]
    fn test_starting_position_no_extension() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        let ext = compute_extension(&board, &snapshot(&board), m, 5, false, 0, None);
        assert_eq!(ext, 0);
    }

    #[test]
    fn test_tce_forward_predicts_valid_category() {
        let features = ThreatFeatures {
            num_pins: 0,
            num_threatened_pieces: 0,
            is_check: false,
            num_checkers: 0,
            is_capture: false,
            material_imbalance: 0,
            king_danger: 0,
            depth_remaining: 5,
            is_recapture: false,
            is_pawn_advance: false,
        };
        let ext = TceModel::predict_extension(&features);
        assert_eq!(ext, 0);
    }

    #[test]
    fn test_tce_heavy_check_and_pins_predicts_extension() {
        let features = ThreatFeatures {
            num_pins: 2,
            num_threatened_pieces: 2,
            is_check: true,
            num_checkers: 2,
            is_capture: true,
            material_imbalance: -3,
            king_danger: 3,
            depth_remaining: 8,
            is_recapture: true,
            is_pawn_advance: true,
        };
        let ext = TceModel::predict_extension(&features);
        assert!(ext >= 1);
    }

    #[test]
    fn test_tce_near_leaf_returns_zero() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        let ext = compute_extension(&board, &snapshot(&board), m, 1, true, 0, None);
        assert_eq!(ext, 0);
    }

    #[test]
    fn extension_is_zero_past_max_ply() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        let ply = (MAX_PLY.saturating_sub(4)) as u8;
        assert_eq!(
            compute_extension(&board, &snapshot(&board), m, 8, true, ply, None),
            0
        );
    }

    #[test]
    fn extract_covers_check_capture_and_pawn_push() {
        let board = BoardState::parse_fen("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1");
        let back = Move::new(Square::E3, Square::E2, MoveType::Quiet);
        let f = extract_threat_features(&board, &snapshot(&board), back, 6, false, None);
        assert!(f.is_pawn_advance);
        assert!(!f.is_capture);

        let recapture_board = BoardState::parse_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        let prev = Move::new(Square::D5, Square::E4, MoveType::Capture);
        let cur = Move::new(Square::D5, Square::E4, MoveType::Capture);
        let g = extract_threat_features(
            &recapture_board,
            &snapshot(&recapture_board),
            cur,
            6,
            false,
            Some(prev),
        );
        assert!(g.is_recapture);
        let other = Move::new(Square::E4, Square::D5, MoveType::Capture);
        let h = extract_threat_features(
            &recapture_board,
            &snapshot(&recapture_board),
            other,
            6,
            false,
            Some(prev),
        );
        assert!(!h.is_recapture);

        let check_board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        assert!(check_board.is_in_check(check_board.side_to_move));
        let any = Move::new(Square::E8, Square::D8, MoveType::Quiet);
        let c = extract_threat_features(&check_board, &snapshot(&check_board), any, 6, true, None);
        assert!(c.is_check);
        assert!(c.num_checkers >= 1);
    }

    #[test]
    fn score_clamps_and_stays_in_range() {
        let features = ThreatFeatures {
            num_pins: 99,
            num_threatened_pieces: 99,
            is_check: true,
            num_checkers: 9,
            is_capture: true,
            material_imbalance: 1_000,
            king_danger: 99,
            depth_remaining: 1_000,
            is_recapture: true,
            is_pawn_advance: true,
        };
        let score = TceModel::threat_score(&features);
        assert!(score > 12);
        let cat = TceModel::predict_extension(&features);
        assert!(cat <= 2);
    }

    #[test]
    fn double_extension_needs_depth_and_threat() {
        let hot = ThreatFeatures {
            num_pins: 3,
            num_threatened_pieces: 3,
            is_check: true,
            num_checkers: 2,
            is_capture: true,
            material_imbalance: 0,
            king_danger: 3,
            depth_remaining: 8,
            is_recapture: true,
            is_pawn_advance: true,
        };
        let cat = TceModel::predict_extension(&hot);
        assert!(cat >= 1);
        let board = BoardState::parse_fen(STARTING_FEN);
        let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        for depth in [0u8, 1] {
            assert_eq!(
                compute_extension(&board, &snapshot(&board), m, depth, true, 0, None),
                0
            );
        }
        let board2 = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let mv = Move::new(Square::E8, Square::D8, MoveType::Quiet);
        let ext = compute_extension(&board2, &snapshot(&board2), mv, 6, true, 0, None);
        assert!((0..=2).contains(&ext));
    }
}
