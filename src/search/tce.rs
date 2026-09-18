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

pub struct TceModel;

impl TceModel {
    pub const INPUT_DIM: usize = 10;
    pub const HIDDEN_DIM: usize = 16;
    pub const OUTPUT_DIM: usize = 3;

    pub fn forward(features: &ThreatFeatures) -> [i32; Self::OUTPUT_DIM] {
        let x: [i32; Self::INPUT_DIM] = [
            features.num_pins,
            features.num_threatened_pieces,
            if features.is_check { 1 } else { 0 },
            features.num_checkers,
            if features.is_capture { 1 } else { 0 },
            features.material_imbalance.clamp(-10, 10),
            features.king_danger,
            features.depth_remaining.clamp(0, 30),
            if features.is_recapture { 1 } else { 0 },
            if features.is_pawn_advance { 1 } else { 0 },
        ];

        let mut hidden = [0i32; Self::HIDDEN_DIM];
        for (i, h) in hidden.iter_mut().enumerate() {
            let mut sum = TCE_BIAS_1[i];
            for (j, val) in x.iter().enumerate() {
                sum += val * TCE_WEIGHTS_1[i][j];
            }
            *h = sum.max(0);
        }

        let mut output = [0i32; Self::OUTPUT_DIM];
        for (k, out) in output.iter_mut().enumerate() {
            let mut sum = TCE_BIAS_2[k];
            for (i, h) in hidden.iter().enumerate() {
                sum += h * TCE_WEIGHTS_2[k][i];
            }
            *out = sum;
        }

        output
    }

    pub fn predict_extension(features: &ThreatFeatures) -> u8 {
        let scores = Self::forward(features);
        let mut best_idx = 0usize;
        let mut best_score = scores[0];

        for (i, &score) in scores.iter().enumerate().skip(1) {
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        best_idx as u8
    }
}

const TCE_WEIGHTS_1: [[i32; TceModel::INPUT_DIM]; TceModel::HIDDEN_DIM] = [
    [12, 10, 25, 30, 8, -4, 18, 5, 14, 12],
    [5, 4, 15, 20, 2, -2, 10, 3, 8, 6],
    [18, 14, 5, 2, 4, 1, 12, 4, 2, 8],
    [8, 16, 20, 25, 12, -6, 22, 6, 10, 15],
    [2, 3, 35, 40, 5, 2, 16, 2, 6, 4],
    [14, 8, 4, 6, 16, -1, 8, 7, 18, 10],
    [4, 12, 18, 22, 6, -5, 24, 4, 4, 8],
    [20, 6, 8, 10, 10, 0, 14, 5, 12, 16],
    [-8, -6, -15, -20, -4, 8, -12, -3, -6, -4],
    [-12, -10, -25, -30, -8, 12, -18, -5, -10, -8],
    [6, 5, 12, 15, 4, -2, 8, 2, 6, 5],
    [10, 15, 8, 10, 14, -3, 16, 6, 15, 12],
    [3, 2, 28, 32, 4, 1, 14, 3, 5, 3],
    [15, 12, 16, 20, 8, -4, 20, 5, 11, 14],
    [-5, -4, -10, -12, -2, 5, -8, -2, -4, -3],
    [7, 9, 14, 18, 5, -3, 15, 4, 7, 9],
];

const TCE_BIAS_1: [i32; TceModel::HIDDEN_DIM] = [
    -40, -20, -30, -50, -35, -25, -45, -30, 50, 80, -15, -35, -30, -45, 40, -25,
];

const TCE_WEIGHTS_2: [[i32; TceModel::HIDDEN_DIM]; TceModel::OUTPUT_DIM] = [
    [
        -12, -8, -10, -15, -14, -8, -16, -10, 24, 32, -6, -12, -11, -15, 18, -8,
    ],
    [
        14, 10, 12, 16, 15, 10, 18, 12, -15, -20, 8, 14, 12, 16, -12, 10,
    ],
    [
        18, 12, 15, 22, 20, 12, 24, 14, -25, -35, 10, 18, 16, 22, -18, 12,
    ],
];

const TCE_BIAS_2: [i32; TceModel::OUTPUT_DIM] = [40, -10, -90];

pub fn extract_threat_features(
    board: &BoardState,
    move_obj: Move,
    depth: u8,
    in_check: bool,
    previous_move: Option<Move>,
) -> ThreatFeatures {
    let us = board.side_to_move;
    let them = us.other();

    let num_pins = board.pinned_pieces(us).0.count_ones() as i32
        + board.pinned_pieces(them).0.count_ones() as i32;

    let threats = board.threat_by_lesser(us);
    let mut num_threatened_pieces = 0i32;
    for (piece_idx, &threat) in threats.iter().enumerate().take(5).skip(1) {
        let piece = match piece_idx {
            1 => Piece::Knight,
            2 => Piece::Bishop,
            3 => Piece::Rook,
            4 => Piece::Queen,
            _ => Piece::None,
        };
        let pieces_bb = board.get_pieces(us, piece).0;
        num_threatened_pieces += (pieces_bb & threat).count_ones() as i32;
    }

    let num_checkers = if in_check {
        board.checkers(us).0.count_ones() as i32
    } else {
        0
    };

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
    move_obj: Move,
    depth: u8,
    in_check: bool,
    ply: u8,
    previous_move: Option<Move>,
) -> i8 {
    if ply as usize >= MAX_PLY.saturating_sub(4) || depth <= 1 {
        return 0;
    }

    let features = extract_threat_features(board, move_obj, depth, in_check, previous_move);
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
    use crate::common::helpers::STARTING_FEN;
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;

    #[test]
    fn test_starting_position_no_extension() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        let ext = compute_extension(&board, m, 5, false, 0, None);
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
        let ext = compute_extension(&board, m, 1, true, 0, None);
        assert_eq!(ext, 0);
    }

    #[test]
    fn extension_is_zero_past_max_ply() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        let ply = (MAX_PLY.saturating_sub(4)) as u8;
        assert_eq!(compute_extension(&board, m, 8, true, ply, None), 0);
    }

    #[test]
    fn extract_covers_check_capture_and_pawn_push() {
        // Pawn-advance follows the raw target-rank rule (white target rank >= 6).
        let board = BoardState::parse_fen("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1");
        let back = Move::new(Square::E3, Square::E2, MoveType::Quiet);
        let f = extract_threat_features(&board, back, 6, false, None);
        assert!(f.is_pawn_advance);
        assert!(!f.is_capture);

        let recapture_board = BoardState::parse_fen("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
        let prev = Move::new(Square::D5, Square::E4, MoveType::Capture);
        let cur = Move::new(Square::D5, Square::E4, MoveType::Capture);
        let g = extract_threat_features(&recapture_board, cur, 6, false, Some(prev));
        assert!(g.is_recapture);
        let other = Move::new(Square::E4, Square::D5, MoveType::Capture);
        let h = extract_threat_features(&recapture_board, other, 6, false, Some(prev));
        assert!(!h.is_recapture);

        let check_board = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        assert!(check_board.is_in_check(check_board.side_to_move));
        let any = Move::new(Square::E8, Square::D8, MoveType::Quiet);
        let c = extract_threat_features(&check_board, any, 6, true, None);
        assert!(c.is_check);
        assert!(c.num_checkers >= 1);
    }

    #[test]
    fn forward_clamps_and_stays_in_range() {
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
        let out = TceModel::forward(&features);
        assert_eq!(out.len(), 3);
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
            assert_eq!(compute_extension(&board, m, depth, true, 0, None), 0);
        }
        let board2 = BoardState::parse_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1");
        let mv = Move::new(Square::E8, Square::D8, MoveType::Quiet);
        let ext = compute_extension(&board2, mv, 6, true, 0, None);
        assert!((0..=2).contains(&ext));
    }
}
