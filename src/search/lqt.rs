use crate::board::state::BoardState;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LqtFeatures {
    pub stand_pat_score: i16,
    pub num_pinned_pieces: i32,
    pub num_discovered_attacks: i32,
    pub mobility_delta: i32,
    pub king_distance: i32,
    pub depth_in_qs: u8,
    pub window_width: i32,
}

pub struct LqtModel;

impl LqtModel {
    pub const INPUT_DIM: usize = 7;
    pub const HIDDEN_1: usize = 16;
    pub const HIDDEN_2: usize = 8;

    pub fn forward(features: &LqtFeatures) -> i32 {
        let x: [i32; Self::INPUT_DIM] = [
            features.stand_pat_score as i32 / 32,
            features.num_pinned_pieces,
            features.num_discovered_attacks,
            features.mobility_delta.clamp(-15, 15),
            features.king_distance.clamp(1, 8),
            features.depth_in_qs as i32,
            (features.window_width / 64).clamp(0, 30),
        ];

        let mut h1 = [0i32; Self::HIDDEN_1];
        for (i, h) in h1.iter_mut().enumerate() {
            let mut sum = LQT_B1[i];
            for (j, val) in x.iter().enumerate() {
                sum += val * LQT_W1[i][j];
            }
            *h = sum.max(0);
        }

        let mut h2 = [0i32; Self::HIDDEN_2];
        for (i, h) in h2.iter_mut().enumerate() {
            let mut sum = LQT_B2[i];
            for (j, val) in h1.iter().enumerate() {
                sum += val * LQT_W2[i][j];
            }
            *h = sum.max(0);
        }

        let mut logit = LQT_B3;
        for (i, val) in h2.iter().enumerate() {
            logit += val * LQT_W3[i];
        }

        logit
    }

    pub fn predict_probability(features: &LqtFeatures) -> i32 {
        let logit = Self::forward(features);
        let clamped = logit.clamp(-1000, 1000);
        (clamped + 1000) * 1024 / 2000
    }
}

const LQT_W1: [[i32; LqtModel::INPUT_DIM]; LqtModel::HIDDEN_1] = [
    [-4, 18, 16, -3, -6, -20, 12],
    [-2, 12, 14, -2, -4, -15, 8],
    [5, -6, -8, 4, 3, -10, -4],
    [-6, 22, 20, -5, -8, -25, 16],
    [3, -4, -6, 2, 2, -8, -2],
    [-3, 14, 12, -2, -5, -18, 10],
    [-5, 20, 18, -4, -7, -22, 14],
    [4, -8, -10, 5, 4, -12, -6],
    [-2, 10, 8, -1, -3, -14, 6],
    [-6, 24, 22, -6, -9, -28, 18],
    [2, -3, -5, 1, 1, -6, -1],
    [-4, 16, 15, -3, -6, -19, 11],
    [6, -10, -12, 6, 5, -15, -8],
    [-3, 13, 11, -2, -4, -16, 9],
    [-5, 21, 19, -5, -8, -24, 15],
    [1, -2, -4, 1, 1, -5, 0],
];

const LQT_B1: [i32; LqtModel::HIDDEN_1] = [
    -15, -10, 20, -25, 15, -12, -20, 25, -8, -30, 10, -16, 30, -11, -22, 5,
];

const LQT_W2: [[i32; LqtModel::HIDDEN_1]; LqtModel::HIDDEN_2] = [
    [
        14, 10, -12, 18, -10, 11, 16, -15, 8, 20, -6, 13, -18, 10, 17, -4,
    ],
    [11, 8, -9, 14, -8, 9, 12, -12, 6, 16, -5, 10, -14, 8, 13, -3],
    [
        -15, -11, 14, -20, 12, -13, -18, 18, -9, -22, 7, -15, 20, -11, -19, 5,
    ],
    [
        16, 12, -14, 21, -12, 13, 19, -17, 9, 23, -7, 15, -21, 12, 20, -5,
    ],
    [
        -12, -9, 11, -16, 9, -10, -14, 14, -7, -18, 6, -12, 16, -9, -15, 4,
    ],
    [
        13, 9, -11, 17, -10, 10, 15, -14, 8, 19, -6, 12, -17, 10, 16, -4,
    ],
    [
        18, 13, -16, 23, -13, 14, 21, -19, 10, 25, -8, 17, -23, 13, 22, -6,
    ],
    [
        -10, -7, 9, -13, 8, -8, -11, 11, -6, -14, 5, -9, 13, -7, -12, 3,
    ],
];

const LQT_B2: [i32; LqtModel::HIDDEN_2] = [-15, -10, 20, -20, 15, -12, -25, 10];

const LQT_W3: [i32; LqtModel::HIDDEN_2] = [18, 14, -22, 22, -18, 16, 25, -14];
const LQT_B3: i32 = -30;

pub fn extract_lqt_features(
    board: &BoardState,
    nt: &crate::board::node_threats::NodeThreats,
    stand_pat: i16,
    alpha: i16,
    beta: i16,
    depth_in_qs: u8,
) -> LqtFeatures {
    let us = board.side_to_move;
    let them = us.other();

    let num_pinned_pieces = nt.pinned.count_ones() as i32;

    let mut num_discovered_attacks = 0i32;
    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        num_discovered_attacks += nt.threatened_count(board, piece) as i32;
    }

    let mobility_delta =
        board.occupancies[us].count_ones() as i32 - board.occupancies[them].count_ones() as i32;

    let white_king = board.get_pieces(Side::White, Piece::King);
    let black_king = board.get_pieces(Side::Black, Piece::King);
    let king_distance = if !white_king.is_empty() && !black_king.is_empty() {
        let w_sq = Square::from(white_king.get_lsb() as usize);
        let b_sq = Square::from(black_king.get_lsb() as usize);
        let r_diff = (w_sq.rank() as i32 - b_sq.rank() as i32).abs();
        let f_diff = (w_sq.file() as i32 - b_sq.file() as i32).abs();
        r_diff.max(f_diff)
    } else {
        4
    };

    let window_width = (beta as i32 - alpha as i32).clamp(0, 2000);

    LqtFeatures {
        stand_pat_score: stand_pat,
        num_pinned_pieces,
        num_discovered_attacks,
        mobility_delta,
        king_distance,
        depth_in_qs,
        window_width,
    }
}

pub fn should_continue_quiescence(
    board: &BoardState,
    nt: &crate::board::node_threats::NodeThreats,
    stand_pat: i16,
    alpha: i16,
    beta: i16,
    depth_in_qs: u8,
) -> bool {
    if depth_in_qs >= 4 {
        return false;
    }

    let features = extract_lqt_features(board, nt, stand_pat, alpha, beta, depth_in_qs);
    let prob = LqtModel::predict_probability(&features);

    prob >= 620
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_starting_position_quiescence_terminates() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        let should_continue = should_continue_quiescence(&board, &nt, 0, -50, 50, 0);
        assert!(!should_continue);
    }

    #[test]
    fn test_deep_in_qs_always_terminates() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        assert!(!should_continue_quiescence(&board, &nt, 0, -50, 50, 4));
        assert!(!should_continue_quiescence(&board, &nt, 0, -50, 50, 10));
    }

    #[test]
    fn test_high_pin_and_threats_continues_qs() {
        let features = LqtFeatures {
            stand_pat_score: -100,
            num_pinned_pieces: 3,
            num_discovered_attacks: 3,
            mobility_delta: -4,
            king_distance: 2,
            depth_in_qs: 1,
            window_width: 300,
        };
        let prob = LqtModel::predict_probability(&features);
        assert!(prob >= 620);
    }

    #[test]
    fn test_calm_endgame_terminates() {
        let features = LqtFeatures {
            stand_pat_score: 50,
            num_pinned_pieces: 0,
            num_discovered_attacks: 0,
            mobility_delta: 2,
            king_distance: 6,
            depth_in_qs: 2,
            window_width: 40,
        };
        let prob = LqtModel::predict_probability(&features);
        assert!(prob < 620);
    }
}
