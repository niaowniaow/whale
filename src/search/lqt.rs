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

pub struct LqtScorer;

impl LqtScorer {
    pub fn tactical_score(features: &LqtFeatures) -> i32 {
        let mut score = 512i32;
        score += features.num_pinned_pieces * 70;
        score += features.num_discovered_attacks * 70;
        score += (8 - features.king_distance.clamp(1, 8)) * 25;
        score -= features.depth_in_qs as i32 * 110;
        score += ((features.window_width - 100) / 4).clamp(-60, 60);
        score += features.mobility_delta.clamp(-8, 8) * 4;
        score
    }

    pub fn predict_probability(features: &LqtFeatures) -> i32 {
        Self::tactical_score(features).clamp(0, 1024)
    }
}

pub use LqtScorer as LqtModel;

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
    threshold: i16,
) -> bool {
    if depth_in_qs >= 4 {
        return false;
    }

    let features = extract_lqt_features(board, nt, stand_pat, alpha, beta, depth_in_qs);
    let prob = LqtModel::predict_probability(&features);

    prob >= threshold as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_starting_position_quiescence_terminates() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        let should_continue = should_continue_quiescence(&board, &nt, 0, -50, 50, 0, 620);
        assert!(!should_continue);
    }

    #[test]
    fn test_deep_in_qs_always_terminates() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        assert!(!should_continue_quiescence(&board, &nt, 0, -50, 50, 4, 620));
        assert!(!should_continue_quiescence(
            &board, &nt, 0, -50, 50, 10, 620
        ));
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
