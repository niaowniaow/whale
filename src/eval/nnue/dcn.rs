use crate::board::state::BoardState;
use crate::common::piece::Piece;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthRegime {
    Tactical,
    Blended,
    Strategic,
}

#[derive(Debug, Clone, Copy)]
pub struct DcnConfig {
    pub enabled: bool,
    pub tactical_threshold: u8,
    pub strategic_threshold: u8,
}

impl Default for DcnConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tactical_threshold: 5,
            strategic_threshold: 15,
        }
    }
}

pub struct DcnModel;

impl DcnModel {
    pub const EMBED_DIM: usize = 16;
    pub const MAX_DEPTH: usize = 64;

    #[inline(always)]
    pub fn get_regime(depth: u8) -> DepthRegime {
        if depth < 5 {
            DepthRegime::Tactical
        } else if depth <= 15 {
            DepthRegime::Blended
        } else {
            DepthRegime::Strategic
        }
    }

    #[inline(always)]
    pub fn compute_film_params(depth: u8) -> (i32, i32) {
        let d = (depth as usize).min(Self::MAX_DEPTH - 1);
        let embed = &DEPTH_EMBEDDINGS[d];

        let mut gamma_raw = 0i32;
        let mut beta_raw = 0i32;

        for (i, &val) in embed.iter().enumerate() {
            gamma_raw += val as i32 * FILM_GAMMA_WEIGHTS[i] as i32;
            beta_raw += val as i32 * FILM_BETA_WEIGHTS[i] as i32;
        }

        let gamma = 256 + (gamma_raw / 128).clamp(-64, 64);
        let beta = (beta_raw / 128).clamp(-120, 120);

        (gamma, beta)
    }

    pub fn condition_evaluation(
        raw_eval: i16,
        depth: u8,
        board: &BoardState,
        config: DcnConfig,
    ) -> i16 {
        if !config.enabled {
            return raw_eval;
        }

        let (gamma, beta) = Self::compute_film_params(depth);
        let modulated = (raw_eval as i64 * gamma as i64) / 256 + beta as i64;

        let regime = Self::get_regime(depth);
        let positional_adjustment = match regime {
            DepthRegime::Tactical => {
                let us = board.side_to_move;
                let them = us.other();
                let checks = board.checkers(us).0.count_ones() as i64;
                let threats = (board.threat_by_lesser(us)[Piece::Queen as usize]
                    & board.get_pieces(us, Piece::Queen).0)
                    .count_ones() as i64;
                let enemy_threats = (board.threat_by_lesser(them)[Piece::Queen as usize]
                    & board.get_pieces(them, Piece::Queen).0)
                    .count_ones() as i64;

                (enemy_threats - threats) * 28 - checks * 32
            }
            DepthRegime::Strategic => {
                let us = board.side_to_move;
                let them = us.other();
                let us_pawns = board.get_pieces(us, Piece::Pawn).0.count_ones() as i64;
                let them_pawns = board.get_pieces(them, Piece::Pawn).0.count_ones() as i64;
                let us_bishops = board.get_pieces(us, Piece::Bishop).0.count_ones() as i64;
                let them_bishops = board.get_pieces(them, Piece::Bishop).0.count_ones() as i64;

                let bishop_pair_bonus = if us_bishops >= 2 && them_bishops < 2 {
                    24
                } else if them_bishops >= 2 && us_bishops < 2 {
                    -24
                } else {
                    0
                };

                (us_pawns - them_pawns) * 8 + bishop_pair_bonus
            }
            DepthRegime::Blended => {
                let d = depth as i64;
                let tactical_weight = (15 - d).max(0);
                let strategic_weight = (d - 4).max(0);

                let us = board.side_to_move;
                let checks = board.checkers(us).0.count_ones() as i64;
                let tactical_signal = -checks * 16;

                let us_pawns = board.get_pieces(us, Piece::Pawn).0.count_ones() as i64;
                let them_pawns = board.get_pieces(us.other(), Piece::Pawn).0.count_ones() as i64;
                let strategic_signal = (us_pawns - them_pawns) * 6;

                (tactical_signal * tactical_weight + strategic_signal * strategic_weight) / 11
            }
        };

        (modulated + positional_adjustment).clamp(-29000, 29000) as i16
    }
}

const DEPTH_EMBEDDINGS: [[i16; DcnModel::EMBED_DIM]; DcnModel::MAX_DEPTH] = {
    let mut table = [[0i16; DcnModel::EMBED_DIM]; DcnModel::MAX_DEPTH];
    let mut d = 0usize;
    while d < DcnModel::MAX_DEPTH {
        let mut i = 0usize;
        while i < DcnModel::EMBED_DIM {
            let freq = i as i32 * 3 + 1;
            let val = ((d as i32 * freq * 13) % 255) - 127;
            table[d][i] = val as i16;
            i += 1;
        }
        d += 1;
    }
    table
};

const FILM_GAMMA_WEIGHTS: [i16; DcnModel::EMBED_DIM] =
    [12, 10, -8, 6, -4, 9, 7, -5, 8, -6, 11, -7, 5, -4, 8, -6];

const FILM_BETA_WEIGHTS: [i16; DcnModel::EMBED_DIM] =
    [-6, 8, 14, -10, 5, -8, 12, -7, 6, -9, 8, -12, 7, -5, 10, -8];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_dcn_regime_classification() {
        assert_eq!(DcnModel::get_regime(0), DepthRegime::Tactical);
        assert_eq!(DcnModel::get_regime(4), DepthRegime::Tactical);
        assert_eq!(DcnModel::get_regime(5), DepthRegime::Blended);
        assert_eq!(DcnModel::get_regime(15), DepthRegime::Blended);
        assert_eq!(DcnModel::get_regime(16), DepthRegime::Strategic);
        assert_eq!(DcnModel::get_regime(30), DepthRegime::Strategic);
    }

    #[test]
    fn test_film_params_bounded() {
        for d in 0..64 {
            let (gamma, beta) = DcnModel::compute_film_params(d);
            assert!((192..=320).contains(&gamma));
            assert!((-120..=120).contains(&beta));
        }
    }

    #[test]
    fn test_dcn_disabled_returns_raw_eval() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let config = DcnConfig {
            enabled: false,
            tactical_threshold: 5,
            strategic_threshold: 15,
        };
        let eval = DcnModel::condition_evaluation(120, 10, &board, config);
        assert_eq!(eval, 120);
    }

    #[test]
    fn test_dcn_consistency_across_adjacent_depths() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let config = DcnConfig::default();
        let eval19 = DcnModel::condition_evaluation(100, 19, &board, config);
        let eval20 = DcnModel::condition_evaluation(100, 20, &board, config);
        let diff = (eval20 as i32 - eval19 as i32).abs();
        assert!(diff <= 60);
    }
}
