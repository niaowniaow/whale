#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlpFeatures {
    pub eval_margin: i32,
    pub depth: i32,
    pub move_index: usize,
    pub is_null_move: bool,
    pub is_capture: bool,
    pub is_pv: bool,
    pub in_check: bool,
    pub history_score: i32,
    pub momentum: i32,
}

pub struct AlpModel;

impl AlpModel {
    pub const INPUT_DIM: usize = 9;
    pub const HIDDEN1_DIM: usize = 16;
    pub const HIDDEN2_DIM: usize = 8;

    pub fn forward(features: &AlpFeatures) -> i32 {
        if features.is_pv || features.in_check {
            return 0;
        }

        let norm_eval = (features.eval_margin / 32).clamp(-128, 128);
        let norm_depth = features.depth.clamp(0, 32);
        let norm_move = (features.move_index as i32).clamp(0, 64);
        let norm_history = (features.history_score / 512).clamp(-64, 64);
        let norm_momentum = (features.momentum / 32).clamp(-64, 64);

        let x: [i32; Self::INPUT_DIM] = [
            norm_eval,
            norm_depth,
            norm_move,
            if features.is_null_move { 1 } else { 0 },
            if features.is_capture { 1 } else { 0 },
            if features.is_pv { 1 } else { 0 },
            if features.in_check { 1 } else { 0 },
            norm_history,
            norm_momentum,
        ];

        let mut h1 = [0i32; Self::HIDDEN1_DIM];
        for i in 0..Self::HIDDEN1_DIM {
            let mut sum = ALP_BIAS_1[i];
            for j in 0..Self::INPUT_DIM {
                sum += x[j] * ALP_WEIGHTS_1[i][j];
            }
            h1[i] = sum.max(0);
        }

        let mut h2 = [0i32; Self::HIDDEN2_DIM];
        for i in 0..Self::HIDDEN2_DIM {
            let mut sum = ALP_BIAS_2[i];
            for j in 0..Self::HIDDEN1_DIM {
                sum += h1[j] * ALP_WEIGHTS_2[i][j];
            }
            h2[i] = sum.max(0);
        }

        let mut out = ALP_BIAS_3;
        for i in 0..Self::HIDDEN2_DIM {
            out += h2[i] * ALP_WEIGHTS_3[i];
        }

        (out / 64).clamp(0, 100)
    }

    pub fn predict_safe_prune_prob(features: &AlpFeatures) -> u8 {
        Self::forward(features) as u8
    }

    pub fn should_prune(features: &AlpFeatures, threshold_percent: u8) -> bool {
        Self::predict_safe_prune_prob(features) >= threshold_percent
    }

    pub fn adversarial_loss(pruned_ratio: f32, move_diff: f32, lambda1: f32, lambda2: f32) -> f32 {
        lambda1 * (1.0 - pruned_ratio).max(0.0) + lambda2 * move_diff
    }
}

const ALP_WEIGHTS_1: [[i32; AlpModel::INPUT_DIM]; AlpModel::HIDDEN1_DIM] = [
    [-8, -5, 14, -20, -50, -100, -100, -6, -4],
    [-12, -8, 18, -30, -60, -120, -120, -10, -6],
    [10, 4, -5, 45, -20, -100, -100, 2, 8],
    [-6, -4, 12, -15, -40, -80, -80, -5, -3],
    [14, 6, -8, 55, -25, -100, -100, 4, 12],
    [-15, -10, 22, -40, -80, -150, -150, -12, -8],
    [-4, -3, 8, -10, -30, -60, -60, -4, -2],
    [8, 2, -4, 35, -15, -90, -90, 1, 6],
    [-10, -7, 16, -25, -55, -110, -110, -8, -5],
    [-7, -5, 11, -18, -45, -85, -85, -6, -4],
    [12, 5, -6, 50, -20, -100, -100, 3, 10],
    [-9, -6, 15, -22, -50, -95, -95, -7, -5],
    [-5, -4, 9, -12, -35, -70, -70, -4, -3],
    [6, 1, -3, 30, -10, -80, -80, 1, 4],
    [-11, -8, 17, -28, -58, -115, -115, -9, -6],
    [-8, -5, 13, -20, -48, -90, -90, -6, -4],
];

const ALP_BIAS_1: [i32; AlpModel::HIDDEN1_DIM] = [
    -50, -60, -40, -45, -35, -70, -30, -38, -55, -48, -36, -52, -32, -30, -62, -46,
];

const ALP_WEIGHTS_2: [[i32; AlpModel::HIDDEN1_DIM]; AlpModel::HIDDEN2_DIM] = [
    [4, 6, -5, 3, -6, 8, 2, -4, 5, 3, -5, 4, 2, -3, 6, 4],
    [-3, -4, 8, -2, 10, -6, -1, 6, -4, -3, 9, -3, -2, 5, -5, -3],
    [5, 7, -6, 4, -7, 9, 3, -5, 6, 4, -6, 5, 3, -4, 7, 5],
    [-4, -5, 10, -3, 12, -7, -2, 7, -5, -4, 11, -4, -2, 6, -6, -4],
    [3, 5, -4, 2, -5, 7, 2, -3, 4, 3, -4, 3, 2, -2, 5, 3],
    [-2, -3, 6, -1, 8, -4, -1, 5, -3, -2, 7, -2, -1, 4, -4, -2],
    [6, 8, -7, 5, -8, 10, 4, -6, 7, 5, -7, 6, 4, -5, 8, 6],
    [-5, -6, 11, -4, 13, -8, -3, 8, -6, -5, 12, -5, -3, 7, -7, -5],
];

const ALP_BIAS_2: [i32; AlpModel::HIDDEN2_DIM] = [-20, -25, -22, -28, -18, -15, -24, -30];

const ALP_WEIGHTS_3: [i32; AlpModel::HIDDEN2_DIM] = [12, 14, 15, 16, 10, 8, 18, 20];

const ALP_BIAS_3: i32 = 120;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alp_pv_in_check_never_prunes() {
        let pv_features = AlpFeatures {
            eval_margin: -500,
            depth: 3,
            move_index: 20,
            is_null_move: false,
            is_capture: false,
            is_pv: true,
            in_check: false,
            history_score: -1000,
            momentum: -200,
        };
        assert_eq!(AlpModel::predict_safe_prune_prob(&pv_features), 0);
        assert!(!AlpModel::should_prune(&pv_features, 50));

        let check_features = AlpFeatures {
            eval_margin: -500,
            depth: 3,
            move_index: 20,
            is_null_move: false,
            is_capture: false,
            is_pv: false,
            in_check: true,
            history_score: -1000,
            momentum: -200,
        };
        assert_eq!(AlpModel::predict_safe_prune_prob(&check_features), 0);
        assert!(!AlpModel::should_prune(&check_features, 50));
    }

    #[test]
    fn test_alp_late_quiet_move_prunes() {
        let late_quiet_features = AlpFeatures {
            eval_margin: -600,
            depth: 2,
            move_index: 25,
            is_null_move: false,
            is_capture: false,
            is_pv: false,
            in_check: false,
            history_score: -800,
            momentum: -100,
        };
        let prob = AlpModel::predict_safe_prune_prob(&late_quiet_features);
        assert!(prob >= 70);
        assert!(AlpModel::should_prune(&late_quiet_features, 70));
    }

    #[test]
    fn test_alp_adversarial_loss() {
        let loss = AlpModel::adversarial_loss(0.8, 0.1, 1.0, 2.0);
        assert!((loss - (0.2 + 0.2)).abs() < 1e-5);
    }
}
