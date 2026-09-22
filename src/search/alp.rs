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

pub struct AlpPruner;

impl AlpPruner {
    pub fn score(features: &AlpFeatures) -> i32 {
        if features.is_pv || features.in_check {
            return 0;
        }
        let mut score = 0i32;
        score += ((features.move_index as i32) - 8).clamp(0, 24) * 6;
        if features.is_capture {
            score -= 60;
        }
        if features.is_null_move {
            score += 55;
        }
        score += (features.eval_margin / 64).clamp(-10, 30);
        score -= (features.history_score / 512).clamp(-10, 30);
        score += (features.momentum / 64).clamp(-10, 10);
        score -= (features.depth * 3).clamp(0, 24);
        score.clamp(0, 100)
    }

    pub fn predict_safe_prune_prob(features: &AlpFeatures) -> u8 {
        Self::score(features) as u8
    }

    pub fn should_prune(features: &AlpFeatures, threshold_percent: u8) -> bool {
        Self::predict_safe_prune_prob(features) >= threshold_percent
    }
}

pub use AlpPruner as AlpModel;

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
    fn test_alp_null_context_prunes() {
        let null_features = AlpFeatures {
            eval_margin: 0,
            depth: 3,
            move_index: 0,
            is_null_move: true,
            is_capture: false,
            is_pv: false,
            in_check: false,
            history_score: 0,
            momentum: 0,
        };
        assert!(AlpModel::should_prune(&null_features, 30));
    }

    #[test]
    fn test_alp_early_good_moves_survive() {
        let early = AlpFeatures {
            eval_margin: 0,
            depth: 3,
            move_index: 2,
            is_null_move: false,
            is_capture: false,
            is_pv: false,
            in_check: false,
            history_score: 400,
            momentum: 0,
        };
        assert!(!AlpModel::should_prune(&early, 30));
    }

    #[test]
    fn test_alp_captures_protected() {
        let cap = AlpFeatures {
            eval_margin: 200,
            depth: 2,
            move_index: 20,
            is_null_move: false,
            is_capture: true,
            is_pv: false,
            in_check: false,
            history_score: 0,
            momentum: 0,
        };
        assert!(!AlpModel::should_prune(&cap, 70));
    }

    #[test]
    fn test_alp_score_monotone_in_move_index() {
        let base = AlpFeatures {
            eval_margin: 0,
            depth: 2,
            move_index: 8,
            is_null_move: false,
            is_capture: false,
            is_pv: false,
            in_check: false,
            history_score: 0,
            momentum: 0,
        };
        let mut prev = AlpModel::score(&base);
        for idx in 9..30 {
            let mut f = base;
            f.move_index = idx;
            let cur = AlpModel::score(&f);
            assert!(cur >= prev);
            prev = cur;
        }
    }
}
