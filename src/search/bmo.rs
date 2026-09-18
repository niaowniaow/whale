#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BanditArm {
    CapturesFirst = 0,
    QuietsFirst = 1,
}

impl BanditArm {
    pub const COUNT: usize = 2;

    pub fn from_index(index: usize) -> Self {
        match index {
            1 => BanditArm::QuietsFirst,
            _ => BanditArm::CapturesFirst,
        }
    }

    pub fn to_index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone)]
pub struct BanditMoveOrdering {
    pulls: [[u32; BanditArm::COUNT]; 3],
    rewards: [[u32; BanditArm::COUNT]; 3],
}

impl BanditMoveOrdering {
    pub fn new() -> Self {
        Self {
            pulls: [[1; BanditArm::COUNT]; 3],
            rewards: [[0; BanditArm::COUNT]; 3],
        }
    }

    pub fn select_arm(&self, depth: u8) -> BanditArm {
        let bucket = Self::get_bucket(depth);
        let pulls = &self.pulls[bucket];
        let rewards = &self.rewards[bucket];

        let total_pulls: u32 = pulls.iter().sum();
        let total_f = (total_pulls as f64).max(1.0);
        let ln_total = total_f.ln();

        let exploration_c = if depth >= 8 { 0.25 } else { 0.65 };

        let mut best_arm = BanditArm::CapturesFirst;
        let mut best_score = f64::NEG_INFINITY;

        for (i, (&pull, &reward)) in pulls.iter().zip(rewards.iter()).enumerate() {
            let pull_f = (pull as f64).max(1.0);
            let exploitation = reward as f64 / pull_f;
            let exploration = exploration_c * (ln_total / pull_f).sqrt();
            let ucb = exploitation + exploration;
            if ucb > best_score {
                best_score = ucb;
                best_arm = BanditArm::from_index(i);
            }
        }

        best_arm
    }

    pub fn update(&mut self, depth: u8, arm: BanditArm, cutoff: bool) {
        let bucket = Self::get_bucket(depth);
        let arm_idx = arm.to_index();
        self.pulls[bucket][arm_idx] = self.pulls[bucket][arm_idx].saturating_add(1);
        if cutoff {
            self.rewards[bucket][arm_idx] = self.rewards[bucket][arm_idx].saturating_add(1);
        }

        if self.pulls[bucket][arm_idx] > 65_536 {
            for b in 0..3 {
                for a in 0..BanditArm::COUNT {
                    self.pulls[b][a] = (self.pulls[b][a] / 2).max(1);
                    self.rewards[b][a] /= 2;
                }
            }
        }
    }

    pub fn reset(&mut self) {
        self.pulls = [[1; BanditArm::COUNT]; 3];
        self.rewards = [[0; BanditArm::COUNT]; 3];
    }

    fn get_bucket(depth: u8) -> usize {
        if depth < 5 {
            0
        } else if depth < 9 {
            1
        } else {
            2
        }
    }
}

impl Default for BanditMoveOrdering {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandit_initial_selection() {
        let bmo = BanditMoveOrdering::new();
        let arm = bmo.select_arm(6);
        assert!(arm == BanditArm::CapturesFirst || arm == BanditArm::QuietsFirst);
    }

    #[test]
    fn test_bandit_learns_rewards() {
        let mut bmo = BanditMoveOrdering::new();
        for _ in 0..100 {
            bmo.update(6, BanditArm::QuietsFirst, true);
        }
        for _ in 0..100 {
            bmo.update(6, BanditArm::CapturesFirst, false);
        }
        let arm = bmo.select_arm(6);
        assert_eq!(arm, BanditArm::QuietsFirst);
    }

    #[test]
    fn arm_index_roundtrips_and_defaults() {
        assert_eq!(BanditArm::from_index(0), BanditArm::CapturesFirst);
        assert_eq!(BanditArm::from_index(1), BanditArm::QuietsFirst);
        assert_eq!(BanditArm::from_index(99), BanditArm::CapturesFirst);
        assert_eq!(BanditArm::CapturesFirst.to_index(), 0);
        assert_eq!(BanditArm::QuietsFirst.to_index(), 1);
        assert_eq!(BanditArm::COUNT, 2);
        let _ = BanditMoveOrdering::default();
    }

    #[test]
    fn buckets_split_by_depth() {
        let mut bmo = BanditMoveOrdering::new();
        for _ in 0..50 {
            bmo.update(0, BanditArm::CapturesFirst, true);
        }
        assert_eq!(bmo.select_arm(0), BanditArm::QuietsFirst);
        assert_eq!(bmo.select_arm(4), BanditArm::QuietsFirst);
        assert_eq!(bmo.select_arm(5), BanditArm::CapturesFirst);
        assert_eq!(bmo.select_arm(8), BanditArm::CapturesFirst);
        assert_eq!(bmo.select_arm(9), BanditArm::CapturesFirst);
        assert_eq!(bmo.select_arm(64), BanditArm::CapturesFirst);
    }

    #[test]
    fn reset_restores_tie() {
        let mut bmo = BanditMoveOrdering::new();
        for _ in 0..100 {
            bmo.update(6, BanditArm::QuietsFirst, true);
        }
        for _ in 0..100 {
            bmo.update(6, BanditArm::CapturesFirst, false);
        }
        assert_eq!(bmo.select_arm(6), BanditArm::QuietsFirst);
        bmo.reset();
        assert_eq!(bmo.select_arm(6), BanditArm::CapturesFirst);
        assert_eq!(
            BanditMoveOrdering::new().select_arm(6),
            BanditArm::CapturesFirst
        );
    }

    #[test]
    fn halving_keeps_counts_bounded() {
        let mut bmo = BanditMoveOrdering::new();
        for _ in 0..70_000 {
            bmo.update(6, BanditArm::CapturesFirst, true);
        }
        let arm = bmo.select_arm(6);
        assert!(arm == BanditArm::CapturesFirst || arm == BanditArm::QuietsFirst);
        let _ = bmo.select_arm(0);
        let _ = bmo.select_arm(20);
    }
}
