#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WeaknessKind {
    Temporary,
    Latent,
    Structural,
    Permanent,
}

pub fn classify_weakness(persistence_plies: u32) -> WeaknessKind {
    if persistence_plies >= 12 {
        WeaknessKind::Permanent
    } else if persistence_plies >= 6 {
        WeaknessKind::Structural
    } else if persistence_plies >= 2 {
        WeaknessKind::Latent
    } else {
        WeaknessKind::Temporary
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Concession {
    pub swing_cp: i16,
    pub kind: WeaknessKind,
}

pub const IMMEDIATE_CONFIRM_SWING: i16 = 300;

pub fn confirmation_threshold(
    base_cp: i16,
    volatility: crate::world::position_state::VolatilityLevel,
) -> i16 {
    use crate::world::position_state::VolatilityLevel;
    let bonus = match volatility {
        VolatilityLevel::Low => 0,
        VolatilityLevel::Medium => 15,
        VolatilityLevel::High => 30,
        VolatilityLevel::Extreme => 60,
    };
    base_cp.saturating_add(bonus)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConcessionTracker {
    pub pending_swing: i16,
    pub confirmed: Option<Concession>,
}
impl ConcessionTracker {
    pub fn observe(&mut self, prev: i16, curr: i16, threshold: i16) -> Option<Concession> {        let swing = curr.saturating_sub(prev);
        if swing >= IMMEDIATE_CONFIRM_SWING {
            let confirmed = Concession {
                swing_cp: swing,
                kind: WeaknessKind::Latent,
            };
            self.confirmed = Some(confirmed);
            self.pending_swing = 0;
            return Some(confirmed);
        }
        if swing >= threshold {
            if self.pending_swing > 0 && swing >= self.pending_swing / 2 {
                let confirmed = Concession {
                    swing_cp: swing,
                    kind: WeaknessKind::Latent,
                };
                self.confirmed = Some(confirmed);
                self.pending_swing = 0;
                return Some(confirmed);
            }
            self.pending_swing = swing;
            return self.confirmed;
        }
        if swing <= 0 {
            self.pending_swing = 0;
        }
        self.confirmed
    }
}

pub fn amplify_on_defender_drop(
    kind: WeaknessKind,
    defenders_before: u32,
    defenders_after: u32,
) -> WeaknessKind {
    if defenders_after >= defenders_before {
        return kind;
    }
    match kind {
        WeaknessKind::Temporary => WeaknessKind::Latent,
        WeaknessKind::Latent => WeaknessKind::Structural,
        WeaknessKind::Structural | WeaknessKind::Permanent => WeaknessKind::Permanent,
    }
}

pub fn detect_concession(prev: i16, curr: i16, persistence_plies: u32) -> Option<Concession> {
    let swing = curr.saturating_sub(prev);
    if swing < 12 {
        return None;
    }

    let kind = if swing >= 80 && persistence_plies == 0 {
        WeaknessKind::Latent
    } else {
        classify_weakness(persistence_plies)
    };
    Some(Concession {
        swing_cp: swing,
        kind,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ScoreTrend {
    pub recent: [i16; 4],
    pub depths: [u8; 4],
    pub len: u8,
}

impl ScoreTrend {
    pub fn push(&mut self, white_pov: i16, depth: u8) {
        let i = (self.len % 4) as usize;
        self.recent[i] = white_pov;
        self.depths[i] = depth;
        self.len = self.len.saturating_add(1);
    }

    pub fn count(&self) -> usize {
        self.len.min(4) as usize
    }

    pub fn ordered(&self, k: usize) -> i16 {
        let start = if self.len > 4 {
            (self.len % 4) as usize
        } else {
            0
        };
        self.recent[(start + k) % 4]
    }

    pub fn smoothed(&self) -> Option<i16> {
        let n = self.count();
        if n == 0 {
            return None;
        }
        let mut sum = 0i32;
        for k in 0..n {
            sum += self.ordered(k) as i32;
        }
        Some((sum / n as i32) as i16)
    }

    pub fn noise_floor(&self) -> i16 {
        let n = self.count();
        if n < 2 {
            return 0;
        }
        let mut sum = 0i32;
        for k in 1..n {
            sum += (self.ordered(k).saturating_sub(self.ordered(k - 1))).abs() as i32;
        }
        (sum / (n as i32 - 1)) as i16
    }

    pub fn adaptive_threshold(&self, base: i16) -> i16 {
        let floor = self.noise_floor() as i32;
        base.max(((floor * 3 / 2).min(80)) as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_order() {
        assert!(WeaknessKind::Permanent > WeaknessKind::Structural);
        assert!(WeaknessKind::Structural > WeaknessKind::Latent);
        assert!(WeaknessKind::Latent > WeaknessKind::Temporary);
    }

    #[test]
    fn noise_is_not_concession() {
        assert_eq!(detect_concession(20, 25, 0), None);
        assert_eq!(detect_concession(20, 10, 0), None);
    }

    #[test]
    fn small_swing_detected() {
        let c = detect_concession(20, 45, 3).expect("should detect");
        assert_eq!(c.swing_cp, 25);
        assert_eq!(c.kind, WeaknessKind::Latent);
    }

    #[test]
    fn persistent_weakness_upgrades() {
        let c = detect_concession(0, 30, 14).expect("should detect");
        assert_eq!(c.kind, WeaknessKind::Permanent);
    }

    #[test]
    fn defender_drop_amplifies_once_and_saturates() {        assert_eq!(
            amplify_on_defender_drop(WeaknessKind::Temporary, 3, 2),
            WeaknessKind::Latent
        );
        assert_eq!(
            amplify_on_defender_drop(WeaknessKind::Permanent, 3, 0),
            WeaknessKind::Permanent
        );
        assert_eq!(
            amplify_on_defender_drop(WeaknessKind::Latent, 2, 2),
            WeaknessKind::Latent
        );
        assert_eq!(
            amplify_on_defender_drop(WeaknessKind::Latent, 1, 3),
            WeaknessKind::Latent
        );
    }

    #[test]
    fn single_blip_does_not_confirm() {
        let mut t = ConcessionTracker::default();
        assert_eq!(t.observe(20, 45, 12), None);
        assert_eq!(t.observe(20, 44, 12).map(|c| c.swing_cp), Some(24));
    }

    #[test]
    fn faded_pending_is_dropped() {
        let mut t = ConcessionTracker::default();
        assert_eq!(t.observe(20, 45, 12), None);
        assert_eq!(t.observe(45, 40, 12), None);
        assert_eq!(t.pending_swing, 0);
    }

    #[test]
    fn mate_scale_swing_confirms_immediately() {
        let mut t = ConcessionTracker::default();
        let c = t.observe(10, 30999, 60).expect("mate jump confirms");
        assert_eq!(c.swing_cp, 30989);
    }

    #[test]
    fn volatile_positions_need_bigger_swings() {
        use crate::world::position_state::VolatilityLevel;
        assert_eq!(confirmation_threshold(12, VolatilityLevel::Low), 12);
        assert_eq!(confirmation_threshold(12, VolatilityLevel::Medium), 27);
        assert_eq!(confirmation_threshold(12, VolatilityLevel::High), 42);
        assert_eq!(confirmation_threshold(12, VolatilityLevel::Extreme), 72);
        let mut t = ConcessionTracker::default();
        assert_eq!(t.observe(0, 40, 72), None);
        assert_eq!(t.observe(0, 41, 72), None);
    }

    #[test]
    fn trend_smooths_and_measures_noise() {
        let mut t = ScoreTrend::default();
        assert_eq!(t.smoothed(), None);
        assert_eq!(t.noise_floor(), 0);
        t.push(10, 2);
        assert_eq!(t.smoothed(), Some(10));
        t.push(20, 2);
        t.push(15, 3);
        assert_eq!(t.smoothed(), Some(15));
        assert_eq!(t.noise_floor(), 7);
        assert_eq!(t.adaptive_threshold(12), 12);
        t.push(60, 3);
        assert!(t.noise_floor() > 10);
        assert!(t.adaptive_threshold(12) > 12);
    }

    #[test]
    fn trend_ring_overwrites_oldest() {
        let mut t = ScoreTrend::default();
        for v in [10i16, 20, 30, 40, 50] {
            t.push(v, 2);
        }
        assert_eq!(t.count(), 4);
        assert_eq!(t.smoothed(), Some(35));
    }
}
