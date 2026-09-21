use crate::risk::Urgency;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UrgencyThresholds {
    pub min_gain: i32,
    pub musttry_gain: i32,
}

impl Default for UrgencyThresholds {
    fn default() -> Self {
        Self {
            min_gain: 30,
            musttry_gain: 50,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifyParams {
    pub tolerance_cp: i16,
}

impl Default for VerifyParams {
    fn default() -> Self {
        Self { tolerance_cp: 30 }
    }
}

pub fn urgency_for(window_plies: u32, gain_cp: i32) -> Urgency {
    urgency_for_with(window_plies, gain_cp, &UrgencyThresholds::default())
}

pub fn urgency_for_with(window_plies: u32, gain_cp: i32, th: &UrgencyThresholds) -> Urgency {
    if gain_cp < th.min_gain {
        return Urgency::Low;
    }
    match window_plies {
        0..=2 => {
            if gain_cp >= th.musttry_gain {
                Urgency::MustTry
            } else {
                Urgency::High
            }
        }
        3..=5 => Urgency::High,
        6..=10 => Urgency::Medium,
        _ => Urgency::Low,
    }
}

pub fn verification_depth(urgency: Urgency) -> u8 {
    match urgency {
        Urgency::Low | Urgency::Medium => 0,
        Urgency::High => 2,
        Urgency::MustTry => 3,
    }
}

pub fn attack_creates_value(
    king_displacement: bool,
    pawn_weakness: bool,
    material_gain_cp: i32,
    permanent_weakness: bool,
) -> bool {
    king_displacement || pawn_weakness || permanent_weakness || material_gain_cp >= 25
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urgency_ladder() {
        assert_eq!(urgency_for(1, 100), Urgency::MustTry);
        assert_eq!(urgency_for(4, 100), Urgency::High);
        assert_eq!(urgency_for(8, 100), Urgency::Medium);
        assert_eq!(urgency_for(20, 100), Urgency::Low);
        assert_eq!(urgency_for(1, 10), Urgency::Low);
    }

    #[test]
    fn urgency_thresholds_tunable() {
        let th = UrgencyThresholds {
            min_gain: 60,
            musttry_gain: 90,
        };
        assert_eq!(urgency_for_with(1, 50, &th), Urgency::Low);
        assert_eq!(urgency_for_with(1, 70, &th), Urgency::High);
        assert_eq!(urgency_for_with(1, 95, &th), Urgency::MustTry);
    }

    #[test]
    fn verification_bounded() {
        assert_eq!(verification_depth(Urgency::MustTry), 3);
        assert_eq!(verification_depth(Urgency::High), 2);

        assert_eq!(verification_depth(Urgency::Medium), 0);
        assert_eq!(verification_depth(Urgency::Low), 0);
    }

    #[test]
    fn conversion_values() {
        assert!(attack_creates_value(true, false, 0, false));
        assert!(attack_creates_value(false, false, 30, false));
        assert!(!attack_creates_value(false, false, 0, false));
    }
}
