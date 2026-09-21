use crate::risk::Urgency;

pub fn urgency_for(window_plies: u32, gain_cp: i32) -> Urgency {
    if gain_cp < 30 {
        return Urgency::Low;
    }
    match window_plies {
        0..=2 => {
            if gain_cp >= 50 {
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
