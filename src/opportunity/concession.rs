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
    fn defender_drop_amplifies_once_and_saturates() {
        assert_eq!(
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
}
