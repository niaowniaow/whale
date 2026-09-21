#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Urgency {
    Low,
    Medium,
    High,
    MustTry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Normal,
    Elevated,
    MustTry,
    Forcing,
}

#[derive(Debug, Clone, Copy)]
pub struct RiskEnvelope {
    pub normal_max: i32,
    pub elevated_max: i32,
    pub must_try_max: i32,

    pub min_gain_cp: i32,
}

impl Default for RiskEnvelope {
    fn default() -> Self {
        Self {
            normal_max: 30,
            elevated_max: 70,
            must_try_max: 120,
            min_gain_cp: 30,
        }
    }
}

impl RiskEnvelope {
    pub fn level_for(&self, risk: i32) -> RiskLevel {
        if risk <= self.normal_max {
            RiskLevel::Normal
        } else if risk <= self.elevated_max {
            RiskLevel::Elevated
        } else if risk <= self.must_try_max {
            RiskLevel::MustTry
        } else {
            RiskLevel::Forcing
        }
    }
}

pub fn risk_score(cpi_created: i32, king_danger: i32, irreversible: bool) -> i32 {
    let mut r = cpi_created.clamp(0, 200) + king_danger.clamp(0, 100);
    if irreversible {
        r += 25;
    }
    r.clamp(0, 400)
}

#[derive(Debug, Clone, Copy)]
pub struct MustTryInput {
    pub gain_cp: i32,
    pub urgency: Urgency,
    pub risk: i32,
    pub opp_cpi_after: i32,
}

pub fn must_try_gate(input: &MustTryInput, env: &RiskEnvelope) -> bool {
    if input.gain_cp < env.min_gain_cp {
        return false;
    }
    if input.urgency < Urgency::High {
        return false;
    }
    if input.risk > env.must_try_max {
        return false;
    }

    if input.opp_cpi_after > 150 {
        return false;
    }
    true
}

pub fn controlled_aggression_ok(gain_cp: i32, damage_cp: i32, risk: i32) -> bool {
    gain_cp > damage_cp + risk / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_levels_monotone() {
        let env = RiskEnvelope::default();
        assert_eq!(env.level_for(0), RiskLevel::Normal);
        assert_eq!(env.level_for(50), RiskLevel::Elevated);
        assert_eq!(env.level_for(100), RiskLevel::MustTry);
        assert_eq!(env.level_for(500), RiskLevel::Forcing);
    }

    #[test]
    fn gate_requires_gain_urgency_risk() {
        let env = RiskEnvelope::default();
        let base = MustTryInput {
            gain_cp: 60,
            urgency: Urgency::High,
            risk: 80,
            opp_cpi_after: 40,
        };
        assert!(must_try_gate(&base, &env));

        assert!(!must_try_gate(
            &MustTryInput {
                gain_cp: 10,
                ..base
            },
            &env
        ));

        assert!(!must_try_gate(
            &MustTryInput {
                urgency: Urgency::Low,
                ..base
            },
            &env
        ));

        assert!(!must_try_gate(&MustTryInput { risk: 500, ..base }, &env));

        assert!(!must_try_gate(
            &MustTryInput {
                opp_cpi_after: 250,
                ..base
            },
            &env
        ));
    }

    #[test]
    fn aggression_math() {
        assert!(controlled_aggression_ok(100, 20, 40));
        assert!(!controlled_aggression_ok(30, 40, 40));
    }
}
