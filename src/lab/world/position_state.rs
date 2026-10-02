#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionState {
    Defend,
    Stabilize,
    #[default]
    Improve,
    Press,
    Attack,
    Reset,
    Crush,
    Convert,
}

impl PositionState {
    pub fn as_str(self) -> &'static str {
        match self {
            PositionState::Defend => "defend",
            PositionState::Stabilize => "stabilize",
            PositionState::Improve => "improve",
            PositionState::Press => "press",
            PositionState::Attack => "attack",
            PositionState::Reset => "reset",
            PositionState::Crush => "crush",
            PositionState::Convert => "convert",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StateThresholds {
    pub defend_score: i16,
    pub defend_cpi: i32,
    pub convert_score: i16,
    pub convert_cpi: i32,
    pub crush_score: i16,
    pub attack_score: i16,
    pub attack_cpi: i32,
    pub reset_drop: i16,
    pub concession_min_cp: i16,
}

impl Default for StateThresholds {
    fn default() -> Self {
        Self {
            defend_score: -150,
            defend_cpi: 120,
            convert_score: 250,
            convert_cpi: 30,
            crush_score: 600,
            attack_score: 80,
            attack_cpi: 60,
            reset_drop: 50,
            concession_min_cp: 12,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum VolatilityLevel {
    #[default]
    Low,
    Medium,
    High,
    Extreme,
}

pub struct StateInput {
    pub score: i16,
    pub opp_cpi: i32,
    pub own_cpi: i32,
    pub momentum: i16,
    pub in_check: bool,
    pub volatility: VolatilityLevel,
    pub score_drop: i16,
    pub attack_failed: bool,
}

pub fn classify(input: &StateInput, th: &StateThresholds) -> PositionState {
    classify_with_hysteresis(input, th, None)
}

pub fn classify_with_hysteresis(
    input: &StateInput,
    th: &StateThresholds,
    prev_state: Option<PositionState>,
) -> PositionState {
    if input.in_check || input.score <= th.defend_score || input.opp_cpi >= th.defend_cpi {
        return PositionState::Defend;
    }
    if input.score >= th.crush_score {
        return PositionState::Crush;
    }
    if input.attack_failed {
        return PositionState::Reset;
    }
    if let Some(prev) = prev_state {
        match prev {
            PositionState::Defend
                if input.score <= th.defend_score + 30 || input.opp_cpi >= th.defend_cpi - 20 =>
            {
                return PositionState::Defend;
            }
            PositionState::Attack => {
                if input.score >= th.attack_score - 25
                    && input.own_cpi >= th.attack_cpi - 15
                    && input.score_drop > -th.reset_drop
                {
                    return PositionState::Attack;
                }
                return PositionState::Reset;
            }
            PositionState::Reset => {
                if input.score >= th.convert_score && input.opp_cpi <= th.convert_cpi {
                    return PositionState::Convert;
                }
                return PositionState::Press;
            }
            PositionState::Convert
                if input.score >= th.convert_score - 30 && input.opp_cpi <= th.convert_cpi + 15 =>
            {
                return PositionState::Convert;
            }
            _ => {}
        }
    }

    if input.score >= th.convert_score && input.opp_cpi <= th.convert_cpi {
        return PositionState::Convert;
    }
    if input.score >= th.attack_score && input.own_cpi >= th.attack_cpi {
        return PositionState::Attack;
    }
    if input.momentum > 10 && input.score > -50 {
        return PositionState::Press;
    }
    if input.score.abs() <= 30 && input.opp_cpi < 60 {
        return PositionState::Improve;
    }
    PositionState::Stabilize
}

pub fn recovery_hint(state: PositionState) -> &'static str {
    match state {
        PositionState::Defend => "absorb,close-center,restore-coordination",
        PositionState::Stabilize => "repair,reduce-counterplay",
        PositionState::Attack => "reset-to-press,keep-concessions",
        PositionState::Reset => "recover,restore-coordination,reset-to-press",
        PositionState::Crush | PositionState::Convert => "simplify,remove-counterplay",
        PositionState::Press | PositionState::Improve => "keep-pressing",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(score: i16, opp_cpi: i32, own_cpi: i32, momentum: i16, in_check: bool) -> StateInput {
        StateInput {
            score,
            opp_cpi,
            own_cpi,
            momentum,
            in_check,
            volatility: VolatilityLevel::Low,
            score_drop: 0,
            attack_failed: false,
        }
    }

    #[test]
    fn hysteresis_prevents_thrashing() {
        let th = StateThresholds::default();
        let inp = input(70, 20, 50, 0, false);
        assert_eq!(classify(&inp, &th), PositionState::Stabilize);
        assert_eq!(
            classify_with_hysteresis(&inp, &th, Some(PositionState::Attack)),
            PositionState::Attack
        );
    }

    #[test]
    fn each_state_reachable() {
        let th = StateThresholds::default();
        assert_eq!(
            classify(&input(-400, 0, 0, 0, false), &th),
            PositionState::Defend
        );
        assert_eq!(
            classify(&input(0, 0, 0, 0, true), &th),
            PositionState::Defend
        );
        assert_eq!(
            classify(&input(700, 10, 10, 0, false), &th),
            PositionState::Crush
        );
        assert_eq!(
            classify(&input(300, 10, 10, 0, false), &th),
            PositionState::Convert
        );
        assert_eq!(
            classify(&input(100, 20, 80, 0, false), &th),
            PositionState::Attack
        );
        assert_eq!(
            classify(&input(20, 20, 20, 25, false), &th),
            PositionState::Press
        );
        assert_eq!(
            classify(&input(0, 10, 10, 0, false), &th),
            PositionState::Improve
        );
        assert_eq!(
            classify(&input(-100, 80, 10, -20, false), &th),
            PositionState::Stabilize
        );
        let mut failed = input(100, 20, 80, 0, false);
        failed.attack_failed = true;
        assert_eq!(classify(&failed, &th), PositionState::Reset);
    }

    #[test]
    fn attack_collapse_transitions_to_reset_then_press() {
        let th = StateThresholds::default();
        let mut collapsed = input(70, 20, 50, 0, false);
        collapsed.score_drop = -80;
        assert_eq!(
            classify_with_hysteresis(&collapsed, &th, Some(PositionState::Attack)),
            PositionState::Reset
        );
        let recovering = input(20, 20, 20, 5, false);
        assert_eq!(
            classify_with_hysteresis(&recovering, &th, Some(PositionState::Reset)),
            PositionState::Press
        );
        let converted = input(300, 10, 10, 0, false);
        assert_eq!(
            classify_with_hysteresis(&converted, &th, Some(PositionState::Reset)),
            PositionState::Convert
        );
    }

    #[test]
    fn state_names_stable_for_diagnostics() {
        assert_eq!(PositionState::Press.as_str(), "press");
        assert_eq!(PositionState::Convert.as_str(), "convert");
        assert_eq!(PositionState::Reset.as_str(), "reset");
    }
}
