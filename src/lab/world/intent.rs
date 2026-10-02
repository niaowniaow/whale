use crate::world::position_state::PositionState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchIntent {
    Survival,
    Stabilization,
    #[default]
    Improvement,
    Pressure,
    Attack,
    Verification,
    Recovery,
    Conversion,
}

impl SearchIntent {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchIntent::Survival => "survival",
            SearchIntent::Stabilization => "stabilization",
            SearchIntent::Improvement => "improvement",
            SearchIntent::Pressure => "pressure",
            SearchIntent::Attack => "attack",
            SearchIntent::Verification => "verification",
            SearchIntent::Recovery => "recovery",
            SearchIntent::Conversion => "conversion",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntentEffects {
    pub rfp_adjust: i16,
    pub lmp_adjust: i32,
    pub lmr_adjust: i8,
    pub time_boost_x100: u32,
}

pub fn intent_for(
    state: PositionState,
    musttry_fired: bool,
    simplify: bool,
    is_mate: bool,
) -> SearchIntent {
    if musttry_fired && (!matches!(state, PositionState::Crush | PositionState::Convert) || is_mate)
    {
        return SearchIntent::Verification;
    }
    let base = match state {
        PositionState::Defend => SearchIntent::Survival,
        PositionState::Stabilize => SearchIntent::Stabilization,
        PositionState::Improve => SearchIntent::Improvement,
        PositionState::Press => SearchIntent::Pressure,
        PositionState::Attack => SearchIntent::Attack,
        PositionState::Reset => SearchIntent::Recovery,
        PositionState::Crush | PositionState::Convert => SearchIntent::Conversion,
    };
    if simplify
        && matches!(
            base,
            SearchIntent::Stabilization
                | SearchIntent::Improvement
                | SearchIntent::Pressure
                | SearchIntent::Attack
        )
    {
        return SearchIntent::Conversion;
    }
    base
}

pub fn effects(intent: SearchIntent) -> IntentEffects {
    match intent {
        SearchIntent::Survival => IntentEffects {
            rfp_adjust: 0,
            lmp_adjust: -1,
            lmr_adjust: 0,
            time_boost_x100: 100,
        },
        SearchIntent::Stabilization => IntentEffects {
            rfp_adjust: 0,
            lmp_adjust: 0,
            lmr_adjust: 0,
            time_boost_x100: 100,
        },
        SearchIntent::Improvement => IntentEffects {
            rfp_adjust: 0,
            lmp_adjust: 0,
            lmr_adjust: 0,
            time_boost_x100: 100,
        },
        SearchIntent::Pressure => IntentEffects {
            rfp_adjust: 0,
            lmp_adjust: 0,
            lmr_adjust: 0,
            time_boost_x100: 100,
        },
        SearchIntent::Attack => IntentEffects {
            rfp_adjust: 0,
            lmp_adjust: 2,
            lmr_adjust: -1,
            time_boost_x100: 100,
        },
        SearchIntent::Verification => IntentEffects {
            rfp_adjust: 0,
            lmp_adjust: 2,
            lmr_adjust: -1,
            time_boost_x100: 125,
        },
        SearchIntent::Recovery => IntentEffects {
            rfp_adjust: 40,
            lmp_adjust: 0,
            lmr_adjust: 1,
            time_boost_x100: 100,
        },
        SearchIntent::Conversion => IntentEffects {
            rfp_adjust: -30,
            lmp_adjust: -2,
            lmr_adjust: 0,
            time_boost_x100: 100,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn musttry_drives_verification() {
        assert_eq!(
            intent_for(PositionState::Attack, true, false, false),
            SearchIntent::Verification
        );
        assert_eq!(
            intent_for(PositionState::Improve, true, false, false),
            SearchIntent::Verification
        );
        assert_eq!(
            intent_for(PositionState::Crush, true, false, true),
            SearchIntent::Verification
        );
        assert_eq!(
            intent_for(PositionState::Crush, true, false, false),
            SearchIntent::Conversion
        );
        assert_eq!(
            intent_for(PositionState::Convert, true, true, false),
            SearchIntent::Conversion
        );
    }

    #[test]
    fn states_map_to_intents() {
        assert_eq!(
            intent_for(PositionState::Defend, false, false, false),
            SearchIntent::Survival
        );
        assert_eq!(
            intent_for(PositionState::Stabilize, false, false, false),
            SearchIntent::Stabilization
        );
        assert_eq!(
            intent_for(PositionState::Improve, false, false, false),
            SearchIntent::Improvement
        );
        assert_eq!(
            intent_for(PositionState::Press, false, false, false),
            SearchIntent::Pressure
        );
        assert_eq!(
            intent_for(PositionState::Attack, false, false, false),
            SearchIntent::Attack
        );
        assert_eq!(
            intent_for(PositionState::Reset, false, false, false),
            SearchIntent::Recovery
        );
        assert_eq!(
            intent_for(PositionState::Crush, false, false, false),
            SearchIntent::Conversion
        );
        assert_eq!(
            intent_for(PositionState::Convert, false, false, false),
            SearchIntent::Conversion
        );
    }

    #[test]
    fn simplify_promotes_quiet_intents_to_conversion() {
        assert_eq!(
            intent_for(PositionState::Press, false, true, false),
            SearchIntent::Conversion
        );
        assert_eq!(
            intent_for(PositionState::Defend, false, true, false),
            SearchIntent::Survival
        );
        assert_eq!(
            intent_for(PositionState::Reset, false, true, false),
            SearchIntent::Recovery
        );
    }

    #[test]
    fn effects_preserve_legacy_magnitudes() {
        assert_eq!(effects(SearchIntent::Recovery).rfp_adjust, 40);
        assert_eq!(effects(SearchIntent::Conversion).rfp_adjust, -30);
        assert_eq!(effects(SearchIntent::Verification).lmp_adjust, 2);
        assert_eq!(effects(SearchIntent::Attack).lmp_adjust, 2);
        assert_eq!(effects(SearchIntent::Survival).lmp_adjust, -1);
        assert_eq!(effects(SearchIntent::Verification).time_boost_x100, 125);
    }

    #[test]
    fn intent_names_stable_for_diagnostics() {
        assert_eq!(SearchIntent::Verification.as_str(), "verification");
        assert_eq!(SearchIntent::Recovery.as_str(), "recovery");
    }
}
