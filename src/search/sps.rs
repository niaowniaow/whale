use crate::search::lmr::LmrTable;
use crate::search::search_state::SearchParameters;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPersona {
    Standard,
    Tactical,
    Solid,
    Aggressive,
}

#[inline(always)]
pub fn persona_for_thread(thread_id: usize) -> SearchPersona {
    match thread_id % 4 {
        0 => SearchPersona::Standard,
        1 => SearchPersona::Tactical,
        2 => SearchPersona::Solid,
        3 => SearchPersona::Aggressive,
        _ => SearchPersona::Standard,
    }
}

/// NOTE (FIX PERSONA-TT-POLLUTION): no in-tree caller uses personas anymore.
/// `SearchState::clone_for_worker` intentionally copies params/optimism/lmr
/// verbatim so all threads sharing the Arc TT search identical parameters
/// (Lazy SMP diversity comes from depth stagger only, cf. Reckless/Stockfish).
/// Kept for API compat; Aggressive branch is zero-sum (+30/-30).
pub fn apply_persona(
    persona: SearchPersona,
    params: &mut SearchParameters,
    optimism: &mut [i32; 2],
    lmr_table: &mut LmrTable,
) {
    match persona {
        SearchPersona::Standard => {}
        SearchPersona::Tactical => {
            params.futility_margin_mult = 160;
            params.probcut_margin = 140;
            params.rfp_margin_mult = 130;
            params.lmr_base = 0.50;
            *lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);
        }
        SearchPersona::Solid => {
            params.futility_margin_mult = 95;
            params.rfp_margin_mult = 90;
            params.nmp_depth_div = 2;
            params.lmr_base = 0.75;
            *lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);
        }
        SearchPersona::Aggressive => {
            // FIX ASPIRATION zero-sum: +30/-30 so White/Black optimism sums to 0.
            optimism[0] = optimism[0].saturating_add(30);
            optimism[1] = optimism[1].saturating_sub(30);
            params.futility_margin_mult = 140;
            params.lmr_base = 0.60;
            *lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persona_assignment() {
        assert_eq!(persona_for_thread(0), SearchPersona::Standard);
        assert_eq!(persona_for_thread(1), SearchPersona::Tactical);
        assert_eq!(persona_for_thread(2), SearchPersona::Solid);
        assert_eq!(persona_for_thread(3), SearchPersona::Aggressive);
        assert_eq!(persona_for_thread(4), SearchPersona::Standard);
        assert_eq!(persona_for_thread(5), SearchPersona::Tactical);
    }

    #[test]
    fn test_apply_tactical_persona() {
        let mut params = SearchParameters::default();
        let mut optimism = [0; 2];
        let mut lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);

        apply_persona(
            SearchPersona::Tactical,
            &mut params,
            &mut optimism,
            &mut lmr_table,
        );

        assert_eq!(params.futility_margin_mult, 160);
        assert_eq!(params.probcut_margin, 140);
        assert_eq!(params.rfp_margin_mult, 130);
        assert_eq!(params.lmr_base, 0.50);
        assert_eq!(optimism, [0, 0]);
    }

    #[test]
    fn test_apply_solid_persona() {
        let mut params = SearchParameters::default();
        let mut optimism = [0; 2];
        let mut lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);

        apply_persona(
            SearchPersona::Solid,
            &mut params,
            &mut optimism,
            &mut lmr_table,
        );

        assert_eq!(params.futility_margin_mult, 95);
        assert_eq!(params.rfp_margin_mult, 90);
        assert_eq!(params.nmp_depth_div, 2);
        assert_eq!(params.lmr_base, 0.75);
    }

    #[test]
    fn test_apply_aggressive_persona() {
        // Updated expectation: old test encoded the buggy +30/+30 behaviour;
        // zero-sum fix gives +30/-30.
        let mut params = SearchParameters::default();
        let mut optimism = [10, -10];
        let mut lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);

        apply_persona(
            SearchPersona::Aggressive,
            &mut params,
            &mut optimism,
            &mut lmr_table,
        );

        assert_eq!(optimism[0], 40);
        assert_eq!(optimism[1], -40);
        assert_eq!(params.futility_margin_mult, 140);
        assert_eq!(params.lmr_base, 0.60);
    }

    #[test]
    fn test_apply_standard_persona_unchanged() {
        let mut params = SearchParameters::default();
        let mut optimism = [0; 2];
        let mut lmr_table = LmrTable::new(params.lmr_base, params.lmr_div);

        let orig_futility = params.futility_margin_mult;
        apply_persona(
            SearchPersona::Standard,
            &mut params,
            &mut optimism,
            &mut lmr_table,
        );

        assert_eq!(params.futility_margin_mult, orig_futility);
        assert_eq!(optimism, [0, 0]);
    }
}
