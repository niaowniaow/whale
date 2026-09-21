use crate::search::search_state::SearchState;
use crate::world::intent::SearchIntent;
use crate::world::position_state::PositionState;

#[derive(Debug, Clone)]
pub struct SearchMetrics {
    pub score: i16,
    pub state: PositionState,
    pub intent: SearchIntent,
    pub opp_cpi: i32,
    pub own_cpi: i32,
    pub opp_freedom: u32,
    pub opp_breaks: u32,
    pub risk: i32,
    pub pressure: i32,
    pub sustained_plies: u32,
    pub concession_swing: Option<i16>,
    pub musttry_fired: bool,
    pub verified: Option<i16>,
    pub simplify: bool,
    pub reset: bool,
    pub multipv_count: usize,
}

pub fn collect(state: &SearchState) -> SearchMetrics {
    SearchMetrics {
        score: state.score,
        state: state.last_state,
        intent: state.last_intent,
        opp_cpi: state.prev_opp_cpi.unwrap_or(0),
        own_cpi: state.prev_own_cpi.unwrap_or(0),
        opp_freedom: state.prev_opp_freedom.unwrap_or(0),
        opp_breaks: state.prev_opp_breaks.unwrap_or(0),
        risk: state.last_risk,
        pressure: state.pressure_state.pressure,
        sustained_plies: state.pressure_state.sustained_plies,
        concession_swing: state.last_concession.map(|c| c.swing_cp),
        musttry_fired: state.last_musttry,
        verified: state.last_verified,
        simplify: state.simplify_bias,
        reset: state.reset_mode || state.last_state == PositionState::Reset,
        multipv_count: state.multipv_lines.len(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuiteCategory {
    Defensive,
    Quiet,
    Opportunity,
    MustTry,
    Pressure,
    Conversion,
}

#[derive(Debug, Clone, Default)]
pub struct SuiteSummary {
    pub n: usize,

    pub mtr: f64,

    pub pcr: f64,

    pub cri: f64,

    pub rsi: f64,

    pub odi_proxy: f64,

    pub dsi: f64,

    pub cdi: f64,

    pub aui: f64,

    pub mai: f64,

    pub rae: f64,

    pub ppi: f64,

    pub pei: f64,

    pub ioi: f64,

    pub omr: f64,

    pub fri: f64,

    pub pci: f64,
}

pub fn summarize(samples: &[(SearchMetrics, SuiteCategory)]) -> SuiteSummary {
    let rate = |num: usize, den: usize| {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64
        }
    };
    let mean = |vals: &[f64]| {
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    };
    let mut must_n = 0;
    let mut must_hit = 0;
    let mut must_verified = 0;
    let mut quiet_n = 0;
    let mut quiet_attack = 0;
    let mut conv_n = 0;
    let mut conv_hit = 0;
    let mut def_n = 0;
    let mut def_stable = 0;
    let mut def_reset = 0;
    let mut opp_n = 0;
    let mut opp_hit = 0;
    let mut opp_amplified = 0;
    let mut rae_vals: Vec<f64> = Vec::new();
    let mut ppi_vals: Vec<f64> = Vec::new();
    let mut pei_vals: Vec<f64> = Vec::new();
    let mut denied = 0;
    let mut owned = 0;
    let mut constrained = 0;
    let mut plan_blocked = 0;
    for (m, c) in samples {
        if m.opp_cpi <= 40 {
            denied += 1;
        }
        if m.opp_cpi <= m.own_cpi {
            owned += 1;
        }
        if m.opp_freedom <= 16 {
            constrained += 1;
        }
        if m.opp_breaks == 0 {
            plan_blocked += 1;
        }
        if m.risk > 0 {
            pei_vals.push(m.pressure as f64 / m.risk as f64);
        }
        match c {
            SuiteCategory::MustTry => {
                must_n += 1;
                must_hit += m.musttry_fired as usize;
                must_verified += m.verified.is_some() as usize;
                if m.musttry_fired {
                    rae_vals.push(m.score.max(0) as f64 / m.risk.max(1) as f64);
                }
            }
            SuiteCategory::Quiet => {
                quiet_n += 1;
                quiet_attack +=
                    matches!(m.state, PositionState::Attack | PositionState::Crush) as usize;
            }
            SuiteCategory::Conversion => {
                conv_n += 1;
                conv_hit += m.simplify as usize;
            }
            SuiteCategory::Defensive => {
                def_n += 1;
                def_stable += (m.state == PositionState::Defend) as usize;
                def_reset += m.reset as usize;
            }
            SuiteCategory::Opportunity => {
                opp_n += 1;
                opp_hit += m.concession_swing.is_some() as usize;
                opp_amplified += (m.simplify
                    || matches!(m.state, PositionState::Convert | PositionState::Crush))
                    as usize;
            }
            SuiteCategory::Pressure => {
                ppi_vals.push(m.sustained_plies as f64);
            }
        }
    }
    let total = samples.len();
    SuiteSummary {
        n: total,
        mtr: rate(must_hit, must_n),
        pcr: rate(quiet_attack, quiet_n),
        cri: rate(conv_hit, conv_n),
        rsi: rate(def_reset, def_n),
        odi_proxy: rate(opp_hit, opp_n),
        dsi: rate(def_stable, def_n),
        cdi: rate(denied, total),
        aui: rate(must_verified, must_n),
        mai: rate(opp_amplified, opp_n),
        rae: mean(&rae_vals),
        ppi: mean(&ppi_vals),
        pei: mean(&pei_vals),
        ioi: rate(owned, total),
        omr: 1.0 - rate(opp_hit, opp_n),
        fri: rate(constrained, total),
        pci: rate(plan_blocked, total),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(state: PositionState) -> SearchMetrics {
        SearchMetrics {
            score: 0,
            state,
            intent: SearchIntent::Improvement,
            opp_cpi: 20,
            own_cpi: 20,
            opp_freedom: 20,
            opp_breaks: 8,
            risk: 0,
            pressure: 0,
            sustained_plies: 0,
            concession_swing: None,
            musttry_fired: false,
            verified: None,
            simplify: false,
            reset: false,
            multipv_count: 1,
        }
    }

    #[test]
    fn collect_reads_search_state() {
        let mut state = SearchState::new();
        state.score = 42;
        state.prev_opp_cpi = Some(7);
        state.prev_own_cpi = Some(9);
        state.last_musttry = true;
        state.last_risk = 33;
        let m = collect(&state);
        assert_eq!(m.score, 42);
        assert_eq!(m.opp_cpi, 7);
        assert_eq!(m.own_cpi, 9);
        assert!(m.musttry_fired);
        assert_eq!(m.risk, 33);
        assert_eq!(m.state, PositionState::Improve);
    }

    #[test]
    fn collect_flags_reset_state() {
        let mut state = SearchState::new();
        state.last_state = PositionState::Reset;
        state.reset_mode = false;
        assert!(collect(&state).reset);
    }

    #[test]
    fn summary_rates() {
        let mut a = metric(PositionState::Crush);
        a.musttry_fired = true;
        a.verified = Some(200);
        a.score = 200;
        a.risk = 50;
        let mut b = metric(PositionState::Improve);
        b.simplify = true;
        let samples = vec![
            (a, SuiteCategory::MustTry),
            (metric(PositionState::Improve), SuiteCategory::MustTry),
            (metric(PositionState::Attack), SuiteCategory::Quiet),
            (metric(PositionState::Improve), SuiteCategory::Quiet),
            (b, SuiteCategory::Conversion),
        ];
        let s = summarize(&samples);
        assert_eq!(s.n, 5);
        assert!((s.mtr - 0.5).abs() < 1e-9);
        assert!((s.pcr - 0.5).abs() < 1e-9);
        assert!((s.cri - 1.0).abs() < 1e-9);
        assert!((s.aui - 0.5).abs() < 1e-9);
        assert!((s.rae - 4.0).abs() < 1e-9);
        assert!((s.cdi - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_batch_is_zero_not_nan() {
        let s = summarize(&[]);
        assert_eq!(s.mtr, 0.0);
        assert_eq!(s.pcr, 0.0);
        assert_eq!(s.omr, 1.0);
    }
}
