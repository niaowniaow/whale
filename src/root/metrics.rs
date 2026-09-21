use crate::search::search_state::SearchState;
use crate::world::position_state::PositionState;

#[derive(Debug, Clone)]
pub struct SearchMetrics {
    pub score: i16,
    pub state: PositionState,
    pub opp_cpi: i32,
    pub opp_freedom: u32,
    pub opp_breaks: u32,
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
        opp_cpi: state.prev_opp_cpi.unwrap_or(0),
        opp_freedom: state.prev_opp_freedom.unwrap_or(0),
        opp_breaks: state.prev_opp_breaks.unwrap_or(0),
        pressure: state.pressure_state.pressure,
        sustained_plies: state.pressure_state.sustained_plies,
        concession_swing: state.last_concession.map(|c| c.swing_cp),
        musttry_fired: state.last_musttry,
        verified: state.last_verified,
        simplify: state.simplify_bias,
        reset: state.reset_mode,
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
}

pub fn summarize(samples: &[(SearchMetrics, SuiteCategory)]) -> SuiteSummary {
    let rate = |num: usize, den: usize| {
        if den == 0 {
            0.0
        } else {
            num as f64 / den as f64
        }
    };
    let mut must_n = 0;
    let mut must_hit = 0;
    let mut quiet_n = 0;
    let mut quiet_attack = 0;
    let mut conv_n = 0;
    let mut conv_hit = 0;
    let mut def_n = 0;
    let mut def_hit = 0;
    let mut opp_n = 0;
    let mut opp_hit = 0;
    for (m, c) in samples {
        match c {
            SuiteCategory::MustTry => {
                must_n += 1;
                must_hit += m.musttry_fired as usize;
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
                def_hit += m.reset as usize;
            }
            SuiteCategory::Opportunity => {
                opp_n += 1;
                opp_hit += m.concession_swing.is_some() as usize;
            }
            SuiteCategory::Pressure => {}
        }
    }
    SuiteSummary {
        n: samples.len(),
        mtr: rate(must_hit, must_n),
        pcr: rate(quiet_attack, quiet_n),
        cri: rate(conv_hit, conv_n),
        rsi: rate(def_hit, def_n),
        odi_proxy: rate(opp_hit, opp_n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(state: PositionState) -> SearchMetrics {
        SearchMetrics {
            score: 0,
            state,
            opp_cpi: 0,
            opp_freedom: 20,
            opp_breaks: 8,
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
        state.last_musttry = true;
        let m = collect(&state);
        assert_eq!(m.score, 42);
        assert_eq!(m.opp_cpi, 7);
        assert!(m.musttry_fired);
        assert_eq!(m.state, PositionState::Improve);
    }

    #[test]
    fn summary_rates() {
        let mut a = metric(PositionState::Crush);
        a.musttry_fired = true;
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
    }

    #[test]
    fn empty_batch_is_zero_not_nan() {
        let s = summarize(&[]);
        assert_eq!(s.mtr, 0.0);
        assert_eq!(s.pcr, 0.0);
    }
}
