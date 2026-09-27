use super::*;

pub(super) struct DiagOut {
    pub boost: f64,
    pub musttry: bool,
    pub state_txt: &'static str,
    pub cpi_us: i32,
    pub cpi_opp: i32,
    pub free_us: u32,
    pub free_opp: u32,
    pub breaks_opp: u32,
    pub momentum: i16,
    pub concession: String,
}

#[allow(clippy::needless_late_init)]
pub(super) fn update_diagnostics(
    board_state: &mut BoardState,
    current_score: i16,
    prev_iter_score: i16,
    best_move_so_far: Move,
    move_changed: bool,
    current_depth: u8,
    search_state: &mut SearchState,
) -> DiagOut {
    let aprm_boost;
    let aprm_state_txt;
    let aprm_cpi_us;
    let aprm_cpi_opp;
    let aprm_free_us;
    let aprm_free_opp;
    let aprm_breaks_opp;
    let aprm_momentum;
    let mut aprm_concession_txt = String::new();
    let mut aprm_musttry = false;
    let aprm_start = Instant::now();
    let root_nt = NodeThreats::compute(board_state);
    let root_in_check = root_nt.in_check();
    aprm_cpi_us = counterplay::compute_cpi_fast(&root_nt).cpi;
    aprm_free_us = counterplay::count_freedom(board_state);
    aprm_momentum = current_score.saturating_sub(prev_iter_score);
    let score_drop = current_score.saturating_sub(prev_iter_score);
    let mut opp_cpi = 0i32;
    let mut opp_free = 0u32;
    let mut opp_breaks = 0u32;
    if best_move_so_far != Move::NO_MOVE {
        board_state.make_move(best_move_so_far);
        let child_nt = NodeThreats::compute(board_state);
        opp_cpi = counterplay::compute_cpi_fast(&child_nt).cpi;
        opp_free = counterplay::count_freedom(board_state);

        opp_breaks = plans::available_pawn_breaks(board_state);
        board_state.unmake_move(best_move_so_far);
    }
    aprm_cpi_opp = opp_cpi;
    aprm_free_opp = opp_free;
    aprm_breaks_opp = opp_breaks;

    let heads = crate::eval::heads::compute_heads(
        board_state,
        current_score,
        aprm_cpi_us,
        opp_cpi,
        aprm_free_us,
        opp_free,
        aprm_momentum,
    );
    let king_danger = heads.tactical_danger as i32 / 5 + if root_in_check { 20 } else { 0 };

    let eval_diff = (current_score as i32 - prev_iter_score as i32).abs();
    let volatility = if eval_diff > 120 {
        position_state::VolatilityLevel::Extreme
    } else if eval_diff > 60 {
        position_state::VolatilityLevel::High
    } else if eval_diff > 25 {
        position_state::VolatilityLevel::Medium
    } else {
        position_state::VolatilityLevel::Low
    };

    let st = if search_state.params.state_enabled {
        position_state::classify_with_hysteresis(
            &position_state::StateInput {
                score: current_score,
                opp_cpi,
                own_cpi: aprm_cpi_us,
                momentum: aprm_momentum,
                in_check: root_in_check,
                volatility,
                score_drop,
                attack_failed: search_state.last_attack_failed,
            },
            &search_state.state_thresholds,
            Some(search_state.last_state),
        )
    } else {
        position_state::PositionState::Improve
    };
    aprm_state_txt = st.as_str();
    search_state.last_state = st;

    let stm = board_state.side_to_move;
    let trend_base = search_state.score_trend.smoothed().map(|w| {
        if stm == Side::White {
            w
        } else {
            w.saturating_neg()
        }
    });
    let baseline = trend_base.or(search_state.best_previous_score);
    if let Some(prev) = baseline {
        let vol_threshold = concession::confirmation_threshold(
            search_state.state_thresholds.concession_min_cp,
            volatility,
        );
        let threshold = vol_threshold.max(
            search_state
                .score_trend
                .adaptive_threshold(search_state.state_thresholds.concession_min_cp),
        );
        if let Some(c) = search_state
            .concession_tracker
            .observe(prev, current_score, threshold)
        {
            aprm_concession_txt = format!(" concession={:?}+{}", c.kind, c.swing_cp);
            search_state.last_concession = Some(c);
        }
    }

    if search_state.params.pressure_enabled {
        let cpi_delta = opp_cpi - search_state.prev_opp_cpi.unwrap_or(opp_cpi);
        let free_delta = opp_free as i32 - search_state.prev_opp_freedom.unwrap_or(opp_free) as i32;
        let plans_denied =
            search_state.prev_opp_breaks.unwrap_or(opp_breaks) as i32 - opp_breaks as i32;
        search_state.pressure_state = pressure::update_pressure_weighted(
            &search_state.pressure_state,
            aprm_momentum,
            cpi_delta,
            free_delta,
            plans_denied,
            &search_state.pressure_weights,
        );
    }
    search_state.prev_root_score = Some(current_score);
    search_state.prev_opp_cpi = Some(opp_cpi);
    search_state.prev_own_cpi = Some(aprm_cpi_us);
    search_state.prev_opp_freedom = Some(opp_free);
    search_state.prev_opp_breaks = Some(opp_breaks);

    let is_mate_score = current_score.abs() as i32 >= MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32;
    let gain = if is_mate_score {
        MAX_CENTIPAWN_EVAL as i32
    } else if current_depth == 1 {
        match baseline {
            Some(base) => current_score.saturating_sub(base).max(0) as i32,
            None => 0,
        }
    } else {
        aprm_momentum.max(0) as i32
    };
    let window = if move_changed {
        2
    } else if current_depth > 1 && search_state.best_move_changes > 0 {
        5
    } else if gain >= search_state.urgency_thresholds.musttry_gain {
        2
    } else {
        12
    };
    let urgency = attack::urgency_for_with(window, gain, &search_state.urgency_thresholds);
    search_state.last_urgency = urgency;

    search_state.verification_budget = if search_state.params.attack_enabled {
        attack::verification_depth(urgency)
    } else {
        0
    };

    if search_state.params.state_enabled {
        search_state.reset_mode = matches!(st, position_state::PositionState::Reset);
        if current_score >= prev_iter_score {
            search_state.last_attack_failed = false;
        }
    } else {
        search_state.reset_mode = false;
    }

    search_state.simplify_bias = search_state.params.conversion_enabled
        && conversion::should_convert(current_score, opp_cpi, &search_state.conversion_params);

    search_state.last_musttry = false;
    if search_state.params.risk_enabled {
        let risk = risk::risk_score((opp_cpi - aprm_cpi_us).max(0), king_danger, false);
        search_state.last_risk = risk;
        let input = risk::MustTryInput {
            gain_cp: gain,
            urgency,
            risk,
            opp_cpi_after: opp_cpi,
        };
        if risk::must_try_gate(&input, &search_state.risk_envelope) {
            aprm_musttry = true;
            search_state.last_musttry = true;
        }
    }
    let mut intent = crate::search::intent::intent_for(
        st,
        search_state.last_musttry,
        search_state.simplify_bias,
        is_mate_score,
    );
    if !search_state.params.conversion_enabled
        && intent == crate::search::intent::SearchIntent::Conversion
    {
        intent = match st {
            position_state::PositionState::Defend => crate::search::intent::SearchIntent::Survival,
            position_state::PositionState::Stabilize => {
                crate::search::intent::SearchIntent::Stabilization
            }
            position_state::PositionState::Press => crate::search::intent::SearchIntent::Pressure,
            position_state::PositionState::Attack => crate::search::intent::SearchIntent::Attack,
            position_state::PositionState::Reset => crate::search::intent::SearchIntent::Recovery,
            _ => crate::search::intent::SearchIntent::Improvement,
        };
    }
    search_state.last_intent = intent;
    aprm_boost = crate::search::intent::effects(intent).time_boost_x100 as f64 / 100.0;
    search_state.behavior_us += aprm_start.elapsed().as_micros() as u64;
    DiagOut {
        boost: aprm_boost,
        musttry: aprm_musttry,
        state_txt: aprm_state_txt,
        cpi_us: aprm_cpi_us,
        cpi_opp: aprm_cpi_opp,
        free_us: aprm_free_us,
        free_opp: aprm_free_opp,
        breaks_opp: aprm_breaks_opp,
        momentum: aprm_momentum,
        concession: aprm_concession_txt,
    }
}
