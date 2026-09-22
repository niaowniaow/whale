use crate::board::node_threats::NodeThreats;
use crate::board::plans;
use crate::board::state::BoardState;
use crate::common::constants::{ASPIRATION_WINDOW_MARGIN, MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::side::Side;
use crate::search::attack;
use crate::search::concession;
use crate::search::conversion;
use crate::search::counterplay;
use crate::search::multipv::MultipvLine;
use crate::search::negamax;
use crate::search::position_state;
use crate::search::pressure;
use crate::search::pv_table::PvTable;
use crate::search::risk;
use crate::search::search_state::SearchState;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

fn worker_search(
    mut board: BoardState,
    max_depth: u8,
    cancellation_token: &AtomicBool,
    mut search_state: SearchState,
    thread_id: usize,
) -> (u64, u64, u32) {
    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    let start_depth = if thread_id % 2 == 1 { 1 } else { 2 };
    for current_depth in start_depth..=max_depth {
        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if search_state.max_nodes > 0 && search_state.nodes >= search_state.max_nodes {
            break;
        }

        let score = negamax::search(
            &mut board,
            current_depth,
            i16::MIN + 1,
            i16::MAX - 1,
            cancellation_token,
            &previous_pv,
            &mut pv_table,
            &mut search_state,
        );

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if search_state.max_nodes > 0 && search_state.nodes >= search_state.max_nodes {
            break;
        }
        let _ = score;
        let current_pv = pv_table.line().to_vec();
        if !current_pv.is_empty() {
            previous_pv = current_pv;
        }
    }
    (
        search_state.nodes,
        search_state.tbhits,
        search_state.seldepth,
    )
}

fn search_excluded_root(
    board_state: &mut BoardState,
    depth: u8,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
    excluded: &[Move],
) -> Option<(Move, i16, Vec<Move>)> {
    let mut all = MoveList::new();
    board_state.generate_moves(&mut all);
    let nt = NodeThreats::compute(board_state);
    let saved = std::mem::take(&mut search_state.searchmoves);
    let uci_active = !saved.is_empty();
    let mut allowed = Vec::new();
    for i in 0..all.len() {
        let m = all[i].mv;
        if excluded.contains(&m) {
            continue;
        }
        if uci_active && !saved.contains(&m) {
            continue;
        }
        if board_state.is_legal_with(m, nt.checkers, nt.pinned) {
            allowed.push(m);
        }
    }
    if allowed.is_empty() {
        search_state.searchmoves = saved;
        return None;
    }
    search_state.searchmoves = allowed;
    let mut pv = PvTable::new();
    let score = negamax::search(
        board_state,
        depth,
        i16::MIN + 1,
        i16::MAX - 1,
        cancellation_token,
        &[],
        &mut pv,
        search_state,
    );
    let line = pv.line().to_vec();
    let bm = line.first().copied().unwrap_or(Move::NO_MOVE);
    search_state.searchmoves = saved;
    if cancellation_token.load(Ordering::Relaxed) || bm == Move::NO_MOVE {
        return None;
    }
    Some((bm, score, line))
}

fn search_single_root(
    board_state: &mut BoardState,
    depth: u8,
    candidate: Move,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
) -> Option<(i16, Vec<Move>)> {
    let saved = std::mem::take(&mut search_state.searchmoves);
    search_state.searchmoves = vec![candidate];
    let mut pv = PvTable::new();
    let score = negamax::search(
        board_state,
        depth,
        i16::MIN + 1,
        i16::MAX - 1,
        cancellation_token,
        &[],
        &mut pv,
        search_state,
    );
    let line = pv.line().to_vec();
    search_state.searchmoves = saved;
    if cancellation_token.load(Ordering::Relaxed) {
        return None;
    }
    if line.first().copied().unwrap_or(Move::NO_MOVE) != candidate {
        return None;
    }
    Some((score, line))
}

#[allow(clippy::too_many_arguments)]
fn search_primary(
    board_state: &mut BoardState,
    _depth: u8,
    cancellation_token: &AtomicBool,
    debug_mode: &mut bool,
    search_state: &mut SearchState,
    worker_nodes: &AtomicU64,
    max_depth: u8,
    legal_root_moves: i32,
    use_tm: bool,
) {
    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    let mut last_score: i16 = 0;
    let mut best_move_so_far = Move::NO_MOVE;
    let mut last_best_move_depth = 1u8;
    let mut completed_depth = 0u8;
    let prev_score = search_state.best_previous_score.unwrap_or(0);
    let mut iter_scores = [prev_score; 4];
    let mut iter_idx = 0usize;

    let timer = Instant::now();

    for current_depth in 1..=max_depth {
        let iter_start_nodes = search_state.nodes;

        if search_state.max_nodes > 0
            && search_state.nodes + worker_nodes.load(Ordering::Relaxed) >= search_state.max_nodes
        {
            cancellation_token.store(true, Ordering::Relaxed);
            break;
        }

        let us = board_state.side_to_move;
        let avg_sf = (last_score as i32) * 208 / 100;
        let opt = if avg_sf != 0 {
            114 * avg_sf / (avg_sf.abs() + 85)
        } else {
            0
        };
        search_state.optimism[us as usize] = opt;
        search_state.optimism[us.other() as usize] = -opt;

        let mut alpha = i16::MIN + 1;
        let mut beta = i16::MAX - 1;

        if current_depth > 1 {
            if last_score.abs() as i32 > MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32 {
                alpha = i16::MIN + 1;
                beta = i16::MAX - 1;
            } else {
                alpha = last_score
                    .saturating_sub(ASPIRATION_WINDOW_MARGIN)
                    .max(i16::MIN + 1);
                beta = last_score
                    .saturating_add(ASPIRATION_WINDOW_MARGIN)
                    .min(i16::MAX - 1);
            }
        }

        let mut current_score = last_score;
        let mut completed = false;
        let mut delta = ASPIRATION_WINDOW_MARGIN;

        loop {
            let score = negamax::search(
                board_state,
                current_depth,
                alpha,
                beta,
                cancellation_token,
                &previous_pv,
                &mut pv_table,
                search_state,
            );

            if cancellation_token.load(Ordering::Relaxed) {
                break;
            }

            current_score = score;
            if current_score <= alpha {
                if alpha == i16::MIN + 1 {
                    completed = true;
                    break;
                }
                beta = alpha;
                alpha = current_score.saturating_sub(delta).max(i16::MIN + 1);
            } else if current_score >= beta {
                if beta == i16::MAX - 1 {
                    completed = true;
                    break;
                }
                alpha = beta.saturating_sub(delta).max(alpha);
                beta = current_score.saturating_add(delta).min(i16::MAX - 1);
            } else {
                completed = true;
                break;
            }
            delta = delta
                .saturating_add(delta.saturating_mul(47) / 128)
                .min(1500);
        }

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        let mut aprm_boost = 1.0f64;
        let mut aprm_state_txt = "improve";
        let mut aprm_cpi_us = 0i32;
        let mut aprm_cpi_opp = 0i32;
        let mut aprm_free_us = 0u32;
        let mut aprm_free_opp = 0u32;
        let mut aprm_breaks_opp = 0u32;
        let mut aprm_momentum = 0i16;
        let mut aprm_concession_txt = String::new();
        let mut aprm_musttry = false;
        let mut aprm_move_changed = false;
        if completed {
            let current_pv = pv_table.line().to_vec();
            let new_best_move = current_pv.first().copied().unwrap_or(Move::NO_MOVE);
            let move_changed = current_depth > 1
                && best_move_so_far != Move::NO_MOVE
                && new_best_move != Move::NO_MOVE
                && new_best_move != best_move_so_far;
            if move_changed {
                last_best_move_depth = current_depth;
            }
            aprm_move_changed = move_changed;
            if new_best_move != Move::NO_MOVE {
                best_move_so_far = new_best_move;
                search_state.ponder_move = current_pv.get(1).copied().unwrap_or(Move::NO_MOVE);
            }

            let prev_iter_score = last_score;
            last_score = current_score;
            search_state.score = current_score;
            completed_depth = current_depth;
            if !current_pv.is_empty() {
                previous_pv = current_pv;
            }
            search_state.best_move = best_move_so_far;

            iter_scores[iter_idx] = current_score;
            iter_idx = (iter_idx + 1) & 3;

            {
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
                let king_danger =
                    heads.tactical_danger as i32 / 5 + if root_in_check { 20 } else { 0 };

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
                    if let Some(c) =
                        search_state
                            .concession_tracker
                            .observe(prev, current_score, threshold)
                    {
                        aprm_concession_txt = format!(" concession={:?}+{}", c.kind, c.swing_cp);
                        search_state.last_concession = Some(c);
                    }
                }

                if search_state.params.pressure_enabled {
                    let cpi_delta = opp_cpi - search_state.prev_opp_cpi.unwrap_or(opp_cpi);
                    let free_delta =
                        opp_free as i32 - search_state.prev_opp_freedom.unwrap_or(opp_free) as i32;
                    let plans_denied = search_state.prev_opp_breaks.unwrap_or(opp_breaks) as i32
                        - opp_breaks as i32;
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

                let is_mate_score =
                    current_score.abs() as i32 >= MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32;
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
                let urgency =
                    attack::urgency_for_with(window, gain, &search_state.urgency_thresholds);
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
                    && conversion::should_convert(
                        current_score,
                        opp_cpi,
                        &search_state.conversion_params,
                    );

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
                        position_state::PositionState::Defend => {
                            crate::search::intent::SearchIntent::Survival
                        }
                        position_state::PositionState::Stabilize => {
                            crate::search::intent::SearchIntent::Stabilization
                        }
                        position_state::PositionState::Press => {
                            crate::search::intent::SearchIntent::Pressure
                        }
                        position_state::PositionState::Attack => {
                            crate::search::intent::SearchIntent::Attack
                        }
                        position_state::PositionState::Reset => {
                            crate::search::intent::SearchIntent::Recovery
                        }
                        _ => crate::search::intent::SearchIntent::Improvement,
                    };
                }
                search_state.last_intent = intent;
                aprm_boost = crate::search::intent::effects(intent).time_boost_x100 as f64 / 100.0;
                search_state.behavior_us += aprm_start.elapsed().as_micros() as u64;
            }
        }

        let total_nodes_now = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
        if use_tm {
            let elapsed = timer.elapsed().as_millis() as f64;
            let iter_nodes = search_state.nodes.saturating_sub(iter_start_nodes).max(1);
            let nodes_effort = (search_state.root_best_move_nodes.saturating_mul(100_000)
                / iter_nodes.max(1))
            .clamp(0, 100_000);

            let high_best_move_effort = {
                let t = (nodes_effort as i64 - 75800) as f64 / (104510 - 75800) as f64;
                (0.969 + (0.714 - 0.969) * t).clamp(0.693, 0.838)
            };

            let prev_diff = (prev_score as f64 - current_score as f64) * 2.08;
            let iter_diff = (iter_scores[iter_idx] as f64 - current_score as f64) * 2.08;
            let falling_eval =
                ((11.48 + 2.30 * prev_diff + 1.10 * iter_diff) / 100.0).clamp(0.70, 1.40);

            let stability_depth = current_depth.saturating_sub(last_best_move_depth) as f64;
            let t = (stability_depth - 4.96) / (18.79 - 4.96);
            let time_reduction = (0.639 + (1.712 - 0.639) * t).clamp(0.629, 1.544);
            let reduction =
                (1.468 + search_state.previous_time_reduction) / (2.284 * time_reduction);
            search_state.previous_time_reduction = time_reduction;

            let tot_best_move_changes = search_state.best_move_changes;
            search_state.best_move_changes = 0;
            let best_move_instability = (1.0 + 0.35 * (tot_best_move_changes as f64)).min(1.6);

            let disagreement = (prev_score as i32 - current_score as i32).abs();
            let dad_time_factor = if current_depth >= 6 && search_state.params.dad_enabled {
                if disagreement > 60 {
                    1.25
                } else if disagreement < 15 {
                    0.85
                } else {
                    1.0
                }
            } else {
                1.0
            };

            let mut total_time = (search_state.opt_time as f64)
                * falling_eval
                * reduction
                * best_move_instability
                * high_best_move_effort
                * dad_time_factor
                * aprm_boost;

            if legal_root_moves <= 1 {
                total_time = total_time.min(500.0);
            }

            let max_limit = if search_state.max_time > 0 {
                search_state.max_time as f64
            } else {
                (search_state.opt_time as f64) * 2.5
            };

            let stop_time = total_time.min(max_limit);
            let is_mate = current_score.abs() as i32 >= (MAX_CENTIPAWN_EVAL as i32 - 5);

            let nodes_exceeded =
                search_state.max_nodes > 0 && total_nodes_now >= search_state.max_nodes;

            if elapsed > stop_time || is_mate || elapsed > total_time * 0.60 || nodes_exceeded {
                cancellation_token.store(true, Ordering::Relaxed);
                break;
            }
        }

        let time_ms = timer.elapsed().as_millis().max(1) as f64;
        let total_nodes = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
        let nps = (total_nodes as f64 / time_ms * 1000.0) as i32;

        if *debug_mode {
            let pv_string = previous_pv
                .iter()
                .map(|m| {
                    let promotion = m
                        .promotion_char()
                        .map(|c| c.to_string())
                        .unwrap_or_else(String::new);
                    format!("{}{}{}", m.source, m.target, promotion)
                })
                .collect::<Vec<String>>()
                .join(" ");
            let score_str = format_score(search_state.score);

            let wdl_str = if search_state.show_wdl {
                let (w, d, l) = crate::search::draw::cp_to_wdl(search_state.score);
                format!(" wdl {w} {d} {l}")
            } else {
                String::new()
            };
            println!(
                "info depth {} seldepth {} score {}{} nodes {} tbhits {} time {} nps {} pv {}",
                current_depth,
                search_state.seldepth,
                score_str,
                wdl_str,
                total_nodes,
                search_state.tbhits,
                time_ms,
                nps,
                pv_string
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        if completed && best_move_so_far != Move::NO_MOVE {
            let mut verified_txt = String::new();
            if search_state.params.attack_enabled
                && aprm_musttry
                && (aprm_move_changed || current_depth <= 1)
                && !cancellation_token.load(Ordering::Relaxed)
            {
                let vdepth = current_depth.saturating_add(search_state.verification_budget);
                if let Some((vscore, _vpv)) = search_single_root(
                    board_state,
                    vdepth,
                    best_move_so_far,
                    cancellation_token,
                    search_state,
                ) {
                    search_state.last_verified = Some(vscore);
                    if vscore >= current_score.saturating_sub(search_state.verify_tolerance_cp) {
                        verified_txt = format!(" verified={vscore}");
                    } else {
                        search_state.reset_mode = true;
                        search_state.last_attack_failed = true;
                        search_state.verification_budget = 0;
                        verified_txt = format!(" refuted={vscore}");
                    }
                }
            }
            let want_extra = search_state.multipv.saturating_sub(1);
            let mut lines: Vec<MultipvLine> = Vec::new();
            lines.push(MultipvLine {
                mv: best_move_so_far,
                score: current_score,
                pv: previous_pv.clone(),
                kind: crate::search::multipv::classify_candidate_rich(
                    best_move_so_far,
                    current_score,
                    current_score,
                    search_state.last_musttry,
                    search_state.last_state,
                ),
            });
            if want_extra > 0 && !cancellation_token.load(Ordering::Relaxed) {
                let mut found = vec![best_move_so_far];
                for _ in 0..want_extra {
                    if cancellation_token.load(Ordering::Relaxed) {
                        break;
                    }
                    match search_excluded_root(
                        board_state,
                        current_depth,
                        cancellation_token,
                        search_state,
                        &found,
                    ) {
                        Some((bm, sc, pv)) => {
                            found.push(bm);
                            lines.push(MultipvLine {
                                mv: bm,
                                score: sc,
                                pv: pv.clone(),
                                kind: crate::search::multipv::classify_candidate_rich(
                                    bm,
                                    sc,
                                    current_score,
                                    false,
                                    search_state.last_state,
                                ),
                            });
                        }
                        None => break,
                    }
                }
            }
            search_state.multipv_lines = lines.clone();
            if *debug_mode {
                let time_ms2 = timer.elapsed().as_millis().max(1) as f64;
                let total_nodes2 = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
                let nps2 = (total_nodes2 as f64 / time_ms2 * 1000.0) as i32;
                for (idx, l) in lines.iter().enumerate().skip(1) {
                    let pv_str =
                        l.pv.iter()
                            .map(|m| {
                                let promotion = m
                                    .promotion_char()
                                    .map(|c| c.to_string())
                                    .unwrap_or_else(String::new);
                                format!("{}{}{}", m.source, m.target, promotion)
                            })
                            .collect::<Vec<String>>()
                            .join(" ");
                    println!(
                        "info depth {} seldepth {} multipv {} score {} nodes {} time {} nps {} pv {}",
                        current_depth,
                        search_state.seldepth,
                        idx + 1,
                        format_score(l.score),
                        total_nodes2,
                        time_ms2,
                        nps2,
                        pv_str
                    );
                }
                let conv_txt = if search_state.params.conversion_enabled
                    && conversion::should_convert(
                        current_score,
                        aprm_cpi_opp,
                        &search_state.conversion_params,
                    ) {
                    " convert=yes"
                } else {
                    ""
                };
                let must_txt = if aprm_musttry { " musttry=yes" } else { "" };
                let reset_txt = if search_state.reset_mode {
                    " reset=yes"
                } else {
                    ""
                };
                let simpl_txt = if search_state.simplify_bias {
                    " simplify=yes"
                } else {
                    ""
                };
                println!(
                    "info string aprm depth {} state={} intent={} cpi_us={} cpi_opp={} freedom_us={} freedom_them={} plans_opp={} momentum={} pressure={} sustained={} vbudget={} risk={}{}{}{}{}{}{}",
                    current_depth,
                    aprm_state_txt,
                    search_state.last_intent.as_str(),
                    aprm_cpi_us,
                    aprm_cpi_opp,
                    aprm_free_us,
                    aprm_free_opp,
                    aprm_breaks_opp,
                    aprm_momentum,
                    search_state.pressure_state.pressure,
                    search_state.pressure_state.sustained_plies,
                    search_state.verification_budget,
                    search_state.last_risk,
                    aprm_concession_txt,
                    must_txt,
                    conv_txt,
                    reset_txt,
                    simpl_txt,
                    verified_txt
                );
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
        }
    }
    search_state.best_previous_score = Some(search_state.score);
    if completed_depth >= 1 {
        let white_pov = if board_state.side_to_move == Side::White {
            search_state.score
        } else {
            search_state.score.saturating_neg()
        };
        search_state.score_trend.push(white_pov, completed_depth);
    }
}

pub fn search(
    board_state: &mut BoardState,
    depth: u8,
    cancellation_token: &AtomicBool,
    debug_mode: &mut bool,
    search_state: &mut SearchState,
    num_threads: usize,
) {
    search_state.reset_search();
    search_state.tt.new_search();

    search_state.engine_side = board_state.side_to_move;

    let use_tm = search_state.opt_time > 0;

    let mut legal_root_moves = 0;
    {
        let mut root_moves = MoveList::new();
        board_state.generate_moves(&mut root_moves);

        let root_threats = NodeThreats::compute(board_state);
        let filter = !search_state.searchmoves.is_empty();
        for i in 0..root_moves.len() {
            let m = root_moves[i].mv;
            if filter && !search_state.searchmoves.contains(&m) {
                continue;
            }
            if board_state.is_legal_with(m, root_threats.checkers, root_threats.pinned) {
                legal_root_moves += 1;
                if legal_root_moves > 1 {
                    break;
                }
            }
        }

        if filter && legal_root_moves == 0 {
            for i in 0..root_moves.len() {
                let m = root_moves[i].mv;
                if board_state.is_legal_with(m, root_threats.checkers, root_threats.pinned) {
                    legal_root_moves += 1;
                    if legal_root_moves > 1 {
                        break;
                    }
                }
            }
        }
    }
    let mut max_depth = if use_tm && legal_root_moves <= 1 {
        1
    } else {
        depth
    };

    if search_state.mate_in > 0 {
        let mate_cap = search_state
            .mate_in
            .saturating_mul(2)
            .saturating_add(1)
            .max(1);
        max_depth = max_depth.min(mate_cap);
    }

    let worker_nodes = AtomicU64::new(0);
    let worker_tbhits = AtomicU64::new(0);
    let worker_seldepth = AtomicU32::new(0);
    let num_workers = num_threads.saturating_sub(1);

    if num_workers > 0 {
        std::thread::scope(|s| {
            for thread_id in 1..=num_workers {
                let worker_board = board_state.clone();
                let worker_search_state = search_state.clone_for_worker(thread_id);
                let worker_nodes_ref = &worker_nodes;
                let worker_tbhits_ref = &worker_tbhits;
                let worker_seldepth_ref = &worker_seldepth;
                s.spawn(move || {
                    let (nodes, tbhits, seldepth) = worker_search(
                        worker_board,
                        max_depth,
                        cancellation_token,
                        worker_search_state,
                        thread_id,
                    );
                    worker_nodes_ref.fetch_add(nodes, Ordering::Relaxed);
                    worker_tbhits_ref.fetch_add(tbhits, Ordering::Relaxed);
                    worker_seldepth_ref.fetch_max(seldepth, Ordering::Relaxed);
                });
            }

            search_primary(
                board_state,
                depth,
                cancellation_token,
                debug_mode,
                search_state,
                &worker_nodes,
                max_depth,
                legal_root_moves,
                use_tm,
            );

            cancellation_token.store(true, Ordering::Relaxed);
        });
    } else {
        search_primary(
            board_state,
            depth,
            cancellation_token,
            debug_mode,
            search_state,
            &worker_nodes,
            max_depth,
            legal_root_moves,
            use_tm,
        );
    }

    search_state.nodes += worker_nodes.load(Ordering::Relaxed);
    search_state.tbhits += worker_tbhits.load(Ordering::Relaxed);
    search_state.seldepth = search_state
        .seldepth
        .max(worker_seldepth.load(Ordering::Relaxed));
}

pub fn format_score(score: i16) -> String {
    let score_abs = (score as i32).abs();
    if (MAX_CENTIPAWN_EVAL as i32 - score_abs) <= MAX_PLY as i32 {
        let d = crate::common::constants::MAX_CENTIPAWN_EVAL as i32 - score_abs;
        let y = (d + 1) / 2;
        let sign = if score < 0 { -1 } else { 1 };
        format!("mate {}", y * sign)
    } else {
        format!("cp {}", score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::side::Side;

    #[test]
    fn test_format_score() {
        assert_eq!(format_score(100), "cp 100");
        assert_eq!(format_score(-500), "cp -500");
        assert_eq!(format_score(MAX_CENTIPAWN_EVAL - 1), "mate 1");
        assert_eq!(format_score(MAX_CENTIPAWN_EVAL - 3), "mate 2");
    }

    #[test]
    fn search_records_engine_side_from_root_stm() {
        for (fen, expected) in [
            (crate::common::helpers::STARTING_FEN, Side::White),
            (
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                Side::Black,
            ),
        ] {
            let mut board = BoardState::parse_fen(fen);
            let token = AtomicBool::new(false);
            let mut debug = false;
            let mut state = SearchState::new();
            search(&mut board, 2, &token, &mut debug, &mut state, 1);
            assert_eq!(state.engine_side, expected, "fen {fen}");
        }
    }

    #[test]
    fn contempt_is_engine_relative_at_the_root() {
        for fen in ["8/8/8/8/8/8/8/K6k w - - 0 1", "8/8/8/8/8/8/8/K6k b - - 0 1"] {
            let mut board = BoardState::parse_fen(fen);
            let token = AtomicBool::new(false);
            let mut debug = false;
            let mut state = SearchState::new();
            state.contempt_cp = 50;
            search(&mut board, 3, &token, &mut debug, &mut state, 1);
            assert!(
                (-51..=-50).contains(&state.score),
                "fen {fen}: contempt must devalue the draw for the engine, got {}",
                state.score
            );
        }
    }

    #[test]
    fn test_smp_search_multi_threaded() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 4, &token, &mut debug, &mut state, 4);
        assert_ne!(state.best_move, Move::NO_MOVE);
    }

    #[test]
    fn all_experimentals_off_still_finds_legal_move() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        for flag in [
            &mut state.params.cfss_enabled,
            &mut state.params.ras_enabled,
            &mut state.params.bmo_enabled,
            &mut state.params.tce_enabled,
            &mut state.params.lqt_enabled,
            &mut state.params.sps_enabled,
            &mut state.params.dad_enabled,
            &mut state.params.alp_enabled,
            &mut state.params.psm_enabled,
            &mut state.params.gtp_enabled,
        ] {
            *flag = false;
        }
        search(&mut board, 3, &token, &mut debug, &mut state, 1);
        assert_ne!(state.best_move, Move::NO_MOVE);
        assert!(board.is_legal(state.best_move));
        assert!(state.nodes > 0);
    }

    #[test]
    fn depth_one_finds_legal_move_single_thread() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 1, &token, &mut debug, &mut state, 1);
        assert_ne!(state.best_move, Move::NO_MOVE);
        assert!(board.is_legal(state.best_move));
        assert!(state.nodes > 0);
    }

    #[test]
    fn searchmoves_filter_pins_best_move() {
        use crate::common::move_type::MoveType;
        use crate::common::square::Square;
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        let pinned = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        state.searchmoves = vec![pinned];
        search(&mut board, 1, &token, &mut debug, &mut state, 1);
        assert_eq!(state.best_move, pinned);
    }

    #[test]
    fn illegal_searchmoves_filter_still_completes() {
        use crate::common::move_type::MoveType;
        use crate::common::square::Square;
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.searchmoves = vec![Move::new(Square::A1, Square::A8, MoveType::Quiet)];
        search(&mut board, 1, &token, &mut debug, &mut state, 1);

        assert!(state.nodes > 0);
    }

    #[test]
    fn max_nodes_stops_between_iterations() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.max_nodes = 1;
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert!(state.nodes >= 1);
    }

    #[test]
    fn mate_in_caps_depth_and_debug_prints() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = true;
        let mut state = SearchState::new();
        state.mate_in = 1;
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert_ne!(state.best_move, Move::NO_MOVE);
    }

    #[test]
    fn cancelled_entry_returns_no_move() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(true);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert_eq!(state.best_move, Move::NO_MOVE);
    }

    #[test]
    fn single_move_time_manager_caps_depth() {
        use crate::common::move_type::MoveType;
        use crate::common::square::Square;
        let mut board = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1");
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.opt_time = 100;
        state.max_time = 500;
        let pinned = Move::new(Square::H1, Square::G1, MoveType::Quiet);
        if board.is_legal(pinned) {
            state.searchmoves = vec![pinned];
        }
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert_ne!(state.best_move, Move::NO_MOVE);
    }

    #[test]
    fn mate_score_early_exit_with_time_manager() {
        let mut board = BoardState::parse_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.opt_time = 100;
        state.max_time = 500;
        search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert_ne!(state.best_move, Move::NO_MOVE);
        assert!(state.score > MAX_CENTIPAWN_EVAL - 100);
    }

    #[test]
    fn aspiration_sweep_over_volatile_positions() {
        for fen in [
            crate::common::helpers::STARTING_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
            "7k/8/8/8/8/8/QQQ5/K7 w - - 0 1",
        ] {
            let mut board = BoardState::parse_fen(fen);
            let token = AtomicBool::new(false);
            let mut debug = false;
            let mut state = SearchState::new();
            search(&mut board, 2, &token, &mut debug, &mut state, 1);
            assert!(state.score.abs() < MAX_CENTIPAWN_EVAL);
            assert!(state.nodes > 0);
        }
    }

    #[test]
    fn worker_max_nodes_abort_across_staggered_depths() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.max_nodes = 1;
        search(&mut board, 1, &token, &mut debug, &mut state, 3);
        assert!(state.nodes >= 1);
    }

    #[test]
    fn multithreaded_mate_finds_move_fast() {
        let mut board = BoardState::parse_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1");
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        search(&mut board, 1, &token, &mut debug, &mut state, 2);
        assert_ne!(state.best_move, Move::NO_MOVE);
        assert!(board.is_legal(state.best_move));
    }

    #[test]
    fn time_manager_without_stop_covers_falling_eval() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let token = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        state.opt_time = 100;
        state.max_time = 500;
        search(&mut board, 1, &token, &mut debug, &mut state, 1);
        assert_ne!(state.best_move, Move::NO_MOVE);
        assert!(board.is_legal(state.best_move));
    }
}
