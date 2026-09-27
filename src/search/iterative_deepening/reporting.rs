use super::roots::{search_excluded_root, search_single_root};
use super::*;

pub(super) fn report_iteration(
    debug: bool,
    previous_pv: &[Move],
    current_depth: u8,
    search_state: &SearchState,
    timer: &Instant,
    worker_nodes: &AtomicU64,
) {
    if !debug {
        return;
    }
    let time_ms = timer.elapsed().as_millis().max(1) as f64;
    let total_nodes = search_state.nodes + worker_nodes.load(Ordering::Relaxed);
    let nps = (total_nodes as f64 / time_ms * 1000.0) as i32;

    if debug {
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
        let multipv_prefix = if search_state.multipv > 1 {
            "multipv 1 "
        } else {
            ""
        };
        println!(
            "info depth {} seldepth {} {}score {}{} nodes {} tbhits {} time {} nps {} pv {}",
            current_depth,
            search_state.seldepth,
            multipv_prefix,
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
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_verification(
    board_state: &mut BoardState,
    current_depth: u8,
    current_score: i16,
    best_move_so_far: Move,
    musttry: bool,
    move_changed: bool,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
) -> String {
    let mut verified_txt = String::new();
    if search_state.params.attack_enabled
        && musttry
        && (move_changed || current_depth <= 1)
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
    verified_txt
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_multipv(
    board_state: &mut BoardState,
    current_depth: u8,
    current_score: i16,
    best_move_so_far: Move,
    previous_pv: Vec<Move>,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
    timer: &Instant,
    worker_nodes: &AtomicU64,
) {
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
    if lines.len() > 1 {
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
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn report_diagnostics(
    debug: bool,
    board_state: &BoardState,
    current_depth: u8,
    current_score: i16,
    verified_txt: String,
    state_txt: &str,
    cpi_us: i32,
    cpi_opp: i32,
    free_us: u32,
    free_opp: u32,
    breaks_opp: u32,
    momentum: i16,
    concession_txt: &str,
    musttry: bool,
    search_state: &SearchState,
) {
    if !debug {
        return;
    }
    let conv_txt = if search_state.params.conversion_enabled
        && conversion::should_convert(current_score, cpi_opp, &search_state.conversion_params)
    {
        " convert=yes"
    } else {
        ""
    };
    let must_txt = if musttry { " musttry=yes" } else { "" };
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
    let learned_txt =
        if search_state.params.learned_enabled && crate::search::learned::ensure_loaded() {
            let side = if board_state.side_to_move == Side::White {
                1.0
            } else {
                -1.0
            };
            let features = [
                current_score as f32 / 100.0,
                search_state.pressure_state.pressure as f32 / 50.0,
                cpi_opp as f32 / 50.0,
                cpi_us as f32 / 50.0,
                crate::search::learned::urgency_id(search_state.last_urgency) as f32,
                search_state.last_risk as f32 / 100.0,
                if search_state.last_musttry { 1.0 } else { 0.0 },
                search_state
                    .last_concession
                    .map(|c| c.swing_cp as f32 / 50.0)
                    .unwrap_or(0.0),
                crate::search::learned::state_id(search_state.last_state) as f32,
                crate::search::learned::intent_id(search_state.last_intent) as f32,
                side,
            ];
            match crate::search::learned::predict_cached(features) {
                Some(p) => format!(" learned={:.2}", p),
                None => String::new(),
            }
        } else {
            String::new()
        };
    println!(
        "info string aprm depth {} state={} intent={} cpi_us={} cpi_opp={} freedom_us={} freedom_them={} plans_opp={} momentum={} pressure={} sustained={} vbudget={} risk={}{}{}{}{}{}{}{}",
        current_depth,
        state_txt,
        search_state.last_intent.as_str(),
        cpi_us,
        cpi_opp,
        free_us,
        free_opp,
        breaks_opp,
        momentum,
        search_state.pressure_state.pressure,
        search_state.pressure_state.sustained_plies,
        search_state.verification_budget,
        search_state.last_risk,
        concession_txt,
        must_txt,
        conv_txt,
        reset_txt,
        simpl_txt,
        verified_txt,
        learned_txt
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

pub(super) fn conclude_search(
    board_state: &mut BoardState,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
    debug_mode: bool,
    last_score: i16,
    best_move_so_far: Move,
    completed_depth: u8,
) {
    let conspiracy_unstable = search_state.params.conspiracy_enabled
        && search_state.multipv_lines.len() >= 2
        && conspiracy::needs_resolution(
            &conspiracy::rank_by_stability(
                &search_state
                    .multipv_lines
                    .iter()
                    .map(|l| l.mv)
                    .collect::<Vec<_>>(),
                &search_state
                    .multipv_lines
                    .iter()
                    .map(|l| l.score)
                    .collect::<Vec<_>>(),
            ),
            search_state.params.conspiracy_tolerance,
            1,
        );
    if search_state.params.conspiracy_enabled
        && completed_depth >= 1
        && best_move_so_far != Move::NO_MOVE
        && (conspiracy_unstable || search_state.multipv_lines.len() < 2)
        && !cancellation_token.load(Ordering::Relaxed)
        && let Some((verified_score, _)) = search_single_root(
            board_state,
            completed_depth,
            best_move_so_far,
            cancellation_token,
            search_state,
        )
    {
        let tolerance = search_state.params.conspiracy_tolerance as i32;
        let stable = (verified_score as i32 - last_score as i32).abs() <= tolerance;
        if debug_mode {
            println!(
                "info string conspiracy depth {} {} (iter {} vs verify {})",
                completed_depth,
                if stable {
                    "verified=yes"
                } else {
                    "verified=no"
                },
                last_score,
                verified_score
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        if stable {
            search_state.score = verified_score;
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
