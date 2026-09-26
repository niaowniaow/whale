use super::context::{SearchContext, is_cancelled};
use super::core::search_internal;
use super::*;

pub(super) struct TbOut {
    pub score: Option<i16>,
    pub alpha: i16,
    pub best: i16,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tablebase_probe(
    board_state: &mut BoardState,
    ply: u8,
    depth: u8,
    alpha: i16,
    beta: i16,
    best_score: i16,
    is_pv_node: bool,
    halfmove: u8,
    ctx: &mut SearchContext,
) -> TbOut {
    let mut out = TbOut {
        score: None,
        alpha,
        best: best_score,
    };
    if ply > 0 && ctx.excluded_move.is_none() {
        let pieces = board_state.occupancy().count_ones() as usize;
        let limit = crate::syzygy::probe_limit() as usize;
        let pdepth = crate::syzygy::probe_depth();
        let rule50_ok = board_state.half_move_clock == 0 || !crate::syzygy::use_50mr();
        let castle_ok = board_state.castle == Castle::NONE;
        let depth_ok = pieces < limit || depth >= pdepth;
        if pieces <= limit
            && pieces <= crate::syzygy::TB_MAX_PIECES
            && depth_ok
            && rule50_ok
            && castle_ok
            && let Some((tb_value, tb_bound)) = crate::syzygy::probe_bound(board_state, ply)
        {
            ctx.search_state.tbhits += 1;
            let tb_cutoff = match tb_bound {
                TranspositionEntryType::Exact => true,
                TranspositionEntryType::Beta => tb_value >= beta,
                TranspositionEntryType::Alpha => tb_value <= alpha,
                TranspositionEntryType::None => false,
            };
            if tb_cutoff {
                let tb_depth = (depth + 6).min(constants::MAX_PLY as u8 - 1);
                ctx.search_state.tt.submit_entry(
                    board_state.board_hash,
                    tt::TranspositionTable::adjust_score(tb_value, ply as i32, halfmove),
                    tb_depth,
                    Move::NO_MOVE,
                    tb_bound,
                );
                return TbOut {
                    score: Some(tb_value),
                    alpha: out.alpha,
                    best: out.best,
                };
            }

            if is_pv_node && tb_bound == TranspositionEntryType::Beta && tb_value > best_score {
                out.best = tb_value;
                if tb_value > alpha {
                    out.alpha = tb_value;
                    if alpha >= beta {
                        return TbOut {
                            score: Some(tb_value),
                            alpha: out.alpha,
                            best: out.best,
                        };
                    }
                }
            }
        }
        if is_cancelled(ctx) {
            return TbOut {
                score: Some(0),
                alpha: out.alpha,
                best: out.best,
            };
        }
    }
    out
}

pub(super) struct StaticInfo {
    pub in_check: bool,
    pub has_eval: bool,
    pub static_eval: i16,
    pub momentum: i16,
    pub disagreement: i16,
    pub is_improving: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn static_info(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    beta: i16,
    tt_entry: Option<tt::TranspositionTableEntry>,
    halfmove: u8,
    previous_move: Option<Move>,
    mate_bound: i16,
    ctx: &mut SearchContext,
) -> (NodeThreats, StaticInfo) {
    let nt = NodeThreats::compute(board_state);
    let in_check = nt.in_check();

    let mut static_eval = 0;
    let has_static_eval = !in_check;

    if in_check {
        if (ply as usize) >= 2 {
            ctx.search_state.eval_stack[ply as usize] =
                ctx.search_state.eval_stack[ply as usize - 2];
        } else {
            ctx.search_state.eval_stack[ply as usize] = i16::MIN;
        }
    } else {
        let optimism = ctx.search_state.optimism[board_state.side_to_move as usize];
        let raw_static_eval =
            crate::eval::evaluate_with_depth_cached(&mut *board_state, optimism, depth, &nt);
        let correction = ctx
            .search_state
            .correction_history
            .get_correction(board_state, previous_move);

        static_eval = (raw_static_eval as i32 + correction as i32)
            .clamp(-mate_bound as i32, mate_bound as i32) as i16;
        ctx.search_state.eval_stack[ply as usize] = static_eval;
    }
    let momentum = if !in_check
        && (ply as usize) >= 2
        && ctx.search_state.eval_stack[ply as usize - 2] != i16::MIN
    {
        static_eval.saturating_sub(ctx.search_state.eval_stack[ply as usize - 2])
    } else {
        0
    };
    let structural_disagreement = if has_static_eval && let Some(entry) = tt_entry {
        let tt_val = tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
        (tt_val as i32 - static_eval as i32)
            .abs()
            .min(i16::MAX as i32) as i16
    } else {
        0
    };
    let mut is_improving = !in_check
        && (ply as usize) >= 2
        && ctx.search_state.eval_stack[ply as usize - 2] != i16::MIN
        && static_eval > ctx.search_state.eval_stack[ply as usize - 2];
    if !in_check && static_eval >= beta {
        is_improving = true;
    }
    (
        nt,
        StaticInfo {
            in_check,
            has_eval: has_static_eval,
            static_eval,
            momentum,
            disagreement: structural_disagreement,
            is_improving,
        },
    )
}
pub(super) struct SingOut {
    pub score: Option<i16>,
    pub extension: i8,
    pub depth_bonus: u8,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn singular_search(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    beta: i16,
    static_eval: i16,
    has_static_eval: bool,
    in_check: bool,
    is_pv_node: bool,
    tt_entry: Option<tt::TranspositionTableEntry>,
    tt_best: Option<Move>,
    cut_node: bool,
    previous_move: Option<Move>,
    halfmove: u8,
    mate_bound: i16,
    ctx: &mut SearchContext,
) -> SingOut {
    let mut out = SingOut {
        score: None,
        extension: 0,
        depth_bonus: 0,
    };
    if !in_check
        && ctx.excluded_move.is_none()
        && depth >= 6
        && tt_entry.is_some()
        && tt_best.is_some()
    {
        #[allow(clippy::unnecessary_unwrap)]
        let entry = tt_entry.unwrap();
        if entry.depth >= depth - 3 && entry.entry_type != tt::TranspositionEntryType::Alpha {
            let original_score =
                tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
            let tt_pv_sing = is_pv_node || tt_entry.is_some();
            let tt_capture_sing = tt_best
                .map(|m| m.is_capture() || m.is_promotion())
                .unwrap_or(false);
            let corr_raw = ctx
                .search_state
                .correction_history
                .get_correction(board_state, previous_move) as i32;
            let corr_adj = (corr_raw.abs() / 198368).min(1000) as i16;
            let need = 6 + ((tt_pv_sing && !is_pv_node) as u8);
            if depth >= need && original_score.abs() < mate_bound {
                let singular_beta = (original_score
                    - (59 + 66 * ((tt_pv_sing && !is_pv_node) as i16)) * depth as i16 / 63)
                    .max(-constants::MAX_CENTIPAWN_EVAL);

                let mut se_pv_table = PvTable::new();

                let (se_pv, se_state) = (&mut se_pv_table, &mut *ctx.search_state);
                let mut se_ctx = SearchContext {
                    allow_null_move: false,
                    on_pv_path: false,
                    previous_pv: ctx.previous_pv,
                    excluded_move: tt_best,
                    cut_node,
                    gtp_graph: ctx.gtp_graph,
                    gtp_parent: ctx.gtp_parent,
                    pv_table: se_pv,
                    search_state: se_state,
                    cancellation_token: ctx.cancellation_token,
                };

                let se_score = search_internal(
                    board_state,
                    (depth - 1) / 2,
                    ply,
                    singular_beta - 1,
                    singular_beta,
                    previous_move,
                    &mut se_ctx,
                );

                if is_cancelled(ctx) {
                    return SingOut {
                        score: Some(0),
                        extension: out.extension,
                        depth_bonus: out.depth_bonus,
                    };
                }

                if se_score < singular_beta {
                    let double_margin = -2 + 204 * (is_pv_node as i16)
                        - 152 * ((!tt_capture_sing) as i16)
                        - corr_adj;
                    let triple_margin = 70 + 279 * (is_pv_node as i16)
                        - 188 * ((!tt_capture_sing) as i16)
                        + 81 * (tt_pv_sing as i16)
                        - corr_adj;
                    out.extension = 1
                        + (se_score < singular_beta - double_margin) as i8
                        + (se_score < singular_beta - triple_margin) as i8;
                    out.depth_bonus = 1;
                } else if se_score >= beta && se_score.abs() < mate_bound {
                    if has_static_eval && !in_check && se_score > static_eval {
                        let bonus = (((se_score as i32 - static_eval as i32)
                            * ((depth as i32 - 1) / 2)
                            * 177)
                            / 1024)
                            .clamp(-256, 256);
                        ctx.search_state.correction_history.update(
                            board_state,
                            previous_move,
                            bonus,
                        );
                    }
                    return SingOut {
                        score: Some(se_score),
                        extension: out.extension,
                        depth_bonus: out.depth_bonus,
                    };
                } else if original_score >= beta || cut_node {
                    out.extension = -3;
                }
            }
        }
    }
    out
}

#[inline(always)]
pub(super) fn mate_window(ply: u8, alpha: i16, beta: i16) -> (i16, i16, Option<i16>) {
    let mated = -constants::MAX_CENTIPAWN_EVAL + ply as i16;
    let mating = constants::MAX_CENTIPAWN_EVAL - ply as i16;
    let alpha = alpha.max(mated);
    let beta = beta.min(mating);
    if alpha >= beta {
        (alpha, beta, Some(alpha))
    } else {
        (alpha, beta, None)
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tt_cutoff(
    entry: tt::TranspositionTableEntry,
    is_pv_node: bool,
    has_excluded: bool,
    depth: u8,
    alpha: i16,
    beta: i16,
    cut_node: bool,
    ply: u8,
    halfmove: u8,
) -> Option<i16> {
    if is_pv_node || has_excluded {
        return None;
    }
    let tt_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
    let depth_cond = (entry.depth as i32) > depth as i32 - i32::from(tt_score < beta);
    let bound_ok = match entry.entry_type {
        TranspositionEntryType::Exact => true,
        TranspositionEntryType::Alpha => tt_score <= alpha && (!cut_node || depth > 5),
        TranspositionEntryType::Beta => tt_score >= beta && (cut_node || depth > 5),
        TranspositionEntryType::None => false,
    };
    if depth_cond && bound_ok && draw::allow_tt_cutoff(halfmove) {
        Some(tt_score)
    } else {
        None
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn razor_gate(
    board_state: &mut BoardState,
    alpha: i16,
    beta: i16,
    ply: u8,
    is_pv_node: bool,
    in_check: bool,
    depth: u8,
    static_eval: i16,
    has_static_eval: bool,
    beta_is_mate: bool,
    ctx: &mut SearchContext,
) -> Option<i16> {
    if has_static_eval
        && ctx.search_state.params.razor_enabled
        && razoring::should_razor_with_margin(
            is_pv_node,
            in_check,
            depth,
            static_eval,
            alpha,
            beta_is_mate,
            ctx.excluded_move.is_some(),
            ctx.search_state.params.razor_margin,
        )
    {
        return Some(quiescence::search(
            board_state,
            alpha,
            beta,
            ply,
            ctx.cancellation_token,
            ctx.search_state,
        ));
    }
    None
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn rfp_gate(
    nt: &NodeThreats,
    static_eval: i16,
    beta: i16,
    depth: u8,
    momentum: i16,
    is_improving: bool,
    is_pv_node: bool,
    has_static_eval: bool,
    has_excluded: bool,
    ply: u8,
    tt_entry: Option<tt::TranspositionTableEntry>,
    halfmove: u8,
    mate_bound: i16,
    beta_is_mate: bool,
    ctx: &mut SearchContext,
) -> Option<i16> {
    if is_pv_node || !has_static_eval || has_excluded {
        return None;
    }
    let mut margin = ctx.search_state.params.rfp_margin_mult * depth as i16;
    if !is_improving {
        margin += margin / 3;
    }
    if momentum < -100 {
        margin += 60;
    }
    let tt_hit_rfp = tt_entry.is_some();
    let opponent_worsening = ply > 0
        && ctx.search_state.eval_stack[(ply as usize).saturating_sub(1)] != i16::MIN
        && (static_eval as i32)
            > -(ctx.search_state.eval_stack[(ply as usize).saturating_sub(1)] as i32);
    if !tt_hit_rfp {
        margin = margin.saturating_sub(20 * depth as i16);
    }
    if opponent_worsening {
        margin -= margin / 16;
    }

    if ctx.search_state.params.cpi_enabled {
        margin += counterplay::cpi_rfp_adjust(counterplay::compute_cpi_fast(nt).cpi);
    }

    let intent_eff = crate::search::intent::effects(ctx.search_state.last_intent);
    margin += intent_eff.rfp_adjust;
    if ctx.search_state.last_intent == crate::search::intent::SearchIntent::Conversion {
        margin = margin.max(0);
    }
    margin += ctx
        .search_state
        .ras
        .rfp_margin_adjustment(ply as usize, depth);
    if !beta_is_mate && static_eval.saturating_sub(margin) >= beta {
        return Some(((661 * beta as i32 + 363 * static_eval as i32) / 1024) as i16);
    }

    let small_prob_beta = beta.saturating_add(380);
    if let Some(entry) = tt_entry
        && entry.entry_type != tt::TranspositionEntryType::Alpha
        && entry.depth >= depth.saturating_sub(4)
    {
        let tt_val = tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
        if tt_val >= small_prob_beta && tt_val.abs() < mate_bound {
            return Some(small_prob_beta);
        }
    }
    None
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn probcut_search(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    beta: i16,
    static_eval: i16,
    is_improving: bool,
    beta_is_mate: bool,
    in_check: bool,
    halfmove: u8,
    mate_bound: i16,
    nt: &NodeThreats,
    ctx: &mut SearchContext,
) -> Option<i16> {
    if depth >= 3 && ctx.excluded_move.is_none() && !beta_is_mate {
        let prob_beta = beta.saturating_add(241 - 64 * (is_improving as i16));
        if prob_beta < mate_bound
            && prob_beta > -mate_bound
            && static_eval >= prob_beta.saturating_sub(50)
        {
            let prob_depth = depth
                .saturating_sub(if is_improving { 5 } else { 3 })
                .max(1);
            let mut prob_captures = crate::common::move_list::MoveList::new();
            board_state.generate_captures(&mut prob_captures);
            crate::eval::move_ordering::populate_capture_scores(
                &mut prob_captures,
                board_state,
                &ctx.search_state.move_ordering,
            );
            for i in 0..prob_captures.len() {
                let mut best_idx = i;
                for j in (i + 1)..prob_captures.len() {
                    if prob_captures[j].score > prob_captures[best_idx].score {
                        best_idx = j;
                    }
                }
                if best_idx != i {
                    prob_captures.swap(i, best_idx);
                }
                let mv = prob_captures[i].mv;
                let see_threshold = prob_beta.saturating_sub(static_eval);
                if !board_state.see_ge(mv, see_threshold) {
                    continue;
                }
                if !board_state.is_legal_with(mv, nt.checkers, nt.pinned) {
                    continue;
                }
                board_state.make_move(mv);
                let q_score = -quiescence::search(
                    board_state,
                    -prob_beta,
                    -prob_beta + 1,
                    ply + 1,
                    ctx.cancellation_token,
                    ctx.search_state,
                );
                if is_cancelled(ctx) {
                    board_state.unmake_move(mv);
                    return Some(0);
                }
                let score = if q_score >= prob_beta && prob_depth > 1 {
                    let mut prob_graph = ctx.gtp_graph;
                    let prob_parent = prob_graph.add_node(gtp::GtpNode {
                        depth: prob_depth,
                        eval_margin: static_eval.saturating_sub(prob_beta),
                        is_capture: true,
                        in_check,
                        history_score: 0,
                        parent_idx: ctx.gtp_parent,
                    });
                    let mut prob_pv_table = PvTable::new();
                    let (prob_pv, prob_state) = (&mut prob_pv_table, &mut *ctx.search_state);
                    -search_internal(
                        board_state,
                        prob_depth,
                        ply + 1,
                        -prob_beta,
                        -prob_beta + 1,
                        Some(mv),
                        &mut SearchContext {
                            allow_null_move: false,
                            on_pv_path: false,
                            previous_pv: ctx.previous_pv,
                            excluded_move: None,
                            cut_node: false,
                            gtp_graph: prob_graph,
                            gtp_parent: Some(prob_parent),
                            pv_table: prob_pv,
                            cancellation_token: ctx.cancellation_token,
                            search_state: prob_state,
                        },
                    )
                } else {
                    q_score
                };
                board_state.unmake_move(mv);
                if is_cancelled(ctx) {
                    return Some(0);
                }
                if score >= prob_beta {
                    let returned_score = if score.abs() < constants::MAX_CENTIPAWN_EVAL - 200 {
                        score - (prob_beta - beta)
                    } else {
                        score
                    };
                    if ctx.excluded_move.is_none() && ctx.search_state.tt_store_allowed(ply) {
                        ctx.search_state.tt.submit_entry(
                            board_state.board_hash,
                            tt::TranspositionTable::adjust_score(score, ply as i32, halfmove),
                            prob_depth,
                            mv,
                            TranspositionEntryType::Beta,
                        );
                    }
                    return Some(returned_score);
                }
            }
        }
    }
    None
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn null_move_search(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    beta: i16,
    static_eval: i16,
    momentum: i16,
    is_improving: bool,
    is_pv_node: bool,
    in_check: bool,
    cut_node: bool,
    previous_move: Option<Move>,
    halfmove: u8,
    mate_bound: i16,
    ctx: &mut SearchContext,
) -> Option<i16> {
    if ctx.excluded_move.is_none()
        && nmp::can_prune(
            is_pv_node,
            board_state,
            ctx.allow_null_move,
            depth,
            in_check,
            static_eval,
            beta,
            momentum,
            cut_node,
            ply,
            is_improving,
        )
    {
        board_state.make_null_move();
        let eval_margin = static_eval.saturating_sub(beta);
        let reduction =
            nmp::get_reduction_with_margin(depth, &ctx.search_state.params, momentum, eval_margin);
        let reduced_depth = depth.saturating_sub(reduction).max(1);

        let mut nmp_graph = ctx.gtp_graph;
        let nmp_parent = nmp_graph.add_node(gtp::GtpNode {
            depth: reduced_depth,
            eval_margin,
            is_capture: false,
            in_check,
            history_score: 0,
            parent_idx: ctx.gtp_parent,
        });
        let mut nmp_pv_table = PvTable::new();
        let (nmp_pv, nmp_state) = (&mut nmp_pv_table, &mut *ctx.search_state);

        if (ply as usize) + 1 < constants::MAX_PLY {
            let own = nmp_state.extension_streak[ply as usize];
            nmp_state.extension_streak[(ply as usize) + 1] = own;
        }
        let score = -search_internal(
            board_state,
            reduced_depth,
            ply + 1,
            -beta,
            -beta + 1,
            None,
            &mut SearchContext {
                allow_null_move: false,
                on_pv_path: false,
                previous_pv: ctx.previous_pv,
                excluded_move: None,
                cut_node: false,
                gtp_graph: nmp_graph,
                gtp_parent: Some(nmp_parent),
                pv_table: nmp_pv,
                cancellation_token: ctx.cancellation_token,
                search_state: nmp_state,
            },
        );
        board_state.undo_null_move();

        if is_cancelled(ctx) {
            return Some(0);
        }

        if score >= beta && score < mate_bound {
            if nmp::get_nmp_min_ply() != 0 || depth < 16 {
                nmp::inc_prior_fail_high(ply);
                let mate_bound_inner = MAX_CENTIPAWN_EVAL - MAX_PLY as i16;
                let score_out = if score >= mate_bound_inner {
                    beta
                } else {
                    score
                };
                if ctx.excluded_move.is_none() && ctx.search_state.tt_store_allowed(ply) {
                    ctx.search_state.tt.submit_entry(
                        board_state.board_hash,
                        tt::TranspositionTable::adjust_score(score_out, ply as i32, halfmove),
                        reduced_depth,
                        Move::NO_MOVE,
                        TranspositionEntryType::Beta,
                    );
                }
                return Some(score_out);
            }
            let vern_depth = depth.saturating_sub(reduction).max(1);
            nmp::set_nmp_min_ply(ply.saturating_add(3 * vern_depth.saturating_sub(reduction) / 4));
            let verify_score = search_internal(
                board_state,
                vern_depth,
                ply,
                beta - 1,
                beta,
                previous_move,
                &mut SearchContext {
                    allow_null_move: false,
                    on_pv_path: false,
                    previous_pv: ctx.previous_pv,
                    excluded_move: None,
                    cut_node: false,
                    gtp_graph: ctx.gtp_graph,
                    gtp_parent: ctx.gtp_parent,
                    pv_table: &mut *ctx.pv_table,
                    cancellation_token: ctx.cancellation_token,
                    search_state: ctx.search_state,
                },
            );
            nmp::set_nmp_min_ply(0);
            if is_cancelled(ctx) {
                return Some(0);
            }
            if verify_score >= beta {
                nmp::inc_prior_fail_high(ply);
                let mate_bound_inner = MAX_CENTIPAWN_EVAL - MAX_PLY as i16;
                let score_out = if score >= mate_bound_inner {
                    beta
                } else {
                    score
                };
                if ctx.excluded_move.is_none() && ctx.search_state.tt_store_allowed(ply) {
                    ctx.search_state.tt.submit_entry(
                        board_state.board_hash,
                        tt::TranspositionTable::adjust_score(score_out, ply as i32, halfmove),
                        reduced_depth,
                        Move::NO_MOVE,
                        TranspositionEntryType::Beta,
                    );
                }
                return Some(score_out);
            }
        }
    }
    None
}

#[inline(always)]
pub(super) fn apply_iir(
    current_depth: u8,
    is_pv_node: bool,
    cut_node: bool,
    tt_hit: bool,
    has_excluded: bool,
    iir_enabled: bool,
) -> u8 {
    if !iir_enabled {
        return current_depth;
    }
    current_depth
        .saturating_sub(iir::reduction(
            is_pv_node,
            cut_node,
            current_depth,
            tt_hit,
            has_excluded,
        ))
        .max(1)
}

pub(super) struct CoarseOut {
    pub cancelled: bool,
    pub depth: u8,
    pub failed_low: bool,
    pub tt_best: Option<Move>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn coarse_pass(
    board_state: &mut BoardState,
    current_depth: u8,
    ply: u8,
    alpha: i16,
    beta: i16,
    is_pv_node: bool,
    in_check: bool,
    tt_best: Option<Move>,
    previous_move: Option<Move>,
    ctx: &mut SearchContext,
) -> CoarseOut {
    let mut out = CoarseOut {
        cancelled: false,
        depth: current_depth,
        failed_low: false,
        tt_best,
    };
    if ctx.search_state.params.cfss_enabled
        && cfss::should_run_coarse_pass(
            current_depth,
            is_pv_node,
            in_check,
            tt_best,
            ctx.excluded_move,
        )
    {
        let coarse_depth = cfss::get_coarse_depth(current_depth);

        let mut coarse_pv_table = PvTable::new();
        let (coarse_pv, coarse_state) = (&mut coarse_pv_table, &mut *ctx.search_state);
        let coarse_score = search_internal(
            board_state,
            coarse_depth,
            ply,
            alpha,
            beta,
            previous_move,
            &mut SearchContext {
                allow_null_move: false,
                on_pv_path: false,
                previous_pv: ctx.previous_pv,
                excluded_move: None,
                cut_node: false,
                gtp_graph: ctx.gtp_graph,
                gtp_parent: ctx.gtp_parent,
                pv_table: coarse_pv,
                cancellation_token: ctx.cancellation_token,
                search_state: coarse_state,
            },
        );

        if is_cancelled(ctx) {
            return CoarseOut {
                cancelled: true,
                depth: out.depth,
                failed_low: out.failed_low,
                tt_best: out.tt_best,
            };
        }

        if let Some(entry) = ctx.search_state.tt.probe(board_state.board_hash)
            && entry.best_move != Move::NO_MOVE
        {
            out.tt_best = Some(entry.best_move);
        }

        if coarse_score <= alpha.saturating_sub(250) {
            out.failed_low = true;
            out.depth = current_depth.saturating_sub(1).max(1);
        }
    }
    out
}
