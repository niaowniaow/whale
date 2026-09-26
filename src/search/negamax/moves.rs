use super::context::{SearchContext, is_cancelled};
use super::core::search_internal;
use super::history::update_history_stats;
use super::pvs::search_deeper;
use super::*;

#[inline]
pub(super) fn move_scores(
    board: &BoardState,
    state: &SearchState,
    move_obj: Move,
    cap_or_promo: bool,
    previous_move: Option<Move>,
) -> (i32, i16) {
    let mut history_score = 0;
    let mut captured_see_val: i16 = 0;
    if !cap_or_promo {
        history_score = state
            .move_ordering
            .get_quiet_history_score(board, move_obj, previous_move);
    } else if move_obj.is_capture() {
        let moved_piece = board.get_piece_on(move_obj.source);
        let captured_piece = if move_obj.move_type == MoveType::EnPassant {
            Piece::Pawn
        } else {
            board.piece_mapping[move_obj.target as usize]
        };
        captured_see_val = if captured_piece == Piece::None {
            0
        } else {
            captured_piece.see_value()
        };
        if moved_piece >= 0 && captured_piece != Piece::None {
            history_score = state.move_ordering.capture_history[moved_piece as usize]
                [move_obj.target as usize][captured_piece as usize]
                as i32;
        }
    }
    (history_score, captured_see_val)
}

#[inline(always)]
pub(super) fn root_filter(ply: u8, searchmoves: &[Move], m: Move) -> bool {
    ply == 0 && !searchmoves.is_empty() && !searchmoves.contains(&m)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn see_gate(
    board: &BoardState,
    m: Move,
    is_pv_node: bool,
    in_check: bool,
    depth: u8,
    has_legal: bool,
    has_excluded: bool,
    cap_or_promo: bool,
    has_npm: bool,
    history_score: i32,
) -> bool {
    if is_pv_node || in_check || depth > 8 || !has_legal || has_excluded {
        return false;
    }
    if cap_or_promo {
        let see_margin = 177 * depth as i32 + history_score * 34 / 1024;
        let thr = (-see_margin).clamp(-30000, 30000) as i16;
        !board.see_ge(m, thr)
    } else if has_npm {
        let ld = depth.saturating_sub(1) as i32;
        let see_margin = 23 * ld * ld;
        let thr = (-see_margin).clamp(-30000, 30000) as i16;
        !board.see_ge(m, thr)
    } else {
        false
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn futility_gate(
    is_pv_node: bool,
    has_excluded: bool,
    has_static_eval: bool,
    alpha_is_mate: bool,
    has_npm: bool,
    mate_ok: bool,
    cap_or_promo: bool,
    gives_check: bool,
    lmr_depth_fut: u8,
    static_eval: i16,
    alpha: i16,
    captured_see_val: i16,
    history_score: i32,
    mate_bound: i16,
    best_score: &mut i16,
) -> bool {
    if is_pv_node || has_excluded || !has_static_eval || alpha_is_mate || !has_npm || !mate_ok {
        return false;
    }
    if cap_or_promo {
        if !gives_check && lmr_depth_fut < 8 {
            let fut_val = static_eval as i32
                + 234
                + 247 * lmr_depth_fut as i32
                + captured_see_val as i32
                + history_score * 134 / 1024;
            if fut_val <= alpha as i32 {
                return true;
            }
        }
    } else if lmr_depth_fut < 12 && !gives_check {
        let fut_val = static_eval as i32
            + 119 * lmr_depth_fut as i32
            + 90 * ((static_eval > alpha) as i32)
            + 164;
        if fut_val <= alpha as i32 {
            if *best_score as i32 <= fut_val
                && fut_val.abs() < mate_bound as i32
                && fut_val.abs() < 30000
            {
                *best_score = fut_val as i16;
            }
            return true;
        }
    }
    false
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn lmp_gate(
    is_improving: bool,
    depth: u8,
    found_pv: bool,
    moves: usize,
    last_intent: crate::search::intent::SearchIntent,
    is_pv_node: bool,
    has_excluded: bool,
    has_static_eval: bool,
    cap_or_promo: bool,
    alpha_is_mate: bool,
    has_npm: bool,
    gives_check: bool,
) -> bool {
    let mut lmp_threshold = if is_improving {
        3 + (depth as usize * depth as usize)
    } else {
        (3 + (depth as usize * depth as usize)) / 2
    };
    if !found_pv && moves >= 8 {
        lmp_threshold = lmp_threshold.saturating_sub(2).max(4);
    }
    let lmp_adj = crate::search::intent::effects(last_intent).lmp_adjust;
    if lmp_adj > 0 {
        lmp_threshold = lmp_threshold.saturating_add(lmp_adj as usize);
    } else if lmp_adj < 0 {
        lmp_threshold = lmp_threshold.saturating_sub((-lmp_adj) as usize).max(3);
    }
    !is_pv_node
        && !has_excluded
        && has_static_eval
        && depth < 4
        && !cap_or_promo
        && !alpha_is_mate
        && has_npm
        && moves >= lmp_threshold
        && !gives_check
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn history_gate(
    is_pv_node: bool,
    has_excluded: bool,
    in_check: bool,
    found_pv: bool,
    cap_or_promo: bool,
    depth: u8,
    moves: usize,
    history_score: i32,
    gives_check: bool,
) -> bool {
    !is_pv_node
        && !has_excluded
        && !in_check
        && !found_pv
        && !cap_or_promo
        && depth <= 4
        && moves >= 12
        && history_score < 0
        && !gives_check
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn alp_gate(
    alp_enabled: bool,
    alp_threshold: u8,
    is_pv_node: bool,
    has_excluded: bool,
    in_check: bool,
    found_pv: bool,
    cap_or_promo: bool,
    depth: u8,
    moves: usize,
    gives_check: bool,
    static_eval: i16,
    alpha: i16,
    history_score: i32,
    momentum: i16,
) -> bool {
    if !alp_enabled || is_pv_node || has_excluded || in_check || found_pv || cap_or_promo {
        return false;
    }
    if depth > 4 || moves < 8 || gives_check {
        return false;
    }
    let features = alp::AlpFeatures {
        eval_margin: (static_eval as i32 - alpha as i32).clamp(-32768, 32767),
        depth: depth as i32,
        move_index: moves,
        is_null_move: false,
        is_capture: cap_or_promo,
        is_pv: is_pv_node,
        in_check,
        history_score,
        momentum: momentum as i32,
    };
    alp::AlpModel::should_prune(&features, alp_threshold)
}

#[inline(always)]
pub(super) fn cfss_gate(
    has_excluded: bool,
    coarse_failed_low: bool,
    moves: usize,
    is_tactical: bool,
    current_depth: u8,
) -> bool {
    !has_excluded
        && cfss::should_prune_coarse_quiet(coarse_failed_low, moves, is_tactical, current_depth)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn psm_gate(
    psm_enabled: bool,
    is_pv_node: bool,
    has_excluded: bool,
    in_check: bool,
    cap_or_promo: bool,
    found_pv: bool,
    ply: u8,
    gives_check: bool,
    psm_state: &crate::search::psm::PsmHiddenState,
    moves: usize,
    depth: u8,
    fail_lows: u8,
) -> bool {
    psm_enabled
        && !is_pv_node
        && !has_excluded
        && !in_check
        && !cap_or_promo
        && !found_pv
        && (ply as usize) < constants::MAX_PLY
        && !gives_check
        && psm::PsmEngine::should_prune_sibling(psm_state, moves, depth, fail_lows)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn gtp_gate(
    gtp_enabled: bool,
    is_pv_node: bool,
    has_excluded: bool,
    in_check: bool,
    cap_or_promo: bool,
    found_pv: bool,
    depth: u8,
    moves: usize,
    gives_check: bool,
    graph: &crate::search::gtp::GtpTreeGraph,
    idx: usize,
    threshold: u8,
) -> bool {
    gtp_enabled
        && !is_pv_node
        && !has_excluded
        && !in_check
        && !cap_or_promo
        && !found_pv
        && depth <= 4
        && moves >= 10
        && !gives_check
        && gtp::GtpModel::should_prune_subtree(graph, idx, threshold)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn losing_history_gate(
    depth: u8,
    moves: usize,
    is_pv_node: bool,
    has_excluded: bool,
    in_check: bool,
    cap_or_promo: bool,
    has_npm: bool,
    history_score: i32,
    gives_check: bool,
) -> bool {
    depth <= 3
        && moves > 3
        && !is_pv_node
        && !has_excluded
        && !in_check
        && !cap_or_promo
        && has_npm
        && history_score < -4000 * depth as i32
        && !gives_check
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn extension_depth(
    board: &mut BoardState,
    state: &mut SearchState,
    move_obj: Move,
    tt_best: Option<Move>,
    singular_extension: i8,
    current_depth: u8,
    cap_or_promo: bool,
    gives_check: bool,
    in_check: bool,
    ply: u8,
    previous_move: Option<Move>,
) -> (u8, i8) {
    let mut extension: i8 = if Some(move_obj) == tt_best {
        singular_extension
    } else {
        0
    };

    let is_tactical_move = cap_or_promo || gives_check;
    let tce_ext = if current_depth >= 6 && is_tactical_move && state.params.tce_enabled {
        let nt_child = NodeThreats::compute(board);
        tce::compute_extension(
            board,
            &nt_child,
            move_obj,
            current_depth,
            in_check,
            ply,
            previous_move,
        )
        .clamp(0, 1)
    } else {
        0
    };
    extension = extension.max(tce_ext).clamp(-3, 3);

    if state.verification_budget > 0 && is_tactical_move && extension < 3 {
        extension = (extension + 1).min(3);
    }

    let parent_streak = if ply as usize >= constants::MAX_PLY {
        2
    } else {
        state.extension_streak[ply as usize]
    };
    if state.params.extension_cap_enabled && extension > 0 && parent_streak >= 2 {
        extension = 0;
    }
    if (ply as usize) + 1 < constants::MAX_PLY {
        state.extension_streak[(ply as usize) + 1] = if extension > 0 {
            parent_streak.saturating_add(1)
        } else {
            0
        };
    }

    let depth = (current_depth as i16 + extension as i16).max(1) as u8;
    (depth, extension)
}

pub(super) enum MoveOut {
    Cancelled,
    Score(i16),
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn search_move(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    alpha: i16,
    beta: i16,
    found_pv: bool,
    move_obj: Move,
    cap_or_promo: bool,
    gives_check: bool,
    in_check: bool,
    is_pv_node: bool,
    is_improving: bool,
    has_non_pawn_material: bool,
    history_score: i32,
    static_eval: i16,
    momentum: i16,
    structural_disagreement: i16,
    best_score: i16,
    tt_best: Option<Move>,
    cut_node: bool,
    pv_move: Option<Move>,
    on_pv_path: bool,
    gtp_graph: gtp::GtpTreeGraph,
    gtp_idx: usize,
    number_of_legal_moves: usize,
    consecutive_fail_lows: u8,
    nt: &NodeThreats,
    ctx: &mut SearchContext,
) -> MoveOut {
    let mut score: i16;
    let is_tactical = cap_or_promo || gives_check;
    let next_on_pv = on_pv_path && Some(move_obj) == pv_move;
    let needs_lmr = lmr::needs_reduction(depth, number_of_legal_moves, is_tactical, in_check)
        || (is_tactical
            && lmr::needs_tactical_reduction(
                depth,
                number_of_legal_moves,
                in_check,
                history_score,
            ));
    let child_cut = !cut_node;

    if ctx.search_state.params.psm_enabled && (ply as usize + 1) < constants::MAX_PLY {
        let psm_feat = psm::PsmFeatures {
            static_eval,
            depth,
            alpha,
            beta,
            move_history: history_score,
            is_capture: cap_or_promo,
            sibling_index: number_of_legal_moves,
            failed_low: consecutive_fail_lows > 0,
        };
        ctx.search_state.psm_stack.stack[ply as usize + 1] =
            psm::PsmEngine::step(&ctx.search_state.psm_stack.stack[ply as usize], &psm_feat);
    }

    if needs_lmr {
        let all_node_lmr = !is_pv_node && !ctx.cut_node;
        let tt_capture_lmr = tt_best
            .map(|m| m.is_capture() || m.is_promotion())
            .unwrap_or(false);
        let is_tt_move_lmr = Some(move_obj) == tt_best;
        let cutoff_lmr = lmr::cutoff_count(ply.saturating_add(1));
        let lmr_query = lmr::LmrQuery {
            depth,
            move_count: number_of_legal_moves,
            is_pv_node,
            is_improving,
            gives_check,
            is_tactical: cap_or_promo,
            has_non_pawn_material,
            history_score,
            alpha,
            static_eval,
            momentum,
            found_pv,
            structural_disagreement,

            cut_node: ctx.cut_node,
            tt_pv: is_pv_node,
            cutoff_cnt: cutoff_lmr,
            all_node: all_node_lmr,
            tt_capture: tt_capture_lmr,
            is_tt_move: is_tt_move_lmr,
        };
        let base_reduction = lmr::compute_reduction(
            &lmr_query,
            &ctx.search_state.lmr_table,
            &ctx.search_state.params.lmr_divisor,
        );
        let ras_perturbation = if ctx.search_state.params.ras_enabled {
            ctx.search_state
                .ras
                .lmr_perturbation(ply as usize, depth, board_state.board_hash)
        } else {
            0
        };
        let mut reduction =
            (base_reduction as i8 + ras_perturbation).clamp(0, depth as i8 - 1) as u8;

        let lmr_adj = crate::search::intent::effects(ctx.search_state.last_intent).lmr_adjust;
        if lmr_adj < 0 && (ply <= 4 || is_tactical) {
            reduction = reduction.saturating_sub((-lmr_adj) as u8);
        } else if lmr_adj > 0 && !is_tactical {
            reduction = reduction
                .saturating_add(lmr_adj as u8)
                .min(depth.saturating_sub(1));
        } else if ctx.search_state.params.cpi_enabled && nt.checkers.count_ones() > 0 {
            reduction = reduction.saturating_sub(1);
        }
        score = -search_internal(
            board_state,
            depth.saturating_sub(1 + reduction),
            ply + 1,
            -alpha - 1,
            -alpha,
            Some(move_obj),
            &mut SearchContext {
                allow_null_move: true,
                on_pv_path: false,
                previous_pv: ctx.previous_pv,
                excluded_move: None,
                cut_node: true,
                gtp_graph,
                gtp_parent: Some(gtp_idx),
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            },
        );

        if is_cancelled(ctx) {
            board_state.unmake_move(move_obj);
            return MoveOut::Cancelled;
        }

        if score > alpha {
            let reduced_for_deeper = depth.saturating_sub(1 + reduction);
            let full_for_deeper = depth.saturating_sub(1).max(1);
            let adj =
                lmr::deepen_adjustment(reduced_for_deeper, full_for_deeper, score, best_score);
            let mut deeper_depth = depth;
            if adj != 0 {
                deeper_depth = (depth as i16 + adj as i16).clamp(1, 64) as u8;
            }
            let mut child_ctx = SearchContext {
                allow_null_move: true,
                on_pv_path: next_on_pv,
                previous_pv: ctx.previous_pv,
                excluded_move: None,
                cut_node: if next_on_pv { false } else { child_cut },
                gtp_graph,
                gtp_parent: Some(gtp_idx),
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            };
            score = search_deeper(
                board_state,
                deeper_depth,
                ply,
                alpha,
                beta,
                found_pv,
                Some(move_obj),
                &mut child_ctx,
            );
        }
    } else {
        let mut child_ctx = SearchContext {
            allow_null_move: true,
            on_pv_path: next_on_pv,
            previous_pv: ctx.previous_pv,
            excluded_move: None,
            cut_node: if next_on_pv { false } else { child_cut },
            gtp_graph,
            gtp_parent: Some(gtp_idx),
            pv_table: &mut *ctx.pv_table,
            cancellation_token: ctx.cancellation_token,
            search_state: ctx.search_state,
        };
        score = search_deeper(
            board_state,
            depth,
            ply,
            alpha,
            beta,
            found_pv,
            Some(move_obj),
            &mut child_ctx,
        );
    }
    MoveOut::Score(score)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_node(
    board_state: &mut BoardState,
    ply: u8,
    halfmove: u8,
    mate_bound: i16,
    is_pv_node: bool,
    in_check: bool,
    static_eval: i16,
    has_static_eval: bool,
    current_depth: u8,
    best_score: i16,
    best_move: Move,
    entry_type: TranspositionEntryType,
    previous_move: Option<Move>,
    tried_quiets: &[Move],
    number_of_legal_moves: usize,
    has_legal_moves: bool,
    bandit_arm: crate::search::bmo::BanditArm,
    ctx: &mut SearchContext,
) -> i16 {
    if is_cancelled(ctx) {
        return 0;
    }

    if !is_pv_node
        && ctx.excluded_move.is_none()
        && number_of_legal_moves >= 3
        && ctx.search_state.params.bmo_enabled
    {
        ctx.search_state
            .bmo
            .update(current_depth, bandit_arm, false);
    }

    if !has_legal_moves {
        if in_check {
            return -constants::MAX_CENTIPAWN_EVAL + ply as i16;
        }
        let base = draw::draw_score(ctx.search_state.nodes);
        return draw::apply_contempt(
            base,
            board_state.side_to_move,
            ctx.search_state.engine_side,
            ctx.search_state.contempt_cp,
            ctx.search_state.draw_score_cp,
        );
    }

    if !is_cancelled(ctx) {
        if ctx.excluded_move.is_none() && ctx.search_state.tt_store_allowed(ply) {
            ctx.search_state.tt.submit_entry(
                board_state.board_hash,
                tt::TranspositionTable::adjust_score(best_score, ply as i32, halfmove),
                current_depth,
                best_move,
                entry_type,
            );
        }

        if ctx.excluded_move.is_none()
            && has_static_eval
            && !in_check
            && !best_move.is_capture()
            && best_score.abs() < mate_bound
            && (best_score > static_eval) == (best_move != Move::NO_MOVE)
        {
            let move_factor = if best_move != Move::NO_MOVE { 12 } else { 18 };
            let bonus =
                (((best_score as i32 - static_eval as i32) * current_depth as i32 * move_factor)
                    / 128)
                    .clamp(-256, 256);
            let final_bonus = bonus * 1061 / 1024;
            ctx.search_state
                .correction_history
                .update(board_state, previous_move, final_bonus);
        }

        if ctx.excluded_move.is_none()
            && entry_type == TranspositionEntryType::Exact
            && best_move != Move::NO_MOVE
            && !best_move.is_capture()
        {
            update_history_stats(
                board_state,
                ctx.search_state,
                best_move,
                current_depth,
                previous_move,
                tried_quiets,
            );
        }
    }

    best_score
}
