use super::context::{SearchContext, is_cancelled};
use super::early::{
    apply_iir, coarse_pass, mate_window, null_move_search, probcut_search, razor_gate, rfp_gate,
    singular_search, static_info, tablebase_probe, tt_cutoff,
};
use super::history::beta_cutoff;
use super::moves::{
    MoveOut, alp_gate, cfss_gate, extension_depth, finish_node, futility_gate, gtp_gate,
    history_gate, lmp_gate, losing_history_gate, move_scores, psm_gate, root_filter, search_move,
    see_gate,
};
use super::*;

#[allow(clippy::too_many_arguments)]
pub fn search(
    board_state: &mut BoardState,
    depth: u8,
    alpha: i16,
    beta: i16,
    cancellation_token: &AtomicBool,
    previous_pv: &[Move],
    pv_table: &mut PvTable,
    search_state: &mut SearchState,
) -> i16 {
    lmr::clear_cutoff_counts();
    nmp::clear_nmp_state();
    let mut ctx = SearchContext {
        allow_null_move: true,
        on_pv_path: true,
        previous_pv,
        excluded_move: None,
        cut_node: false,
        gtp_graph: gtp::GtpTreeGraph::new(),
        gtp_parent: None,
        pv_table,
        cancellation_token,
        search_state,
    };

    search_internal(board_state, depth, 0, alpha, beta, None, &mut ctx)
}

pub(super) fn search_internal(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    mut alpha: i16,
    beta: i16,
    previous_move: Option<Move>,
    ctx: &mut SearchContext,
) -> i16 {
    if is_cancelled(ctx) {
        return 0;
    }
    lmr::reset_cutoff(ply.saturating_add(2));
    nmp::reset_prior_fail_high(ply.saturating_add(1));

    let is_pv_node = beta > 1 + alpha;

    let cut_node = ctx.cut_node;
    let halfmove = board_state.half_move_clock;

    if is_pv_node {
        ctx.pv_table.clear(ply as usize);
    }

    ctx.search_state.nodes += 1;
    if ctx.search_state.nodes & 1023 == 0
        && ctx.search_state.max_nodes > 0
        && ctx.search_state.nodes >= ctx.search_state.max_nodes
    {
        ctx.cancellation_token.store(true, Ordering::Relaxed);
        return 0;
    }
    if is_pv_node {
        let sd = ply as u32 + 1;
        if sd > ctx.search_state.seldepth {
            ctx.search_state.seldepth = sd;
        }
    }

    let mut best_move = Move::NO_MOVE;
    let mut best_score = -constants::MAX_CENTIPAWN_EVAL;

    if ply > 0 && ghi::is_draw(board_state, ply as u16) {
        ctx.search_state.rep_draw_ply = ctx.search_state.rep_draw_ply.min(ply);
        let base = draw::draw_score(ctx.search_state.nodes);
        return draw::apply_contempt(
            base,
            board_state.side_to_move,
            ctx.search_state.engine_side,
            ctx.search_state.contempt_cp,
            ctx.search_state.draw_score_cp,
        );
    }

    let (mated_alpha, mated_beta, mate_exit) = mate_window(ply, alpha, beta);
    alpha = mated_alpha;
    let beta = mated_beta;
    if let Some(score) = mate_exit {
        return score;
    }

    let depth = depth.min(constants::MAX_PLY as u8 - 1);

    if ply as usize >= constants::MAX_PLY {
        return quiescence::search(
            board_state,
            alpha,
            beta,
            ply,
            ctx.cancellation_token,
            ctx.search_state,
        );
    }

    let tt_entry = ctx.search_state.tt.probe(board_state.board_hash);
    let mut tt_best = None;

    if let Some(entry) = tt_entry {
        if entry.best_move != Move::NO_MOVE {
            tt_best = Some(entry.best_move);
        }

        if let Some(score) = tt_cutoff(
            entry,
            is_pv_node,
            ctx.excluded_move.is_some(),
            depth,
            alpha,
            beta,
            cut_node,
            ply,
            halfmove,
        ) {
            return score;
        }
    }

    let tb = tablebase_probe(
        board_state,
        ply,
        depth,
        alpha,
        beta,
        best_score,
        is_pv_node,
        halfmove,
        ctx,
    );
    alpha = tb.alpha;
    best_score = tb.best;
    if let Some(score) = tb.score {
        return score;
    }

    if depth == 0 {
        return quiescence::search(
            board_state,
            alpha,
            beta,
            ply,
            ctx.cancellation_token,
            ctx.search_state,
        );
    }

    let mate_bound = constants::MAX_CENTIPAWN_EVAL - constants::MAX_PLY as i16;
    let beta_is_mate = beta.abs() >= mate_bound;

    let (nt, st) = static_info(
        board_state,
        depth,
        ply,
        beta,
        tt_entry,
        halfmove,
        previous_move,
        mate_bound,
        ctx,
    );
    let in_check = st.in_check;
    let has_static_eval = st.has_eval;
    let static_eval = st.static_eval;
    let momentum = st.momentum;
    let structural_disagreement = st.disagreement;
    let is_improving = st.is_improving;

    let sing = singular_search(
        board_state,
        depth,
        ply,
        beta,
        static_eval,
        has_static_eval,
        in_check,
        is_pv_node,
        tt_entry,
        tt_best,
        cut_node,
        previous_move,
        halfmove,
        mate_bound,
        ctx,
    );
    if let Some(score) = sing.score {
        return score;
    }
    let singular_extension = sing.extension;
    let singular_depth_bonus = sing.depth_bonus;

    if let Some(score) = razor_gate(
        board_state,
        alpha,
        beta,
        ply,
        is_pv_node,
        in_check,
        depth,
        static_eval,
        has_static_eval,
        beta_is_mate,
        ctx,
    ) {
        return score;
    }

    if !is_pv_node && has_static_eval && ctx.excluded_move.is_none() {
        if let Some(score) = rfp_gate(
            &nt,
            static_eval,
            beta,
            depth,
            momentum,
            is_improving,
            is_pv_node,
            has_static_eval,
            ctx.excluded_move.is_some(),
            ply,
            tt_entry,
            halfmove,
            mate_bound,
            beta_is_mate,
            ctx,
        ) {
            return score;
        }

        if let Some(score) = probcut_search(
            board_state,
            depth,
            ply,
            beta,
            static_eval,
            is_improving,
            beta_is_mate,
            in_check,
            halfmove,
            mate_bound,
            &nt,
            ctx,
        ) {
            return score;
        }
    }

    if let Some(score) = null_move_search(
        board_state,
        depth,
        ply,
        beta,
        static_eval,
        momentum,
        is_improving,
        is_pv_node,
        in_check,
        cut_node,
        previous_move,
        halfmove,
        mate_bound,
        ctx,
    ) {
        return score;
    }

    let pv_move = if ctx.on_pv_path && (ply as usize) < ctx.previous_pv.len() {
        Some(ctx.previous_pv[ply as usize])
    } else {
        None
    };

    let mut found_pv = false;
    let mut entry_type = TranspositionEntryType::Alpha;
    let mut current_depth = depth
        .saturating_add(singular_depth_bonus)
        .min(constants::MAX_PLY as u8 - 1);

    current_depth = apply_iir(
        current_depth,
        is_pv_node,
        cut_node,
        tt_best.is_some(),
        ctx.excluded_move.is_some(),
        ctx.search_state.params.iir_enabled,
    );

    let coarse = coarse_pass(
        board_state,
        current_depth,
        ply,
        alpha,
        beta,
        is_pv_node,
        in_check,
        tt_best,
        previous_move,
        ctx,
    );
    if coarse.cancelled {
        return 0;
    }
    current_depth = coarse.depth;
    let coarse_failed_low = coarse.failed_low;
    tt_best = coarse.tt_best;

    let bandit_arm = if !is_pv_node && ctx.search_state.params.bmo_enabled {
        ctx.search_state.bmo.select_arm(current_depth)
    } else {
        crate::search::bmo::BanditArm::CapturesFirst
    };

    let mut move_picker = MovePicker::new(
        pv_move,
        tt_best,
        previous_move,
        ply as usize,
        ctx.excluded_move,
        bandit_arm,
    );
    let mut number_of_legal_moves = 0;
    let mut has_legal_moves = false;
    let mut tried_quiets = [Move::NO_MOVE; 64];
    let mut tried_quiets_count = 0;
    let mut tried_captures = [Move::NO_MOVE; 32];
    let mut tried_captures_count = 0;
    let has_non_pawn_material = board_state.has_non_pawn_material(board_state.side_to_move);
    let mut consecutive_fail_lows: u8 = 0;

    while let Some(move_obj) = move_picker.next(
        board_state,
        &ctx.search_state.move_ordering,
        &mut ctx.search_state.captures_stack[ply as usize],
        &mut ctx.search_state.quiets_stack[ply as usize],
        &nt,
    ) {
        if is_cancelled(ctx) {
            break;
        }

        if root_filter(ply, &ctx.search_state.searchmoves, move_obj) {
            continue;
        }

        let cap_or_promo = move_obj.is_capture() || move_obj.is_promotion();
        let (history_score, captured_see_val) = move_scores(
            board_state,
            ctx.search_state,
            move_obj,
            cap_or_promo,
            previous_move,
        );

        if see_gate(
            board_state,
            move_obj,
            is_pv_node,
            in_check,
            current_depth,
            has_legal_moves,
            ctx.excluded_move.is_some(),
            cap_or_promo,
            has_non_pawn_material,
            history_score,
        ) {
            continue;
        }

        if !board_state.is_legal_with(move_obj, nt.checkers, nt.pinned) {
            continue;
        }

        board_state.make_move(move_obj);
        ctx.search_state.tt.prefetch(board_state.board_hash);

        has_legal_moves = true;

        number_of_legal_moves += 1;

        let alpha_is_mate = alpha.abs() >= mate_bound;

        let gives_check = board_state.is_in_check(board_state.side_to_move);

        let (depth, extension) = extension_depth(
            board_state,
            ctx.search_state,
            move_obj,
            tt_best,
            singular_extension,
            current_depth,
            cap_or_promo,
            gives_check,
            in_check,
            ply,
            previous_move,
        );
        let excluded_here = ctx.excluded_move.is_some();

        let lmr_depth_fut = depth.saturating_sub(1);
        if futility_gate(
            is_pv_node,
            excluded_here,
            has_static_eval,
            alpha_is_mate,
            has_non_pawn_material,
            best_score.abs() < mate_bound,
            cap_or_promo,
            gives_check,
            lmr_depth_fut,
            static_eval,
            alpha,
            captured_see_val,
            history_score,
            mate_bound,
            &mut best_score,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        if lmp_gate(
            is_improving,
            depth,
            found_pv,
            number_of_legal_moves,
            ctx.search_state.last_intent,
            is_pv_node,
            excluded_here,
            has_static_eval,
            cap_or_promo,
            alpha_is_mate,
            has_non_pawn_material,
            gives_check,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        if history_gate(
            is_pv_node,
            excluded_here,
            in_check,
            found_pv,
            cap_or_promo,
            depth,
            number_of_legal_moves,
            history_score,
            gives_check,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        if alp_gate(
            ctx.search_state.params.alp_enabled,
            ctx.search_state.params.alp_threshold,
            is_pv_node,
            excluded_here,
            in_check,
            found_pv,
            cap_or_promo,
            depth,
            number_of_legal_moves,
            gives_check,
            static_eval,
            alpha,
            history_score,
            momentum,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        let is_tactical = cap_or_promo || gives_check;

        if cfss_gate(
            excluded_here,
            coarse_failed_low,
            number_of_legal_moves,
            is_tactical,
            current_depth,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        if psm_gate(
            ctx.search_state.params.psm_enabled,
            is_pv_node,
            excluded_here,
            in_check,
            cap_or_promo,
            found_pv,
            ply,
            gives_check,
            &ctx.search_state.psm_stack.stack[ply as usize],
            number_of_legal_moves,
            depth,
            consecutive_fail_lows,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        let gtp_node = gtp::GtpNode {
            depth,
            eval_margin: (static_eval as i32 - alpha as i32).clamp(-32768, 32767) as i16,
            is_capture: cap_or_promo,
            in_check,
            history_score,
            parent_idx: ctx.gtp_parent,
        };
        let mut gtp_graph = ctx.gtp_graph;
        let gtp_idx = gtp_graph.add_node(gtp_node);

        if gtp_gate(
            ctx.search_state.params.gtp_enabled,
            is_pv_node,
            excluded_here,
            in_check,
            cap_or_promo,
            found_pv,
            depth,
            number_of_legal_moves,
            gives_check,
            &gtp_graph,
            gtp_idx,
            ctx.search_state.params.gtp_threshold,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        if losing_history_gate(
            depth,
            number_of_legal_moves,
            is_pv_node,
            excluded_here,
            in_check,
            cap_or_promo,
            has_non_pawn_material,
            history_score,
            gives_check,
        ) {
            board_state.unmake_move(move_obj);
            continue;
        }

        let move_nodes_start = ctx.search_state.nodes;
        let score = match search_move(
            board_state,
            depth,
            ply,
            alpha,
            beta,
            found_pv,
            move_obj,
            cap_or_promo,
            gives_check,
            in_check,
            is_pv_node,
            is_improving,
            has_non_pawn_material,
            history_score,
            static_eval,
            momentum,
            structural_disagreement,
            best_score,
            tt_best,
            cut_node,
            pv_move,
            ctx.on_pv_path,
            gtp_graph,
            gtp_idx,
            number_of_legal_moves,
            consecutive_fail_lows,
            &nt,
            ctx,
        ) {
            MoveOut::Cancelled => return 0,
            MoveOut::Score(s) => s,
        };

        board_state.unmake_move(move_obj);

        if is_cancelled(ctx) {
            return 0;
        }

        if score <= alpha {
            consecutive_fail_lows = consecutive_fail_lows.saturating_add(1);
        } else {
            consecutive_fail_lows = 0;
        }

        if score > best_score {
            best_score = score;
            best_move = move_obj;

            if score > alpha {
                alpha = score;
                entry_type = TranspositionEntryType::Exact;
                found_pv = true;

                if is_pv_node {
                    ctx.pv_table.update(ply as usize, move_obj);
                }
                if ply == 0 {
                    ctx.search_state.root_best_move_nodes =
                        ctx.search_state.nodes.saturating_sub(move_nodes_start);
                    if number_of_legal_moves > 1 {
                        ctx.search_state.best_move_changes += 1;
                    }
                }
            }
        }

        if score >= beta {
            if !is_pv_node && ctx.excluded_move.is_none() && ctx.search_state.params.bmo_enabled {
                let early_cutoff = number_of_legal_moves <= 2;
                ctx.search_state
                    .bmo
                    .update(current_depth, bandit_arm, early_cutoff);
            }
            if extension < 2 || is_pv_node {
                lmr::record_cutoff(ply);
            }

            if is_cancelled(ctx) {
                return score;
            }
            return beta_cutoff(
                score,
                move_obj,
                ply as usize,
                board_state,
                current_depth,
                previous_move,
                ctx.search_state,
                ctx.cancellation_token,
                &tried_quiets[..tried_quiets_count],
                &tried_captures[..tried_captures_count],
                ctx.excluded_move,
            );
        }

        if !move_obj.is_capture() && tried_quiets_count < tried_quiets.len() {
            tried_quiets[tried_quiets_count] = move_obj;
            tried_quiets_count += 1;
        } else if move_obj.is_capture() && tried_captures_count < tried_captures.len() {
            tried_captures[tried_captures_count] = move_obj;
            tried_captures_count += 1;
        }
    }

    finish_node(
        board_state,
        ply,
        halfmove,
        mate_bound,
        is_pv_node,
        in_check,
        static_eval,
        has_static_eval,
        current_depth,
        best_score,
        best_move,
        entry_type,
        previous_move,
        &tried_quiets[..tried_quiets_count],
        number_of_legal_moves,
        has_legal_moves,
        bandit_arm,
        ctx,
    )
}
