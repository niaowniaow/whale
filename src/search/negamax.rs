use crate::board::node_threats::NodeThreats;
use crate::board::state::BoardState;
use crate::common::castle::Castle;
use crate::common::constants::{self, MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::tt::{self, TranspositionEntryType};
use crate::search::move_picker::MovePicker;
use crate::search::pv_table::PvTable;
use crate::search::search_state::SearchState;
use crate::search::{
    alp, cfss, counterplay, draw, ghi, gtp, iir, lmr, nmp, psm, quiescence, razoring, tce,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[inline(always)]
fn is_cancelled(ctx: &SearchContext) -> bool {
    ctx.cancellation_token.load(Ordering::Relaxed)
}

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

fn search_internal(
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
    if is_pv_node {
        let sd = ply as u32 + 1;
        if sd > ctx.search_state.seldepth {
            ctx.search_state.seldepth = sd;
        }
    }

    let mut best_move = Move::NO_MOVE;
    let mut best_score = -constants::MAX_CENTIPAWN_EVAL;

    if ply > 0 && ghi::is_draw(board_state, ply as u16) {
        let base = draw::draw_score(ctx.search_state.nodes);
        return draw::apply_contempt(
            base,
            board_state.side_to_move,
            ctx.search_state.engine_side,
            ctx.search_state.contempt_cp,
            ctx.search_state.draw_score_cp,
        );
    }

    let mated = -constants::MAX_CENTIPAWN_EVAL + ply as i16;
    let mating = constants::MAX_CENTIPAWN_EVAL - ply as i16;
    if alpha < mated {
        alpha = mated;
    }
    let mut beta = beta;
    if beta > mating {
        beta = mating;
    }
    if alpha >= beta {
        return alpha;
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

        if !is_pv_node && ctx.excluded_move.is_none() {
            let tt_score =
                tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
            let depth_cond = (entry.depth as i32) > depth as i32 - i32::from(tt_score < beta);
            let bound_ok = match entry.entry_type {
                TranspositionEntryType::Exact => true,
                TranspositionEntryType::Alpha => tt_score <= alpha && (!cut_node || depth > 5),
                TranspositionEntryType::Beta => tt_score >= beta && (cut_node || depth > 5),
                TranspositionEntryType::None => false,
            };
            if depth_cond && bound_ok && draw::allow_tt_cutoff(halfmove) {
                return tt_score;
            }
        }
    }

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
                return tb_value;
            }

            if is_pv_node && tb_bound == TranspositionEntryType::Beta && tb_value > best_score {
                best_score = tb_value;
                if tb_value > alpha {
                    alpha = tb_value;
                    if alpha >= beta {
                        return tb_value;
                    }
                }
            }
        }
        if is_cancelled(ctx) {
            return 0;
        }
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

    let nt = NodeThreats::compute(board_state);
    let in_check = nt.in_check();

    let mut static_eval = 0;
    let has_static_eval = !in_check;

    let mate_bound = constants::MAX_CENTIPAWN_EVAL - constants::MAX_PLY as i16;
    let beta_is_mate = beta.abs() >= mate_bound;

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

    let mut singular_extension: i8 = 0;
    let mut singular_depth_bonus: u8 = 0;
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
                    return 0;
                }

                if se_score < singular_beta {
                    let double_margin = -2 + 204 * (is_pv_node as i16)
                        - 152 * ((!tt_capture_sing) as i16)
                        - corr_adj;
                    let triple_margin = 70 + 279 * (is_pv_node as i16)
                        - 188 * ((!tt_capture_sing) as i16)
                        + 81 * (tt_pv_sing as i16)
                        - corr_adj;
                    singular_extension = 1
                        + (se_score < singular_beta - double_margin) as i8
                        + (se_score < singular_beta - triple_margin) as i8;
                    singular_depth_bonus = 1;
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
                    return se_score;
                } else if original_score >= beta || cut_node {
                    singular_extension = -3;
                }
            }
        }
    }

    let mut is_improving = !in_check
        && (ply as usize) >= 2
        && ctx.search_state.eval_stack[ply as usize - 2] != i16::MIN
        && static_eval > ctx.search_state.eval_stack[ply as usize - 2];
    if !in_check && static_eval >= beta {
        is_improving = true;
    }

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
        return quiescence::search(
            board_state,
            alpha,
            beta,
            ply,
            ctx.cancellation_token,
            ctx.search_state,
        );
    }

    if !is_pv_node && has_static_eval && ctx.excluded_move.is_none() {
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
            margin += counterplay::cpi_rfp_adjust(counterplay::compute_cpi_fast(&nt).cpi);
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
            return ((661 * beta as i32 + 363 * static_eval as i32) / 1024) as i16;
        }

        let small_prob_beta = beta.saturating_add(380);
        if let Some(entry) = tt_entry
            && entry.entry_type != tt::TranspositionEntryType::Alpha
            && entry.depth >= depth.saturating_sub(4)
        {
            let tt_val = tt::TranspositionTable::retrieve_score(entry.score, ply as i32, halfmove);
            if tt_val >= small_prob_beta && tt_val.abs() < mate_bound {
                return small_prob_beta;
            }
        }

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
                        return 0;
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
                        return 0;
                    }
                    if score >= prob_beta {
                        let returned_score = if score.abs() < constants::MAX_CENTIPAWN_EVAL - 200 {
                            score - (prob_beta - beta)
                        } else {
                            score
                        };
                        if ctx.excluded_move.is_none() {
                            ctx.search_state.tt.submit_entry(
                                board_state.board_hash,
                                tt::TranspositionTable::adjust_score(score, ply as i32, halfmove),
                                prob_depth,
                                mv,
                                TranspositionEntryType::Beta,
                            );
                        }
                        return returned_score;
                    }
                }
            }
        }
    }

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
            return 0;
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
                if ctx.excluded_move.is_none() {
                    ctx.search_state.tt.submit_entry(
                        board_state.board_hash,
                        tt::TranspositionTable::adjust_score(score_out, ply as i32, halfmove),
                        reduced_depth,
                        Move::NO_MOVE,
                        TranspositionEntryType::Beta,
                    );
                }
                return score_out;
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
                return 0;
            }
            if verify_score >= beta {
                nmp::inc_prior_fail_high(ply);
                let mate_bound_inner = MAX_CENTIPAWN_EVAL - MAX_PLY as i16;
                let score_out = if score >= mate_bound_inner {
                    beta
                } else {
                    score
                };
                if ctx.excluded_move.is_none() {
                    ctx.search_state.tt.submit_entry(
                        board_state.board_hash,
                        tt::TranspositionTable::adjust_score(score_out, ply as i32, halfmove),
                        reduced_depth,
                        Move::NO_MOVE,
                        TranspositionEntryType::Beta,
                    );
                }
                return score_out;
            }
        }
    }

    let pv_move = if ctx.on_pv_path && (ply as usize) < ctx.previous_pv.len() {
        Some(ctx.previous_pv[ply as usize])
    } else {
        None
    };

    let mut found_pv = false;
    let mut entry_type = TranspositionEntryType::Alpha;
    let mut coarse_failed_low = false;
    let mut current_depth = depth
        .saturating_add(singular_depth_bonus)
        .min(constants::MAX_PLY as u8 - 1);

    if ctx.search_state.params.iir_enabled {
        let iir_red = iir::reduction(
            is_pv_node,
            cut_node,
            current_depth,
            tt_best.is_some(),
            ctx.excluded_move.is_some(),
        );
        current_depth = current_depth.saturating_sub(iir_red).max(1);
    }

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
            return 0;
        }

        if let Some(entry) = ctx.search_state.tt.probe(board_state.board_hash)
            && entry.best_move != Move::NO_MOVE
        {
            tt_best = Some(entry.best_move);
        }

        if coarse_score <= alpha.saturating_sub(250) {
            coarse_failed_low = true;
            current_depth = current_depth.saturating_sub(1).max(1);
        }
    }

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

        if ply == 0
            && !ctx.search_state.searchmoves.is_empty()
            && !ctx.search_state.searchmoves.contains(&move_obj)
        {
            continue;
        }

        let cap_or_promo = move_obj.is_capture() || move_obj.is_promotion();
        let mut history_score = 0;
        let mut captured_see_val: i16 = 0;
        if !cap_or_promo {
            history_score = ctx.search_state.move_ordering.get_quiet_history_score(
                board_state,
                move_obj,
                previous_move,
            );
        } else if move_obj.is_capture() {
            let moved_piece = board_state.get_piece_on(move_obj.source);
            let captured_piece = if move_obj.move_type == MoveType::EnPassant {
                Piece::Pawn
            } else {
                board_state.piece_mapping[move_obj.target as usize]
            };
            captured_see_val = if captured_piece == Piece::None {
                0
            } else {
                captured_piece.see_value()
            };
            if moved_piece >= 0 && captured_piece != Piece::None {
                history_score = ctx.search_state.move_ordering.capture_history[moved_piece as usize]
                    [move_obj.target as usize][captured_piece as usize]
                    as i32;
            }
        }

        if !is_pv_node
            && !in_check
            && current_depth <= 8
            && has_legal_moves
            && ctx.excluded_move.is_none()
        {
            if cap_or_promo {
                let see_margin = 177 * current_depth as i32 + history_score * 34 / 1024;
                let thr = (-see_margin).clamp(-30000, 30000) as i16;
                if !board_state.see_ge(move_obj, thr) {
                    continue;
                }
            } else if has_non_pawn_material {
                let ld = current_depth.saturating_sub(1) as i32;
                let see_margin = 23 * ld * ld;
                let thr = (-see_margin).clamp(-30000, 30000) as i16;
                if !board_state.see_ge(move_obj, thr) {
                    continue;
                }
            }
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

        let mut extension: i8 = if Some(move_obj) == tt_best {
            singular_extension
        } else {
            0
        };

        let is_tactical_move = cap_or_promo || gives_check;
        let tce_ext =
            if current_depth >= 6 && is_tactical_move && ctx.search_state.params.tce_enabled {
                let nt_child = NodeThreats::compute(board_state);
                tce::compute_extension(
                    board_state,
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

        if ctx.search_state.verification_budget > 0 && is_tactical_move && extension < 3 {
            extension = (extension + 1).min(3);
        }

        let parent_streak = if ply as usize >= constants::MAX_PLY {
            2
        } else {
            ctx.search_state.extension_streak[ply as usize]
        };
        if ctx.search_state.params.extension_cap_enabled && extension > 0 && parent_streak >= 2 {
            extension = 0;
        }
        if (ply as usize) + 1 < constants::MAX_PLY {
            ctx.search_state.extension_streak[(ply as usize) + 1] = if extension > 0 {
                parent_streak.saturating_add(1)
            } else {
                0
            };
        }

        let depth = (current_depth as i16 + extension as i16).max(1) as u8;
        let excluded_here = ctx.excluded_move.is_some();

        let lmr_depth_fut = depth.saturating_sub(1);
        if !is_pv_node
            && !excluded_here
            && has_static_eval
            && !alpha_is_mate
            && has_non_pawn_material
            && best_score.abs() < mate_bound
        {
            if cap_or_promo {
                if !gives_check && lmr_depth_fut < 8 {
                    let fut_val = static_eval as i32
                        + 234
                        + 247 * lmr_depth_fut as i32
                        + captured_see_val as i32
                        + history_score * 134 / 1024;
                    if fut_val <= alpha as i32 {
                        board_state.unmake_move(move_obj);
                        continue;
                    }
                }
            } else if lmr_depth_fut < 12 && !gives_check {
                let fut_val = static_eval as i32
                    + 119 * lmr_depth_fut as i32
                    + 90 * ((static_eval > alpha) as i32)
                    + 164;
                if fut_val <= alpha as i32 {
                    if best_score as i32 <= fut_val
                        && fut_val.abs() < mate_bound as i32
                        && fut_val.abs() < 30000
                    {
                        best_score = fut_val as i16;
                    }
                    board_state.unmake_move(move_obj);
                    continue;
                }
            }
        }

        let mut lmp_threshold = if is_improving {
            3 + (depth as usize * depth as usize)
        } else {
            (3 + (depth as usize * depth as usize)) / 2
        };
        if !found_pv && number_of_legal_moves >= 8 {
            lmp_threshold = lmp_threshold.saturating_sub(2).max(4);
        }
        let lmp_adj = crate::search::intent::effects(ctx.search_state.last_intent).lmp_adjust;
        if lmp_adj > 0 {
            lmp_threshold = lmp_threshold.saturating_add(lmp_adj as usize);
        } else if lmp_adj < 0 {
            lmp_threshold = lmp_threshold.saturating_sub((-lmp_adj) as usize).max(3);
        }
        if !is_pv_node
            && !excluded_here
            && has_static_eval
            && depth < 4
            && !cap_or_promo
            && !alpha_is_mate
            && has_non_pawn_material
            && number_of_legal_moves >= lmp_threshold
            && !gives_check
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        if !is_pv_node
            && !excluded_here
            && !in_check
            && !found_pv
            && !cap_or_promo
            && depth <= 4
            && number_of_legal_moves >= 12
            && history_score < 0
            && !gives_check
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        let alp_features = alp::AlpFeatures {
            eval_margin: (static_eval as i32 - alpha as i32).clamp(-32768, 32767),
            depth: depth as i32,
            move_index: number_of_legal_moves,
            is_null_move: false,
            is_capture: cap_or_promo,
            is_pv: is_pv_node,
            in_check,
            history_score,
            momentum: momentum as i32,
        };

        if ctx.search_state.params.alp_enabled
            && !is_pv_node
            && !excluded_here
            && !in_check
            && !found_pv
            && !cap_or_promo
            && depth <= 4
            && number_of_legal_moves >= 8
            && alp::AlpModel::should_prune(&alp_features, ctx.search_state.params.alp_threshold)
            && !gives_check
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        let is_tactical = cap_or_promo || gives_check;

        if !excluded_here
            && cfss::should_prune_coarse_quiet(
                coarse_failed_low,
                number_of_legal_moves,
                is_tactical,
                current_depth,
            )
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        if ctx.search_state.params.psm_enabled
            && !is_pv_node
            && !excluded_here
            && !in_check
            && !cap_or_promo
            && !found_pv
            && (ply as usize) < constants::MAX_PLY
            && !gives_check
            && psm::PsmEngine::should_prune_sibling(
                &ctx.search_state.psm_stack.stack[ply as usize],
                number_of_legal_moves,
                depth,
                consecutive_fail_lows,
            )
        {
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

        if ctx.search_state.params.gtp_enabled
            && !is_pv_node
            && !excluded_here
            && !in_check
            && !cap_or_promo
            && !found_pv
            && depth <= 4
            && number_of_legal_moves >= 10
            && !gives_check
            && gtp::GtpModel::should_prune_subtree(
                &gtp_graph,
                gtp_idx,
                ctx.search_state.params.gtp_threshold,
            )
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        let needs_lmr = lmr::needs_reduction(depth, number_of_legal_moves, is_tactical, in_check)
            || (is_tactical
                && lmr::needs_tactical_reduction(
                    depth,
                    number_of_legal_moves,
                    in_check,
                    history_score,
                ));

        if depth <= 3
            && number_of_legal_moves > 3
            && !is_pv_node
            && !excluded_here
            && !in_check
            && !cap_or_promo
            && has_non_pawn_material
            && history_score < -4000 * depth as i32
            && !gives_check
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        let mut score;
        let next_on_pv = ctx.on_pv_path && Some(move_obj) == pv_move;
        let move_nodes_start = ctx.search_state.nodes;

        let child_cut = !cut_node;

        if (ply as usize + 1) < constants::MAX_PLY {
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
                return 0;
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
            if !is_pv_node && ctx.excluded_move.is_none() {
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

    if is_cancelled(ctx) {
        return 0;
    }

    if !is_pv_node && ctx.excluded_move.is_none() && number_of_legal_moves >= 3 {
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
        if ctx.excluded_move.is_none() {
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
                &tried_quiets[..tried_quiets_count],
            );
        }
    }

    best_score
}

#[allow(clippy::too_many_arguments)]
fn search_deeper(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    alpha: i16,
    beta: i16,
    found_pv: bool,
    previous_move: Option<Move>,
    ctx: &mut SearchContext,
) -> i16 {
    if found_pv {
        principal_variation_search(board_state, depth, ply, alpha, beta, previous_move, ctx)
    } else {
        let child_cut = ctx.cut_node;
        -search_internal(
            board_state,
            depth.saturating_sub(1),
            ply + 1,
            -beta,
            -alpha,
            previous_move,
            &mut SearchContext {
                allow_null_move: ctx.allow_null_move,
                on_pv_path: ctx.on_pv_path,
                previous_pv: ctx.previous_pv,
                excluded_move: None,
                cut_node: child_cut,
                gtp_graph: ctx.gtp_graph,
                gtp_parent: ctx.gtp_parent,
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            },
        )
    }
}

pub(crate) fn principal_variation_search(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    alpha: i16,
    beta: i16,
    previous_move: Option<Move>,
    ctx: &mut SearchContext,
) -> i16 {
    let child_cut = ctx.cut_node;
    let mut score = -search_internal(
        board_state,
        depth.saturating_sub(1),
        ply + 1,
        -alpha - 1,
        -alpha,
        previous_move,
        &mut SearchContext {
            allow_null_move: ctx.allow_null_move,
            on_pv_path: ctx.on_pv_path,
            previous_pv: ctx.previous_pv,
            excluded_move: None,
            cut_node: child_cut,
            gtp_graph: ctx.gtp_graph,
            gtp_parent: ctx.gtp_parent,
            pv_table: &mut *ctx.pv_table,
            cancellation_token: ctx.cancellation_token,
            search_state: ctx.search_state,
        },
    );
    if score > alpha && score < beta {
        score = -search_internal(
            board_state,
            depth.saturating_sub(1),
            ply + 1,
            -beta,
            -alpha,
            previous_move,
            &mut SearchContext {
                allow_null_move: ctx.allow_null_move,
                on_pv_path: ctx.on_pv_path,
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
    }
    score
}

#[allow(dead_code)]
pub(crate) fn negascout_search(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    alpha: i16,
    beta: i16,
    previous_move: Option<Move>,
    ctx: &mut SearchContext,
) -> i16 {
    principal_variation_search(board_state, depth, ply, alpha, beta, previous_move, ctx)
}

fn update_history_stats(
    board_state: &BoardState,
    search_state: &mut SearchState,
    best_move: Move,
    depth: u8,
    previous_move: Option<Move>,
    tried_quiets: &[Move],
) {
    let bonus = (133 * depth as i32 - 81).clamp(0, 1487);
    let malus = (968 * depth as i32 - 235).clamp(0, 2244);
    let piece = board_state.get_piece_on(best_move.source);
    let side = board_state.side_to_move;

    if piece >= 0 {
        let quiet_bonus = bonus * 899 / 1024;
        search_state
            .move_ordering
            .update_history(piece as usize, best_move, quiet_bonus);
        search_state
            .move_ordering
            .update_quiet_history(side, best_move, quiet_bonus);
        update_continuation(
            search_state,
            board_state,
            previous_move,
            best_move,
            quiet_bonus,
        );
    }

    let mut actual_malus = malus * 1159 / 1024;
    for &quiet_move in tried_quiets {
        if quiet_move != best_move {
            actual_malus = actual_malus * 921 / 1024;
            let q_piece = board_state.get_piece_on(quiet_move.source);
            if q_piece >= 0 {
                search_state.move_ordering.update_history(
                    q_piece as usize,
                    quiet_move,
                    -actual_malus,
                );
                search_state
                    .move_ordering
                    .update_quiet_history(side, quiet_move, -actual_malus);
                update_continuation(
                    search_state,
                    board_state,
                    previous_move,
                    quiet_move,
                    -actual_malus,
                );
            }
        }
    }
}

fn update_continuation(
    search_state: &mut SearchState,
    board_state: &BoardState,
    previous_move: Option<Move>,
    move_obj: Move,
    bonus: i32,
) {
    if let Some(previous_move) = previous_move {
        let prev_target = previous_move.target as usize;
        if prev_target < crate::common::constants::SQUARES {
            let previous_piece = board_state.piece_mapping[prev_target];
            if previous_piece != Piece::None {
                search_state.move_ordering.update_continuation_history(
                    previous_piece,
                    previous_move.target,
                    move_obj,
                    bonus,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn beta_cutoff(
    score: i16,
    move_obj: Move,
    ply: usize,
    board_state: &BoardState,
    depth: u8,
    previous_move: Option<Move>,
    search_state: &mut SearchState,
    cancellation_token: &AtomicBool,
    tried_quiets: &[Move],
    tried_captures: &[Move],
    excluded_move: Option<Move>,
) -> i16 {
    if cancellation_token.load(Ordering::Relaxed) {
        return score;
    }
    if excluded_move.is_some() {
        return score;
    }
    search_state.tt.submit_entry(
        board_state.board_hash,
        tt::TranspositionTable::adjust_score(score, ply as i32, board_state.half_move_clock),
        depth,
        move_obj,
        TranspositionEntryType::Beta,
    );

    if !move_obj.is_capture() {
        search_state.move_ordering.add_killer_move(move_obj, ply);

        update_history_stats(
            board_state,
            search_state,
            move_obj,
            depth,
            previous_move,
            tried_quiets,
        );

        if let Some(prev_mv) = previous_move {
            let prev_target = prev_mv.target as usize;
            if prev_target < crate::common::constants::SQUARES {
                let prev_side = board_state.side_to_move.other();
                let prev_piece = board_state.piece_mapping[prev_target];
                if prev_piece != Piece::None {
                    search_state.move_ordering.add_counter_move(
                        prev_side,
                        prev_piece,
                        prev_mv.target,
                        move_obj,
                    );
                }
            }
        }
    } else {
        let src = move_obj.source as usize;
        let tgt = move_obj.target as usize;
        if src < crate::common::constants::SQUARES && tgt < crate::common::constants::SQUARES {
            let piece = board_state.piece_mapping[src];
            if piece != Piece::None {
                let moved_piece = board_state.get_piece_on(move_obj.source);
                let captured_piece = if move_obj.move_type == MoveType::EnPassant {
                    Piece::Pawn
                } else {
                    board_state.piece_mapping[tgt]
                };
                if moved_piece >= 0 && captured_piece != Piece::None {
                    let bonus = (133 * depth as i32 - 81).clamp(0, 1487);
                    let malus = (968 * depth as i32 - 235).clamp(0, 2244);
                    search_state.move_ordering.update_capture_history(
                        moved_piece as usize,
                        move_obj.target,
                        captured_piece,
                        bonus * 1427 / 1024,
                    );
                    for &prev_cap in tried_captures {
                        let prev_tgt = prev_cap.target as usize;
                        if prev_tgt < crate::common::constants::SQUARES {
                            let prev_moved = board_state.get_piece_on(prev_cap.source);
                            let prev_captured = if prev_cap.move_type == MoveType::EnPassant {
                                Piece::Pawn
                            } else {
                                board_state.piece_mapping[prev_tgt]
                            };
                            if prev_moved >= 0 && prev_captured != Piece::None {
                                search_state.move_ordering.update_capture_history(
                                    prev_moved as usize,
                                    prev_cap.target,
                                    prev_captured,
                                    -malus * 1332 / 1024,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    score
}

pub struct SearchContext<'a> {
    pub allow_null_move: bool,
    pub on_pv_path: bool,
    pub previous_pv: &'a [Move],
    pub excluded_move: Option<Move>,

    pub cut_node: bool,
    gtp_graph: gtp::GtpTreeGraph,
    gtp_parent: Option<usize>,
    pub pv_table: &'a mut PvTable,
    pub cancellation_token: &'a AtomicBool,
    pub search_state: &'a mut SearchState,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;
    use crate::common::square::Square;

    const MATE_IN_ONE: &str = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
    const STALEMATE: &str = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1";
    const IN_CHECK_ESCAPE: &str = "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1";
    const FEW_PIECE_ENDGAME: &str = "4k3/8/8/8/8/8/8/4K2R w K - 0 1";

    fn run_search(fen: &str, depth: u8, alpha: i16, beta: i16) -> (i16, u64) {
        let mut board = BoardState::parse_fen(fen);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let score = search(
            &mut board,
            depth,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        (score, state.nodes)
    }

    #[test]
    fn depth_one_startpos_returns_legal_score() {
        let (score, nodes) = run_search(STARTING_FEN, 1, i16::MIN + 1, i16::MAX - 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn depth_two_endgame_stays_fast_and_bounded() {
        let (score, nodes) = run_search(FEW_PIECE_ENDGAME, 2, i16::MIN + 1, i16::MAX - 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn mate_in_one_scores_near_mate() {
        let (score, _) = run_search(MATE_IN_ONE, 1, i16::MIN + 1, i16::MAX - 1);
        assert!(score > constants::MAX_CENTIPAWN_EVAL - 100);
    }

    #[test]
    fn stalemate_returns_zero() {
        let (score, _) = run_search(STALEMATE, 1, i16::MIN + 1, i16::MAX - 1);

        assert!(score.abs() <= 2, "stalemate score {score}");
    }

    #[test]
    fn cancelled_root_returns_zero() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let cancel = AtomicBool::new(true);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let score = search(
            &mut board,
            1,
            i16::MIN + 1,
            i16::MAX - 1,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert_eq!(score, 0);
    }

    #[test]
    fn fifty_move_draw_returns_zero() {
        let (score, _) = run_search(
            "7k/8/5K2/8/8/8/8/8 b - - 100 150",
            1,
            i16::MIN + 1,
            i16::MAX - 1,
        );
        assert!(score.abs() <= 2, "fifty-move score {score}");
    }

    #[test]
    fn depth_zero_delegates_to_quiescence() {
        let (score, nodes) = run_search(FEW_PIECE_ENDGAME, 0, i16::MIN + 1, i16::MAX - 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn tt_exact_cutoff_hits_on_non_pv() {
        let mut board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(250, 0, board.half_move_clock),
            5,
            Move::NO_MOVE,
            TranspositionEntryType::Exact,
        );
        let score = search(&mut board, 1, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert_eq!(score, 250);
        assert_eq!(state.nodes, 1);
    }

    #[test]
    fn rfp_prunes_with_depressed_beta() {
        let (score, nodes) = run_search(STARTING_FEN, 1, -1000, -999);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert_eq!(nodes, 1);
    }

    #[test]
    fn in_check_evasion_searches_moves() {
        let board = BoardState::parse_fen(IN_CHECK_ESCAPE);
        assert!(board.is_in_check(board.side_to_move));
        let (score, nodes) = run_search(IN_CHECK_ESCAPE, 1, i16::MIN + 1, i16::MAX - 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn root_searchmoves_filter_restricts_to_one() {
        use crate::common::move_type::MoveType;
        let mut board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let only = Move::new(Square::H1, Square::G1, MoveType::Quiet);
        assert!(board.is_legal(only));
        state.searchmoves = vec![only];
        let score = search(
            &mut board,
            1,
            i16::MIN + 1,
            i16::MAX - 1,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert_eq!(pv_table.line().first(), Some(&only));
    }

    #[test]
    fn narrow_window_covers_beta_cutoff_histories() {
        let (score, nodes) = run_search("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1", 2, 0, 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn mate_distance_clamp_returns_alpha_immediately() {
        let (score, nodes) = run_search(STARTING_FEN, 1, 30_000, 30_000);
        assert_eq!(score, 30_000);
        assert_eq!(nodes, 1);
    }

    #[test]
    fn beta_is_mate_skips_rfp_and_nmp_gates() {
        let (score, nodes) = run_search(STARTING_FEN, 1, 30_999, 31_000);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn small_prob_beta_returns_without_full_search() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(1000, 0, board.half_move_clock),
            0,
            Move::NO_MOVE,
            TranspositionEntryType::Exact,
        );
        let score = search(&mut board, 2, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert_eq!(score, 381);
    }

    #[test]
    fn tt_alpha_cutoff_hits_on_shallow_non_pv() {
        let mut board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(-100, 0, board.half_move_clock),
            5,
            Move::NO_MOVE,
            TranspositionEntryType::Alpha,
        );
        let score = search(&mut board, 1, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert_eq!(score, -100);
        assert_eq!(state.nodes, 1);
    }

    #[test]
    fn tt_beta_deep_cutoff_hits_on_stalemate() {
        let mut board = BoardState::parse_fen(STALEMATE);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(500, 0, board.half_move_clock),
            7,
            Move::NO_MOVE,
            TranspositionEntryType::Beta,
        );
        let score = search(&mut board, 6, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert_eq!(score, 500);
        assert_eq!(state.nodes, 1);
    }

    #[test]
    fn tt_shallow_depth_miss_falls_through_to_search() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(100, 0, board.half_move_clock),
            1,
            Move::NO_MOVE,
            TranspositionEntryType::Beta,
        );
        let score = search(&mut board, 2, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 1);
    }

    #[test]
    fn tt_cutoff_skipped_when_halfmove_above_90() {
        let mut board = BoardState::parse_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 97 150");
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(250, 0, board.half_move_clock),
            5,
            Move::NO_MOVE,
            TranspositionEntryType::Exact,
        );
        let score = search(&mut board, 1, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert!(score.abs() <= 2, "gated TT score {score}");
    }

    #[test]
    fn null_move_prune_triggers_on_material_up() {
        let (score, nodes) = run_search("7k/8/8/8/8/8/QQQ5/K7 w - - 0 1", 2, 0, 1);
        assert!(score > 0);
        assert!(score < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn futility_prunes_late_quiets_with_high_alpha() {
        let (score, nodes) = run_search(STARTING_FEN, 2, 20_000, 20_001);
        assert!(score < 20_000);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn see_prune_skips_second_losing_capture() {
        let (score, nodes) = run_search("7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1", 2, 700, 701);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn lmr_reduces_late_quiets_with_pvs_research() {
        let (score, nodes) = run_search(STARTING_FEN, 3, i16::MIN + 1, i16::MAX - 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    fn singular_stalemate_search(tt_score: i16, alpha: i16, beta: i16) -> (i16, u64) {
        use crate::common::move_type::MoveType;
        let mut board = BoardState::parse_fen(STALEMATE);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let tt_best = Move::new(Square::H8, Square::G8, MoveType::Quiet);
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(tt_score, 0, board.half_move_clock),
            6,
            tt_best,
            TranspositionEntryType::Exact,
        );
        let score = search(
            &mut board,
            6,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        (score, state.nodes)
    }

    #[test]
    fn singular_double_extension_on_hopeless_exclusion() {
        let (score, nodes) = singular_stalemate_search(300, 0, 1);
        assert!(score.abs() <= 2, "singular score {score}");
        assert!(nodes > 0);
    }

    #[test]
    fn singular_single_extension_near_margin() {
        let (score, nodes) = singular_stalemate_search(13, 0, 1);
        assert!(score.abs() <= 2, "singular score {score}");
        assert!(nodes > 0);
    }

    #[test]
    fn singular_negative_extension_when_original_above_beta() {
        let (score, nodes) = singular_stalemate_search(10, 0, 1);
        assert!(score.abs() <= 2, "singular score {score}");
        assert!(nodes > 0);
    }

    #[test]
    fn singular_fail_high_returns_with_correction_bonus() {
        let (score, nodes) = singular_stalemate_search(0, -5, -4);
        assert!(score.abs() <= 2, "singular score {score}");
        assert!(nodes > 0);
    }

    #[test]
    fn extension_cap_never_exceeds_two_in_a_row() {
        let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
        crate::eval::nnue::v16::unload_nets();
        let mut board = BoardState::parse_fen(crate::common::helpers::KIWI_PETE_FEN);
        let cancel = AtomicBool::new(false);
        let mut debug = false;
        let mut state = SearchState::new();
        let best = board.find_best_move(8, &cancel, &mut debug, &mut state, 1);
        assert!(state.score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
        assert_ne!(best, Move::NO_MOVE);
        assert!(
            state.nodes < 5_000_000,
            "extension cap failed, visited {} nodes",
            state.nodes
        );
        assert!(
            state.seldepth <= 48,
            "extension cap failed, seldepth {}",
            state.seldepth
        );
    }

    #[test]
    fn extension_streak_slot_roundtrips() {
        let mut state = SearchState::new();
        state.extension_streak[3] = 2;
        assert_eq!(state.extension_streak[3], 2);
        state.reset_search();
        assert_eq!(state.extension_streak[3], 0);
        let worker = state.clone_for_worker(1);
        assert_eq!(worker.extension_streak[3], 0);
    }

    #[test]
    fn check_evasion_deep_copies_eval_stack() {
        let (score, nodes) = run_search(IN_CHECK_ESCAPE, 3, i16::MIN + 1, i16::MAX - 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn mate_against_returns_mated_score() {
        let (score, _) = run_search(
            "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
            1,
            i16::MIN + 1,
            i16::MAX - 1,
        );
        assert_eq!(score, -constants::MAX_CENTIPAWN_EVAL);
    }

    #[test]
    fn coarse_pass_runs_on_deep_stalemate_both_arms() {
        let (lo_score, lo_nodes) = run_search(STALEMATE, 7, 0, 1);
        assert!(lo_score.abs() <= 2, "coarse lo {lo_score}");
        assert!(lo_nodes > 0);
        let (hi_score, hi_nodes) = run_search(STALEMATE, 7, 1000, 1001);
        assert!(hi_score.abs() <= 2, "coarse hi {hi_score}");
        assert!(hi_nodes > 0);
    }

    #[test]
    fn probcut_verifies_captures_on_tiny_board() {
        let (score, nodes) = run_search("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1", 4, 0, 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    fn run_search_at_ply(fen: &str, depth: u8, ply: u8, alpha: i16, beta: i16) -> (i16, u64) {
        let mut board = BoardState::parse_fen(fen);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let score = {
            let mut ctx = SearchContext {
                allow_null_move: true,
                on_pv_path: true,
                previous_pv: &[],
                excluded_move: None,
                cut_node: false,
                gtp_graph: gtp::GtpTreeGraph::new(),
                gtp_parent: None,
                pv_table: &mut pv_table,
                cancellation_token: &cancel,
                search_state: &mut state,
            };
            search_internal(&mut board, depth, ply, alpha, beta, None, &mut ctx)
        };
        (score, state.nodes)
    }

    #[test]
    fn max_ply_delegates_to_quiescence() {
        let (score, nodes) = run_search_at_ply(
            FEW_PIECE_ENDGAME,
            1,
            constants::MAX_PLY as u8,
            i16::MIN + 1,
            i16::MAX - 1,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn psm_disabled_takes_zero_delta() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.params.psm_enabled = false;
        let score = search(
            &mut board,
            1,
            i16::MIN + 1,
            i16::MAX - 1,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }

    #[test]
    fn singular_beta_clamped_to_mate_floor() {
        use crate::common::move_type::MoveType;
        let mut board = BoardState::parse_fen(STALEMATE);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let tt_best = Move::new(Square::H8, Square::G8, MoveType::Quiet);
        state.tt.submit_entry(
            board.board_hash,
            tt::TranspositionTable::adjust_score(-30990, 0, board.half_move_clock),
            5,
            tt_best,
            TranspositionEntryType::Exact,
        );
        let score = search(&mut board, 6, 0, 1, &cancel, &[], &mut pv_table, &mut state);
        assert!(score.abs() <= 2, "singular clamp score {score}");
        assert!(state.nodes > 0);
    }

    const PROBCUT_FEN: &str = "7k/8/8/3pp3/8/8/3R4/K3Q3 w - - 0 1";

    fn probcut_window(fen: &str, depth: u8, sub: i16) -> (i16, i16) {
        let mut tmp = BoardState::parse_fen(fen);
        let raw = crate::eval::evaluate_with_depth(&mut tmp, 0, depth);
        let beta = raw.saturating_sub(sub);
        let alpha = beta.saturating_sub(1);
        (alpha, beta)
    }

    #[test]
    fn probcut_improving_and_capture_sort_hit() {
        let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
        crate::eval::nnue::v16::unload_nets();
        let (alpha, beta) = probcut_window(PROBCUT_FEN, 4, 300);
        let mut board = BoardState::parse_fen(PROBCUT_FEN);
        let hash = board.board_hash;
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let score = search(
            &mut board,
            4,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
        let entry = state.tt.probe(hash).expect("probcut stores TT");
        assert_eq!(entry.entry_type, TranspositionEntryType::Beta);
        assert_eq!(entry.depth, 1);
    }

    #[test]
    fn probcut_verify_with_depth_six_hits_tce() {
        let _eval_guard = crate::eval::nnue::v16::EVAL_TEST_LOCK.lock().unwrap();
        crate::eval::nnue::v16::unload_nets();
        let (alpha, beta) = probcut_window(PROBCUT_FEN, 6, 300);
        let mut board = BoardState::parse_fen(PROBCUT_FEN);
        let hash = board.board_hash;
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        let score = search(
            &mut board,
            6,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
        let entry = state.tt.probe(hash).expect("probcut stores TT");
        assert_eq!(entry.entry_type, TranspositionEntryType::Beta);
        assert_eq!(entry.depth, 1);
    }

    #[test]
    fn coarse_pass_updates_tt_best() {
        let (score, nodes) = run_search_at_ply("7k/8/8/8/3p4/8/3R4/K7 w - - 0 1", 7, 60, 0, 1);
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    #[test]
    fn pawn_push_to_seventh_gets_extension() {
        let (score, nodes) = run_search(
            "4k3/8/4P3/8/8/8/8/4K3 w - - 0 1",
            2,
            i16::MIN + 1,
            i16::MAX - 1,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(nodes > 0);
    }

    const PAWN_KING_FEN: &str = "4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1";

    fn fail_low_window(fen: &str, depth: u8, add: i16) -> (i16, i16) {
        let mut tmp = BoardState::parse_fen(fen);
        let raw = crate::eval::evaluate_with_depth(&mut tmp, 0, depth);
        let alpha = raw.saturating_add(add);
        let beta = alpha.saturating_add(1);
        (alpha, beta)
    }

    #[test]
    fn quiet_history_negative_prunes_late_quiets() {
        let (alpha, beta) = fail_low_window(PAWN_KING_FEN, 4, 200);
        let mut board = BoardState::parse_fen(PAWN_KING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.params.alp_threshold = 101;
        state.params.gtp_threshold = 0;
        for row in state.move_ordering.history_moves.iter_mut() {
            for s in row.iter_mut() {
                *s = -1000;
            }
        }
        for side in state.move_ordering.quiet_history.iter_mut() {
            for row in side.iter_mut() {
                for s in row.iter_mut() {
                    *s = -1000;
                }
            }
        }
        for piece in state.move_ordering.continuation_history.iter_mut() {
            for row in piece.iter_mut() {
                for s in row.iter_mut() {
                    *s = -1000;
                }
            }
        }
        let score = search(
            &mut board,
            4,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }

    #[test]
    fn psm_sibling_prune_triggers() {
        let (alpha, beta) = fail_low_window(PAWN_KING_FEN, 4, 200);
        let mut board = BoardState::parse_fen(PAWN_KING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.params.alp_threshold = 101;
        state.psm_stack.stack[0].consecutive_fail_lows = 5;
        state.psm_stack.stack[0].hidden = [-256; 128];
        let score = search(
            &mut board,
            4,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }

    #[test]
    fn gtp_subtree_prune_triggers() {
        let (alpha, beta) = fail_low_window(PAWN_KING_FEN, 4, 200);
        let mut board = BoardState::parse_fen(PAWN_KING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.params.alp_threshold = 101;
        state.params.gtp_threshold = 100;
        let score = search(
            &mut board,
            4,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }

    #[test]
    fn late_history_prune_triggers_on_startpos() {
        let (alpha, beta) = fail_low_window(STARTING_FEN, 3, 200);
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let cancel = AtomicBool::new(false);
        let mut pv_table = PvTable::new();
        let mut state = SearchState::new();
        state.params.alp_threshold = 101;
        for row in state.move_ordering.history_moves.iter_mut() {
            for s in row.iter_mut() {
                *s = -10000;
            }
        }
        for side in state.move_ordering.quiet_history.iter_mut() {
            for row in side.iter_mut() {
                for s in row.iter_mut() {
                    *s = -10000;
                }
            }
        }
        for piece in state.move_ordering.continuation_history.iter_mut() {
            for row in piece.iter_mut() {
                for s in row.iter_mut() {
                    *s = -10000;
                }
            }
        }
        let score = search(
            &mut board,
            3,
            alpha,
            beta,
            &cancel,
            &[],
            &mut pv_table,
            &mut state,
        );
        assert!(score.abs() < constants::MAX_CENTIPAWN_EVAL);
        assert!(state.nodes > 0);
    }

    #[test]
    fn beta_cutoff_cancelled_returns_score() {
        let board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
        let cancel = AtomicBool::new(true);
        let mut state = SearchState::new();
        let mv = Move::new(Square::H1, Square::G1, MoveType::Quiet);
        let score = beta_cutoff(
            123,
            mv,
            0,
            &board,
            2,
            None,
            &mut state,
            &cancel,
            &[],
            &[],
            None,
        );
        assert_eq!(score, 123);
    }

    #[test]
    fn beta_cutoff_excluded_returns_score() {
        let board = BoardState::parse_fen(FEW_PIECE_ENDGAME);
        let cancel = AtomicBool::new(false);
        let mut state = SearchState::new();
        let mv = Move::new(Square::H1, Square::G1, MoveType::Quiet);
        let excluded = Move::new(Square::H1, Square::H2, MoveType::Quiet);
        let score = beta_cutoff(
            77,
            mv,
            0,
            &board,
            2,
            None,
            &mut state,
            &cancel,
            &[],
            &[],
            Some(excluded),
        );
        assert_eq!(score, 77);
    }

    #[test]
    fn beta_cutoff_en_passant_updates_capture_history() {
        use crate::common::move_type::MoveType;
        let board = BoardState::parse_fen("7k/8/2b5/3pP3/8/8/8/K2RQ3 w - d6 0 1");
        let cancel = AtomicBool::new(false);
        let mut state = SearchState::new();
        let ep = Move::new(Square::E5, Square::D6, MoveType::EnPassant);
        assert!(board.piece_mapping[Square::E5 as usize] == Piece::Pawn);
        let normal = Move::new(Square::D1, Square::D5, MoveType::Capture);
        let tried = [ep, normal];
        let score = beta_cutoff(
            250,
            ep,
            0,
            &board,
            2,
            None,
            &mut state,
            &cancel,
            &[],
            &tried,
            None,
        );
        assert_eq!(score, 250);
        assert!(state.tt.probe(board.board_hash).is_some());
    }
}
