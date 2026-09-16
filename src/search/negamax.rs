use crate::board::state::BoardState;
use crate::common::constants;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::tt::{self, TranspositionEntryType};
use crate::eval::evaluate_with_optimism;
use crate::search::move_picker::MovePicker;
use crate::search::pv_table::PvTable;
use crate::search::search_state::SearchState;
use crate::search::{lmr, nmp, quiescence};
use std::sync::atomic::{AtomicBool, Ordering};

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
    let mut ctx = SearchContext {
        allow_null_move: true,
        on_pv_path: true,
        previous_pv,
        excluded_move: None,
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
    if ctx.cancellation_token.load(Ordering::Relaxed) {
        return 0;
    }

    ctx.pv_table.clear(ply as usize);

    let mut best_move = Move::NO_MOVE;
    let mut best_score = -constants::MAX_CENTIPAWN_EVAL;

    let is_pv_node = beta > 1 + alpha;
    let in_check = board_state.is_in_check(board_state.side_to_move);

    if ply > 0 && board_state.is_draw_in_search(ply as u16) {
        ctx.search_state.nodes += 1;
        return 0;
    }

    // Syzygy tablebase bounds: never walk out of a won ending (or into a
    // lost one) once few enough pieces remain.
    if ply > 0
        && let Some(tb) = crate::syzygy::probe_bound(board_state, ply)
    {
        return tb;
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

    let depth = if in_check {
        (depth + 1).min(constants::MAX_PLY as u8)
    } else {
        depth
    };

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

        if entry.depth >= depth && !is_pv_node && ctx.excluded_move.is_none() {
            let tt_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
            let cutoff = match entry.entry_type {
                TranspositionEntryType::Exact => true,
                TranspositionEntryType::Alpha => tt_score <= alpha,
                TranspositionEntryType::Beta => tt_score >= beta,
                TranspositionEntryType::None => false,
            };
            if cutoff {
                ctx.search_state.nodes += 1;
                return tt_score;
            }
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

    ctx.search_state.nodes += 1;

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
        let raw_static_eval = evaluate_with_optimism(&mut *board_state, optimism);
        let correction = ctx
            .search_state
            .correction_history
            .get_correction(board_state, previous_move);

        static_eval = (raw_static_eval as i32 + correction as i32)
            .clamp(-mate_bound as i32, mate_bound as i32) as i16;
        ctx.search_state.eval_stack[ply as usize] = static_eval;
    }

    let mut singular_extension: i8 = 0;
    if !in_check
        && ctx.excluded_move.is_none()
        && depth >= 6
        && tt_entry.is_some()
        && tt_best.is_some()
    {
        #[allow(clippy::unnecessary_unwrap)]
        let entry = tt_entry.unwrap();
        if entry.depth >= depth - 3 && entry.entry_type != tt::TranspositionEntryType::Alpha {
            let original_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
            let margin = (depth as i16) * 2;
            let mut singular_beta = original_score - margin;
            if singular_beta < -constants::MAX_CENTIPAWN_EVAL {
                singular_beta = -constants::MAX_CENTIPAWN_EVAL;
            }

            let mut se_ctx = SearchContext {
                allow_null_move: false,
                on_pv_path: false,
                previous_pv: ctx.previous_pv,
                excluded_move: tt_best,
                pv_table: ctx.pv_table,
                search_state: ctx.search_state,
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

            if se_score < singular_beta {
                if se_score < singular_beta - margin {
                    singular_extension = 2;
                } else {
                    singular_extension = 1;
                }
            } else if se_score >= beta {
                if has_static_eval && !in_check && se_score > static_eval {
                    let bonus = (((se_score as i32 - static_eval as i32) * ((depth as i32 - 1) / 2) * 177) / 1024)
                        .clamp(-256, 256);
                    ctx.search_state.correction_history.update(board_state, previous_move, bonus);
                }
                return se_score;
            } else if original_score >= beta {
                singular_extension = -1;
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

    if !is_pv_node && has_static_eval {
        let mut margin = ctx.search_state.params.rfp_margin_mult * depth as i16;
        if !is_improving {
            margin += margin / 3;
        }
        if !beta_is_mate && static_eval.saturating_sub(margin) >= beta {
            return static_eval;
        }

        let small_prob_beta = beta.saturating_add(380);
        if let Some(entry) = tt_entry
            && entry.entry_type != tt::TranspositionEntryType::Alpha
            && entry.depth >= depth.saturating_sub(4)
        {
            let tt_val = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
            if tt_val >= small_prob_beta && tt_val.abs() < mate_bound {
                return small_prob_beta;
            }
        }

        if depth >= 4 && ctx.excluded_move.is_none() && !beta_is_mate {
            let prob_margin = if is_improving {
                ctx.search_state
                    .params
                    .probcut_margin
                    .saturating_sub(50)
                    .max(100)
            } else {
                ctx.search_state.params.probcut_margin
            };
            let prob_beta = beta.saturating_add(prob_margin);
            if prob_beta < mate_bound
                && prob_beta > -mate_bound
                && static_eval >= prob_beta.saturating_sub(50)
            {
                let prob_depth = depth.saturating_sub(4).max(1);
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
                    if board_state.see(mv) < see_threshold {
                        continue;
                    }
                    if !board_state.is_legal(mv) {
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
                    let score = if q_score >= prob_beta && prob_depth > 1 {
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
                                pv_table: &mut *ctx.pv_table,
                                cancellation_token: ctx.cancellation_token,
                                search_state: ctx.search_state,
                            },
                        )
                    } else {
                        q_score
                    };
                    board_state.unmake_move(mv);
                    if ctx.cancellation_token.load(Ordering::Relaxed) {
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
                                tt::TranspositionTable::adjust_score(score, ply as i32),
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

    // PRUNE: Null Move Pruning
    if nmp::can_prune(
        is_pv_node,
        board_state,
        ctx.allow_null_move,
        depth,
        in_check,
        static_eval,
        beta,
    ) {
        board_state.make_null_move();
        let reduction = nmp::get_reduction(depth, &ctx.search_state.params);
        let reduced_depth = depth.saturating_sub(reduction).max(1);
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
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            },
        );
        board_state.undo_null_move();

        if ctx.cancellation_token.load(Ordering::Relaxed) {
            return 0;
        }

        if score >= beta {
            if ctx.excluded_move.is_none() {
                ctx.search_state.tt.submit_entry(
                    board_state.board_hash,
                    tt::TranspositionTable::adjust_score(score, ply as i32),
                    reduced_depth,
                    Move::NO_MOVE,
                    TranspositionEntryType::Beta,
                );
            }
            return score;
        }
    }

    let pv_move = if ctx.on_pv_path && (ply as usize) < ctx.previous_pv.len() {
        Some(ctx.previous_pv[ply as usize])
    } else {
        None
    };

    let mut found_pv = false;
    let mut entry_type = TranspositionEntryType::Alpha;

    let mut move_picker = MovePicker::new(
        pv_move,
        tt_best,
        previous_move,
        ply as usize,
        ctx.excluded_move,
    );
    let mut number_of_legal_moves = 0;
    let mut has_legal_moves = false;
    let mut tried_quiets = [Move::NO_MOVE; 64];
    let mut tried_quiets_count = 0;
    let mut tried_captures = [Move::NO_MOVE; 32];
    let mut tried_captures_count = 0;
    let has_non_pawn_material = board_state.has_non_pawn_material(board_state.side_to_move);

    while let Some(move_obj) = move_picker.next(
        board_state,
        &ctx.search_state.move_ordering,
        &mut ctx.search_state.captures_stack[ply as usize],
        &mut ctx.search_state.quiets_stack[ply as usize],
    ) {
        if ctx.cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        let cap_or_promo = move_obj.is_capture() || move_obj.is_promotion();
        let mut history_score = 0;
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
            if moved_piece >= 0 && captured_piece != Piece::None {
                history_score = ctx.search_state.move_ordering.capture_history[moved_piece as usize]
                    [move_obj.target as usize][captured_piece as usize]
                    as i32;
            }
        }

        // PRUNE: SEE Pruning
        if !is_pv_node && !in_check && depth <= 8 && has_legal_moves {
            let see_threshold = if cap_or_promo {
                -100 * depth as i16
            } else if has_non_pawn_material {
                -35 * (depth as i16) * (depth as i16)
            } else {
                i16::MIN
            };
            if see_threshold > i16::MIN && board_state.see(move_obj) < see_threshold {
                continue;
            }
        }

        if !board_state.is_legal(move_obj) {
            continue;
        }

        board_state.make_move(move_obj);
        ctx.search_state.tt.prefetch(board_state.board_hash);

        has_legal_moves = true;

        let alpha_is_mate = alpha.abs() >= mate_bound;

        let mut gives_check = false;
        let mut gives_check_computed = false;

        let mut extension: i8 = if Some(move_obj) == tt_best {
            singular_extension
        } else {
            0
        };

        let prev_side = board_state.side_to_move.other();
        let piece = board_state.get_piece_on_side(move_obj.target, prev_side);
        if piece == Piece::Pawn as usize {
            let target_rank = (move_obj.target as usize) / 8;
            if (prev_side == Side::White && target_rank == 1)
                || (prev_side == Side::Black && target_rank == 6)
            {
                extension = extension.max(1);
            }
        } else if let Some(prev) = previous_move
            && prev.is_capture()
            && move_obj.is_capture()
            && move_obj.target == prev.target
        {
            extension = extension.max(1);
        }
        let depth = (depth as i16 + extension as i16).max(1) as u8;

        // PRUNE: Futility Pruning
        if !is_pv_node
            && number_of_legal_moves > 0
            && has_static_eval
            && depth < 3
            && !cap_or_promo
            && !alpha_is_mate
            && has_non_pawn_material
            && static_eval
                .saturating_add(ctx.search_state.params.futility_margin_mult * depth as i16)
                <= alpha
        {
            gives_check = board_state.is_in_check(board_state.side_to_move);
            gives_check_computed = true;
            if !gives_check {
                board_state.unmake_move(move_obj);
                continue;
            }
        }

        // PRUNE: Late Move Pruning
        let lmp_threshold = if is_improving {
            3 + (depth as usize * depth as usize)
        } else {
            (3 + (depth as usize * depth as usize)) / 2
        };
        if !is_pv_node
            && has_static_eval
            && depth < 4
            && !cap_or_promo
            && !alpha_is_mate
            && has_non_pawn_material
            && number_of_legal_moves >= lmp_threshold
        {
            if !gives_check_computed {
                gives_check = board_state.is_in_check(board_state.side_to_move);
                gives_check_computed = true;
            }
            if !gives_check {
                board_state.unmake_move(move_obj);
                continue;
            }
        }

        let is_tactical = if cap_or_promo {
            true
        } else if gives_check_computed {
            gives_check
        } else if depth >= 3 && number_of_legal_moves >= 3 && !in_check {
            board_state.is_in_check(board_state.side_to_move)
        } else {
            false
        };

        let needs_lmr = lmr::needs_reduction(depth, number_of_legal_moves, is_tactical, in_check)
            || (is_tactical
                && lmr::needs_tactical_reduction(
                    depth,
                    number_of_legal_moves,
                    in_check,
                    history_score,
                ));

        // PRUNE: History-based Pruning (Late Move Pruning)
        if depth <= 3
            && number_of_legal_moves > 3
            && !is_pv_node
            && !in_check
            && !cap_or_promo
            && has_non_pawn_material
            && history_score < -4000 * depth as i32
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        let mut score;
        let next_on_pv = ctx.on_pv_path && Some(move_obj) == pv_move;
        let move_nodes_start = ctx.search_state.nodes;

        // REDUCTION: Late Move Reductions
        if needs_lmr {
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
            };
            let reduction = lmr::compute_reduction(
                &lmr_query,
                &ctx.search_state.lmr_table,
                &ctx.search_state.params.lmr_divisor,
            );
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
                    pv_table: &mut *ctx.pv_table,
                    cancellation_token: ctx.cancellation_token,
                    search_state: ctx.search_state,
                },
            );

            if score > alpha {
                let mut child_ctx = SearchContext {
                    allow_null_move: true,
                    on_pv_path: next_on_pv,
                    previous_pv: ctx.previous_pv,
                    excluded_move: None,
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
        } else {
            let mut child_ctx = SearchContext {
                allow_null_move: true,
                on_pv_path: next_on_pv,
                previous_pv: ctx.previous_pv,
                excluded_move: None,
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

        number_of_legal_moves += 1;

        board_state.unmake_move(move_obj);

        if ctx.cancellation_token.load(Ordering::Relaxed) {
            return 0;
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                alpha_update(score, move_obj, &mut alpha, &mut best_move);
                entry_type = TranspositionEntryType::Exact;
                found_pv = true;

                ctx.pv_table.update(ply as usize, move_obj);
                if ply == 0 {
                    ctx.search_state.root_best_move_nodes =
                        (ctx.search_state.nodes - move_nodes_start) as i64;
                    if number_of_legal_moves > 1 {
                        ctx.search_state.best_move_changes += 1;
                    }
                }
            }
        }

        if score >= beta {
            return beta_cutoff(
                score,
                move_obj,
                ply as usize,
                board_state,
                depth,
                previous_move,
                ctx.search_state,
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

    if !has_legal_moves {
        if in_check {
            return -constants::MAX_CENTIPAWN_EVAL + ply as i16;
        }
        return 0;
    }

    if !ctx.cancellation_token.load(Ordering::Relaxed) {
        if ctx.excluded_move.is_none() {
            ctx.search_state.tt.submit_entry(
                board_state.board_hash,
                tt::TranspositionTable::adjust_score(best_score, ply as i32),
                depth,
                best_move,
                entry_type,
            );
        }

        if has_static_eval
            && !in_check
            && !best_move.is_capture()
            && best_score.abs() < mate_bound
            && (best_score > static_eval) == (best_move != Move::NO_MOVE)
        {
            let move_factor = if best_move != Move::NO_MOVE { 12 } else { 18 };
            let bonus = (((best_score as i32 - static_eval as i32) * depth as i32 * move_factor) / 128)
                .clamp(-256, 256);
            let final_bonus = bonus * 1061 / 1024;
            ctx.search_state
                .correction_history
                .update(board_state, previous_move, final_bonus);
        }

        if entry_type == TranspositionEntryType::Exact
            && best_move != Move::NO_MOVE
            && !best_move.is_capture()
        {
            update_history_stats(
                board_state,
                ctx.search_state,
                best_move,
                depth,
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
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            },
        )
    }
}

fn principal_variation_search(
    board_state: &mut BoardState,
    depth: u8,
    ply: u8,
    alpha: i16,
    beta: i16,
    previous_move: Option<Move>,
    ctx: &mut SearchContext,
) -> i16 {
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
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            },
        );
    }
    score
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
    let piece = board_state.get_piece_on(best_move.source) as usize;
    let side = board_state.side_to_move;

    let quiet_bonus = bonus * 899 / 1024;
    search_state
        .move_ordering
        .update_history(piece, best_move, quiet_bonus);
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

    let mut actual_malus = malus * 1159 / 1024;
    for &quiet_move in tried_quiets {
        if quiet_move != best_move {
            actual_malus = actual_malus * 921 / 1024;
            let q_piece = board_state.get_piece_on(quiet_move.source) as usize;
            search_state
                .move_ordering
                .update_history(q_piece, quiet_move, -actual_malus);
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

fn update_continuation(
    search_state: &mut SearchState,
    board_state: &BoardState,
    previous_move: Option<Move>,
    move_obj: Move,
    bonus: i32,
) {
    if let Some(previous_move) = previous_move {
        let previous_piece = board_state.piece_mapping[previous_move.target as usize];
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

#[inline(always)]
fn alpha_update(score: i16, move_obj: Move, alpha: &mut i16, best_move: &mut Move) {
    *alpha = score;
    *best_move = move_obj;
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
    tried_quiets: &[Move],
    tried_captures: &[Move],
    excluded_move: Option<Move>,
) -> i16 {
    if excluded_move.is_none() {
        search_state.tt.submit_entry(
            board_state.board_hash,
            tt::TranspositionTable::adjust_score(score, ply as i32),
            depth,
            move_obj,
            TranspositionEntryType::Beta,
        );
    }

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
            let prev_side = board_state.side_to_move.other();
            let prev_piece = board_state.piece_mapping[prev_mv.target as usize];
            if prev_piece != Piece::None {
                search_state.move_ordering.add_counter_move(
                    prev_side,
                    prev_piece,
                    prev_mv.target,
                    move_obj,
                );
            }
        }
    } else {
        let piece = board_state.piece_mapping[move_obj.source as usize];
        if piece != Piece::None {
            let moved_piece = board_state.get_piece_on(move_obj.source);
            let captured_piece = if move_obj.move_type == MoveType::EnPassant {
                Piece::Pawn
            } else {
                board_state.piece_mapping[move_obj.target as usize]
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
                    let prev_moved = board_state.get_piece_on(prev_cap.source);
                    let prev_captured = if prev_cap.move_type == MoveType::EnPassant {
                        Piece::Pawn
                    } else {
                        board_state.piece_mapping[prev_cap.target as usize]
                    };
                    if prev_moved >= 0 && prev_captured != Piece::None {
                        search_state.move_ordering.update_capture_history(
                            prev_moved as usize,
                            prev_cap.target,
                            prev_captured,
                            -malus * 1489 / 1024,
                        );
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
    pub pv_table: &'a mut PvTable,
    pub cancellation_token: &'a AtomicBool,
    pub search_state: &'a mut SearchState,
}
