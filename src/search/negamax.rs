use crate::board::state::BoardState;
use crate::common::constants;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::tt::{self, TranspositionEntryType};
use crate::eval::evaluate;
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
        excluded_move: None,
        allow_null_move: true,
        on_pv_path: true,
        previous_pv,
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

    if board_state.is_draw_in_search(ply as u16) {
        ctx.search_state.nodes += 1;
        return 0;
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
    if let Some(entry) = tt_entry
        && entry.best_move != Move::NO_MOVE {
            tt_best = Some(entry.best_move);
        }

    let (has_value, tt_score, _) =
        ctx.search_state
            .tt
            .get_entry(board_state.board_hash, alpha, beta, depth, ply);

    if has_value && !is_pv_node && ctx.excluded_move.is_none() {
        ctx.search_state.nodes += 1;
        return tt_score;
    }

    let mut singular_extension = 0;
    if !in_check
        && ctx.excluded_move.is_none()
        && depth >= 8
        && tt_entry.is_some()
        && tt_best.is_some()
    {
        #[allow(clippy::unnecessary_unwrap)]
        let entry = tt_entry.unwrap();
        if entry.depth >= depth - 3 && entry.entry_type != tt::TranspositionEntryType::Alpha {
            let original_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
            let margin = depth as i16 * ctx.search_state.params.singular_margin_mult;
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
                depth / 2,
                ply,
                singular_beta - 1,
                singular_beta,
                previous_move,
                &mut se_ctx,
            );

            if se_score < singular_beta {
                singular_extension = 1;
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

    // PRUNE: Reverse Futility Pruning
    // TODO: tune conditions
    let mut static_eval = 0;
    let mut raw_static_eval = 0;
    let has_static_eval = !is_pv_node && !in_check;

    let mate_bound = constants::MAX_CENTIPAWN_EVAL - constants::MAX_PLY as i16;
    let beta_is_mate = beta.abs() >= mate_bound;

    if has_static_eval {
        raw_static_eval = evaluate(&mut *board_state);
        static_eval = raw_static_eval;

        let stm = board_state.side_to_move as usize;
        let pawn_hash = crate::common::zobrist::get_pawn_hash(board_state);
        let pawn_idx = (pawn_hash & 0x3FFF) as usize;
        let non_pawn_hash = board_state.board_hash ^ pawn_hash;
        let non_pawn_idx = (non_pawn_hash & 0x3FFF) as usize;

        let pawn_corr = ctx.search_state.pawn_correction_history[stm][pawn_idx];
        let non_pawn_corr = ctx.search_state.non_pawn_correction_history[stm][non_pawn_idx];
        let correction = (pawn_corr + non_pawn_corr) / 256;

        static_eval =
            (static_eval as i32 + correction).clamp(-mate_bound as i32, mate_bound as i32) as i16;

        let margin = ctx.search_state.params.rfp_margin_mult * depth as i16;
        if !beta_is_mate && static_eval.saturating_sub(margin) >= beta {
            return static_eval;
        }
    }

    // PRUNE: ProbCut
    if !is_pv_node && !in_check && depth >= 5 && !beta_is_mate {
        let prob_beta = beta + ctx.search_state.params.probcut_margin;

        let mut probcut_success = false;
        if let Some(entry) = &tt_entry {
            let tt_score = tt::TranspositionTable::retrieve_score(entry.score, ply as i32);
            if entry.depth >= depth - 3
                && entry.entry_type != tt::TranspositionEntryType::Alpha
                && tt_score >= prob_beta
            {
                probcut_success = true;
            }
        }

        if probcut_success {
            return prob_beta;
        }

        let probcut_ply = (ply + 1) as usize;
        let mut prob_picker = MovePicker::new(None, tt_best, previous_move, ply as usize, None);

        while let Some(move_obj) = prob_picker.next(
            board_state,
            &ctx.search_state.move_ordering,
            &mut ctx.search_state.captures_stack[probcut_ply],
            &mut ctx.search_state.quiets_stack[probcut_ply],
        ) {
            if !move_obj.is_capture() && !move_obj.is_promotion() {
                continue;
            }

            board_state.make_move(move_obj);
            if board_state.is_in_check(board_state.side_to_move.other()) {
                board_state.unmake_move(move_obj);
                continue;
            }

            let mut prob_ctx = SearchContext {
                excluded_move: None,
                allow_null_move: false,
                on_pv_path: false,
                previous_pv: ctx.previous_pv,
                pv_table: &mut *ctx.pv_table,
                cancellation_token: ctx.cancellation_token,
                search_state: ctx.search_state,
            };

            let score = -search_internal(
                board_state,
                depth.saturating_sub(4).max(1),
                ply + 1,
                -prob_beta,
                -prob_beta + 1,
                Some(move_obj),
                &mut prob_ctx,
            );
            board_state.unmake_move(move_obj);

            if score >= prob_beta {
                return prob_beta;
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
                excluded_move: None,
                allow_null_move: false,
                on_pv_path: false,
                previous_pv: ctx.previous_pv,
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

    while let Some(move_obj) = move_picker.next(
        board_state,
        &ctx.search_state.move_ordering,
        &mut ctx.search_state.captures_stack[ply as usize],
        &mut ctx.search_state.quiets_stack[ply as usize],
    ) {
        if ctx.cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        let mut extension = if Some(move_obj) == tt_best {
            singular_extension
        } else {
            0
        };

        let cap_or_promo = move_obj.is_capture() || move_obj.is_promotion();
        let mut history_score = 0;
        if !cap_or_promo {
            history_score = ctx.search_state.move_ordering.get_quiet_history_score(
                board_state,
                move_obj,
                previous_move,
            );
        }

        board_state.make_move(move_obj);
        if board_state.is_in_check(board_state.side_to_move.other()) {
            board_state.unmake_move(move_obj);
            continue;
        }

        has_legal_moves = true;

        let alpha_is_mate = alpha.abs() >= mate_bound;

        let mut gives_check = false;
        let mut gives_check_computed = false;

        if extension > 0 {
            gives_check = board_state.is_in_check(board_state.side_to_move);
            gives_check_computed = true;
            if gives_check {
                extension = 2;
            }
        }
        let depth = depth + extension;

        // PRUNE: Futility Pruning
        if number_of_legal_moves > 0
            && has_static_eval
            && depth < 3
            && !cap_or_promo
            && !alpha_is_mate
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
        // TODO: tune base and depth limit
        if has_static_eval
            && depth < 4
            && !cap_or_promo
            && !alpha_is_mate
            && number_of_legal_moves >= 3 + (depth as usize * depth as usize)
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

        let needs_lmr = lmr::needs_reduction(depth, number_of_legal_moves, is_tactical, in_check);

        // PRUNE: History-based Pruning (Late Move Pruning)
        if depth <= 3
            && number_of_legal_moves > 3
            && !is_pv_node
            && !in_check
            && !cap_or_promo
            && history_score < -4000 * depth as i32
        {
            board_state.unmake_move(move_obj);
            continue;
        }

        let mut score;
        let next_on_pv = ctx.on_pv_path && Some(move_obj) == pv_move;

        // REDUCTION: Late Move Reductions
        if needs_lmr {
            let mut reduction = lmr::get_reduction(
                depth,
                number_of_legal_moves,
                is_pv_node,
                &ctx.search_state.params,
            );
            if !cap_or_promo {
                let history_bonus = history_score / 8192;
                reduction = (reduction as i32 - history_bonus).clamp(0, depth as i32 - 1) as u8;
            }
            score = -search_internal(
                board_state,
                depth.saturating_sub(1 + reduction),
                ply + 1,
                -alpha - 1,
                -alpha,
                Some(move_obj),
                &mut SearchContext {
                    excluded_move: None,
                    allow_null_move: true,
                    on_pv_path: false,
                    previous_pv: ctx.previous_pv,
                    pv_table: &mut *ctx.pv_table,
                    cancellation_token: ctx.cancellation_token,
                    search_state: ctx.search_state,
                },
            );

            if score > alpha {
                let mut child_ctx = SearchContext {
                    excluded_move: None,
                    allow_null_move: true,
                    on_pv_path: next_on_pv,
                    previous_pv: ctx.previous_pv,
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
                excluded_move: None,
                allow_null_move: true,
                on_pv_path: next_on_pv,
                previous_pv: ctx.previous_pv,
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
                ctx.excluded_move,
            );
        }

        if !move_obj.is_capture() && tried_quiets_count < tried_quiets.len() {
            tried_quiets[tried_quiets_count] = move_obj;
            tried_quiets_count += 1;
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
            && best_score.abs() < (constants::MAX_CENTIPAWN_EVAL - constants::MAX_PLY as i16)
        {
            let diff = best_score as i32 - raw_static_eval as i32;
            let weight = std::cmp::min(
                (1 + depth as i32) * ctx.search_state.params.history_weight_mult,
                ctx.search_state.params.history_weight_max,
            );

            let stm = board_state.side_to_move as usize;
            let pawn_hash = crate::common::zobrist::get_pawn_hash(board_state);
            let pawn_idx = (pawn_hash & 0x3FFF) as usize;
            let non_pawn_hash = board_state.board_hash ^ pawn_hash;
            let non_pawn_idx = (non_pawn_hash & 0x3FFF) as usize;

            let pawn_corr = &mut ctx.search_state.pawn_correction_history[stm][pawn_idx];
            *pawn_corr = (*pawn_corr * (256 - weight) + diff * weight * 256) / 256;

            let non_pawn_corr =
                &mut ctx.search_state.non_pawn_correction_history[stm][non_pawn_idx];
            *non_pawn_corr = (*non_pawn_corr * (256 - weight) + diff * weight * 256) / 256;
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
                excluded_move: None,
                allow_null_move: ctx.allow_null_move,
                on_pv_path: ctx.on_pv_path,
                previous_pv: ctx.previous_pv,
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
            excluded_move: None,
            allow_null_move: ctx.allow_null_move,
            on_pv_path: ctx.on_pv_path,
            previous_pv: ctx.previous_pv,
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
                excluded_move: None,
                allow_null_move: ctx.allow_null_move,
                on_pv_path: ctx.on_pv_path,
                previous_pv: ctx.previous_pv,
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
    let bonus = (300 * depth as i32) - 250;
    let piece = board_state.get_piece_on(best_move.source) as usize;
    let side = board_state.side_to_move;
    search_state
        .move_ordering
        .update_history(piece, best_move, bonus);
    search_state
        .move_ordering
        .update_quiet_history(side, best_move, bonus);
    update_continuation(search_state, board_state, previous_move, best_move, bonus);

    for &quiet_move in tried_quiets {
        if quiet_move != best_move {
            let q_piece = board_state.get_piece_on(quiet_move.source) as usize;
            let penalty = -bonus;
            search_state
                .move_ordering
                .update_history(q_piece, quiet_move, penalty);
            search_state
                .move_ordering
                .update_quiet_history(side, quiet_move, penalty);
            update_continuation(
                search_state,
                board_state,
                previous_move,
                quiet_move,
                penalty,
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
