use super::context::SearchContext;
use super::core::search_internal;
use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn search_deeper(
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
