use crate::board::state::BoardState;
use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};

// TODO: tune conditions and reduction

#[inline(always)]
pub fn can_prune(
    is_pv_node: bool,
    board_state: &BoardState,
    allow_null_move: bool,
    depth: u8,
    in_check: bool,
    static_eval: i16,
    beta: i16,
) -> bool {
    let side = board_state.side_to_move;
    let has_non_pawn_material = board_state.has_non_pawn_material(side);

    let mate_bound = MAX_CENTIPAWN_EVAL - MAX_PLY as i16;
    allow_null_move
        && !is_pv_node
        && !in_check
        && depth >= 2
        && beta.abs() < mate_bound
        && static_eval >= beta
        && has_non_pawn_material
}

#[inline(always)]
pub fn get_reduction(depth: u8, params: &crate::search::search_state::SearchParameters) -> u8 {
    params.nmp_base + depth / params.nmp_depth_div
}
