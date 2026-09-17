use crate::board::state::BoardState;
use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::search::alp::{AlpFeatures, AlpModel};

// TODO: tune conditions and reduction

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn can_prune(
    is_pv_node: bool,
    board_state: &BoardState,
    allow_null_move: bool,
    depth: u8,
    in_check: bool,
    static_eval: i16,
    beta: i16,
    momentum: i16,
) -> bool {
    let side = board_state.side_to_move;
    let has_non_pawn_material = board_state.has_non_pawn_material(side);

    let mate_bound = MAX_CENTIPAWN_EVAL - MAX_PLY as i16;
    if !allow_null_move
        || is_pv_node
        || in_check
        || depth < 2
        || beta.abs() >= mate_bound
        || !has_non_pawn_material
        || momentum < -150
    {
        return false;
    }

    let eval_margin = if momentum < -60 { 40 } else { 0 };
    if static_eval < beta.saturating_add(eval_margin) {
        return false;
    }

    let alp_features = AlpFeatures {
        eval_margin: (static_eval as i32 - beta as i32).clamp(-32768, 32767),
        depth: depth as i32,
        move_index: 0,
        is_null_move: true,
        is_capture: false,
        is_pv: is_pv_node,
        in_check,
        history_score: 0,
        momentum: momentum as i32,
    };

    AlpModel::should_prune(&alp_features, 30)
}

#[inline(always)]
pub fn get_reduction(
    depth: u8,
    params: &crate::search::search_state::SearchParameters,
    momentum: i16,
) -> u8 {
    let mut red = params.nmp_base + depth / params.nmp_depth_div;
    if momentum > 120 && depth >= 6 {
        red += 1;
    }
    red
}
