use crate::board::state::BoardState;
use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::search::alp::{AlpFeatures, AlpModel};

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
    get_reduction_with_margin(depth, params, momentum, 0)
}

#[inline(always)]
pub fn get_reduction_with_margin(
    depth: u8,
    params: &crate::search::search_state::SearchParameters,
    momentum: i16,
    eval_margin: i16,
) -> u8 {
    let mut red = params.nmp_base + depth / params.nmp_depth_div;
    if momentum > 120 && depth >= 6 {
        red += 1;
    }
    if eval_margin > 200 {
        let bonus = ((eval_margin - 200) / 200).clamp(0, 2) as u8;
        red = red.saturating_add(bonus);
    }
    red
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    const BARE_KINGS: &str = "8/8/8/4k3/8/8/4K3/8 w - - 0 1";

    #[test]
    fn guards_reject_all_preconditions() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let _ = can_prune(false, &board, true, 3, false, 500, 0, 0);
        assert!(!can_prune(true, &board, true, 3, false, 500, 0, 0));
        assert!(!can_prune(false, &board, false, 3, false, 500, 0, 0));
        assert!(!can_prune(false, &board, true, 3, true, 500, 0, 0));
        assert!(!can_prune(false, &board, true, 1, false, 500, 0, 0));
        assert!(!can_prune(
            false,
            &board,
            true,
            3,
            false,
            500,
            MAX_CENTIPAWN_EVAL,
            0
        ));
        assert!(!can_prune(false, &board, true, 3, false, 500, 0, -151));
        let bare = BoardState::parse_fen(BARE_KINGS);
        assert!(!can_prune(false, &bare, true, 3, false, 500, 0, 0));
    }

    #[test]
    fn static_eval_below_beta_with_momentum_margin() {
        let board = BoardState::parse_fen(STARTING_FEN);
        assert!(!can_prune(false, &board, true, 4, false, 100, 100, -70));
        assert!(!can_prune(false, &board, true, 4, false, 139, 100, -70));
        assert!(!can_prune(false, &board, true, 4, false, 99, 100, 0));
    }

    #[test]
    fn matches_alp_verdict_on_sweep() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut saw_false = false;
        for depth in [2u8, 4, 6, 8] {
            for margin in [-600i32, -100, 0, 200, 600, 1500] {
                let beta = 100i16;
                let static_eval = beta.saturating_add(margin.clamp(-30000, 30000) as i16);
                let got = can_prune(false, &board, true, depth, false, static_eval, beta, 0);
                let expect = AlpModel::should_prune(
                    &AlpFeatures {
                        eval_margin: (static_eval as i32 - beta as i32).clamp(-32768, 32767),
                        depth: depth as i32,
                        move_index: 0,
                        is_null_move: true,
                        is_capture: false,
                        is_pv: false,
                        in_check: false,
                        history_score: 0,
                        momentum: 0,
                    },
                    30,
                ) && static_eval >= beta;
                assert_eq!(got, expect);
                if !got {
                    saw_false = true;
                }
            }
        }
        assert!(saw_false);
    }

    #[test]
    fn reduction_adds_momentum_bonus_only_when_deep() {
        let params = crate::search::search_state::SearchParameters::default();
        let base = params.nmp_base + 6 / params.nmp_depth_div;
        assert_eq!(get_reduction(6, &params, 121), base + 1);
        assert_eq!(get_reduction(6, &params, 120), base);
        assert_eq!(
            get_reduction(5, &params, 200),
            params.nmp_base + 5 / params.nmp_depth_div
        );
        assert_eq!(
            get_reduction(2, &params, 0),
            params.nmp_base + 2 / params.nmp_depth_div
        );
    }

    #[test]
    fn reduction_scales_with_eval_margin() {
        let params = crate::search::search_state::SearchParameters::default();
        let base = params.nmp_base + 6 / params.nmp_depth_div;
        // Eval margin <= 200 adds no bonus
        assert_eq!(get_reduction_with_margin(6, &params, 0, 150), base);
        // Eval margin > 200 adds bonus
        assert_eq!(get_reduction_with_margin(6, &params, 0, 400), base + 1);
        assert_eq!(get_reduction_with_margin(6, &params, 0, 800), base + 2);
    }
}
