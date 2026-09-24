use crate::board::state::BoardState;
use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::search::alp::{AlpFeatures, AlpModel};
use std::cell::{Cell, RefCell};

thread_local! {
    static NMP_MIN_PLY: Cell<u8> = Cell::new(0);
    static PRIOR_FAIL_HIGH: RefCell<[u8; 64]> = RefCell::new([0; 64]);
}

#[inline(always)]
pub fn get_nmp_min_ply() -> u8 {
    NMP_MIN_PLY.with(|c| c.get())
}

#[inline(always)]
pub fn set_nmp_min_ply(v: u8) {
    NMP_MIN_PLY.with(|c| c.set(v))
}

#[inline(always)]
pub fn get_prior_fail_high(ply: u8) -> u8 {
    PRIOR_FAIL_HIGH.with(|c| c.borrow()[(ply as usize).min(63)])
}

#[inline(always)]
pub fn inc_prior_fail_high(ply: u8) {
    PRIOR_FAIL_HIGH.with(|c| {
        let mut arr = c.borrow_mut();
        let idx = (ply as usize).min(63);
        arr[idx] = arr[idx].saturating_add(1);
    })
}

#[inline(always)]
pub fn clear_nmp_state() {
    NMP_MIN_PLY.with(|c| c.set(0));
    PRIOR_FAIL_HIGH.with(|c| *c.borrow_mut() = [0; 64]);
}

#[inline(always)]
pub fn reset_prior_fail_high(ply: u8) {
    PRIOR_FAIL_HIGH.with(|c| {
        c.borrow_mut()[(ply as usize).min(63)] = 0;
    });
}

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
    cut_node: bool,
    ply: u8,
    is_improving: bool,
) -> bool {
    if !cut_node {
        return false;
    }
    if ply < get_nmp_min_ply() {
        return false;
    }
    if beta < -2000 {
        return false;
    }
    let prior = get_prior_fail_high(ply) as i32;
    let imp = if is_improving { 1 } else { 0 };
    let threshold = beta as i32 - 13 * depth as i32 - 47 * imp + 365;
    if static_eval as i32 + 50 * prior < threshold {
        return false;
    }
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
    let _ = params;
    let mut red = 7 + depth / 3 + ((eval_margin as i32 / 256).max(0).min(6) as u8);
    if momentum > 120 && depth >= 6 {
        red = red.saturating_add(1);
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
        clear_nmp_state();
        let board = BoardState::parse_fen(STARTING_FEN);
        let _ = can_prune(false, &board, true, 3, false, 500, 0, 0, true, 0, true);
        assert!(!can_prune(true, &board, true, 3, false, 500, 0, 0, true, 0, true));
        assert!(!can_prune(false, &board, false, 3, false, 500, 0, 0, true, 0, true));
        assert!(!can_prune(false, &board, true, 3, true, 500, 0, 0, true, 0, true));
        assert!(!can_prune(false, &board, true, 1, false, 500, 0, 0, true, 0, true));
        assert!(!can_prune(
            false,
            &board,
            true,
            3,
            false,
            500,
            MAX_CENTIPAWN_EVAL,
            0,
            true,
            0,
            true
        ));
        assert!(!can_prune(false, &board, true, 3, false, 500, 0, -151, true, 0, true));
        let bare = BoardState::parse_fen(BARE_KINGS);
        assert!(!can_prune(false, &bare, true, 3, false, 500, 0, 0, true, 0, true));
        assert!(!can_prune(false, &board, true, 3, false, 500, 0, 0, false, 0, true));
        assert!(!can_prune(false, &board, true, 3, false, 500, -3000, 0, true, 0, true));
        set_nmp_min_ply(4);
        assert!(!can_prune(false, &board, true, 6, false, 2000, 0, 0, true, 2, true));
        set_nmp_min_ply(0);
        assert!(!can_prune(false, &board, true, 6, false, 0, 500, 0, true, 0, false));
    }

    #[test]
    fn static_eval_below_beta_with_momentum_margin() {
        clear_nmp_state();
        let board = BoardState::parse_fen(STARTING_FEN);
        assert!(!can_prune(false, &board, true, 4, false, 100, 100, -70, true, 0, true));
        assert!(!can_prune(false, &board, true, 4, false, 139, 100, -70, true, 0, true));
        assert!(!can_prune(false, &board, true, 4, false, 99, 100, 0, true, 0, true));
    }

    #[test]
    fn matches_alp_verdict_on_sweep() {
        clear_nmp_state();
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut saw_false = false;
        for depth in [2u8, 4, 6, 8] {
            for margin in [-600i32, -100, 0, 200, 600, 1500] {
                let beta = 100i16;
                let static_eval = beta.saturating_add(margin.clamp(-30000, 30000) as i16);
                let got = can_prune(false, &board, true, depth, false, static_eval, beta, 0, true, 0, true);
                let alp = AlpModel::should_prune(
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
                let threshold = beta as i32 - 13 * depth as i32 - 47 + 365;
                let sf_gate = static_eval as i32 >= threshold;
                let expect = alp && sf_gate;
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
        let base = 7 + 6 / 3;
        assert_eq!(get_reduction(6, &params, 121), base + 1);
        assert_eq!(get_reduction(6, &params, 120), base);
        assert_eq!(get_reduction(5, &params, 200), 7 + 5 / 3);
        assert_eq!(get_reduction(2, &params, 0), 7 + 2 / 3);
    }

    #[test]
    fn reduction_scales_with_eval_margin() {
        let params = crate::search::search_state::SearchParameters::default();
        let base = 7 + 6 / 3;

        assert_eq!(get_reduction_with_margin(6, &params, 0, 150), base);

        assert_eq!(get_reduction_with_margin(6, &params, 0, 400), base + 1);
        assert_eq!(get_reduction_with_margin(6, &params, 0, 800), base + 3);
    }
}
