use crate::common::moves::Move;

#[inline(always)]
pub fn should_run_coarse_pass(
    depth: u8,
    is_pv_node: bool,
    in_check: bool,
    tt_best: Option<Move>,
    excluded_move: Option<Move>,
) -> bool {
    depth >= 7 && !is_pv_node && !in_check && tt_best.is_none() && excluded_move.is_none()
}

#[inline(always)]
pub fn get_coarse_depth(depth: u8) -> u8 {
    depth.saturating_sub(3).max(1)
}

#[inline(always)]
pub fn should_prune_coarse_quiet(
    coarse_failed_low: bool,
    number_of_legal_moves: usize,
    is_tactical: bool,
    depth: u8,
) -> bool {
    coarse_failed_low && !is_tactical && number_of_legal_moves >= 8 && depth <= 5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_run_coarse_pass() {
        assert!(should_run_coarse_pass(7, false, false, None, None));
        assert!(should_run_coarse_pass(10, false, false, None, None));
        assert!(!should_run_coarse_pass(6, false, false, None, None));
        assert!(!should_run_coarse_pass(8, true, false, None, None));
        assert!(!should_run_coarse_pass(8, false, true, None, None));
    }

    #[test]
    fn test_coarse_depth_calculation() {
        assert_eq!(get_coarse_depth(7), 4);
        assert_eq!(get_coarse_depth(10), 7);
        assert_eq!(get_coarse_depth(2), 1);
    }

    #[test]
    fn test_should_prune_coarse_quiet() {
        assert!(should_prune_coarse_quiet(true, 8, false, 4));
        assert!(!should_prune_coarse_quiet(false, 8, false, 4));
        assert!(!should_prune_coarse_quiet(true, 5, false, 4));
        assert!(!should_prune_coarse_quiet(true, 8, true, 4));
    }
}
