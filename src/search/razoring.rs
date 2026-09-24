#[inline(always)]
pub fn margin(depth: u8, base: i16) -> i16 {
    base + depth as i16 * 120
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn can_razor(
    is_pv_node: bool,
    in_check: bool,
    depth: u8,
    static_eval: i16,
    alpha: i16,
    beta_is_mate: bool,
    excluded: bool,
) -> bool {
    if is_pv_node || in_check || excluded || beta_is_mate {
        return false;
    }
    if depth != 1 {
        return false;
    }
    static_eval + margin(depth, 350) <= alpha
}

#[inline(always)]
pub fn should_razor_with_margin(
    is_pv_node: bool,
    in_check: bool,
    depth: u8,
    static_eval: i16,
    alpha: i16,
    beta_is_mate: bool,
    excluded: bool,
    razor_base: i16,
) -> bool {
    if is_pv_node || in_check || excluded || beta_is_mate {
        return false;
    }
    if depth != 1 {
        return false;
    }
    static_eval + margin(depth, razor_base) <= alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_grows_with_depth() {
        assert!(margin(2, 350) < margin(4, 350));
        assert_eq!(margin(1, 350), 470);
    }

    #[test]
    fn pv_and_check_never_razor() {
        assert!(!can_razor(true, false, 1, -1000, 0, false, false));
        assert!(!can_razor(false, true, 1, -1000, 0, false, false));
        assert!(!can_razor(false, false, 2, -1000, 0, false, false));
        assert!(!can_razor(false, false, 5, -1000, 0, false, false));
        assert!(can_razor(false, false, 1, -1000, 0, false, false));
        assert!(!can_razor(false, false, 1, 0, 0, false, false));
    }

    #[test]
    fn param_gate_respects_custom_base() {
        assert!(should_razor_with_margin(
            false, false, 1, -1000, 0, false, false, 200
        ));
        assert!(!should_razor_with_margin(
            false, false, 1, 0, 0, false, false, 200
        ));
    }
}
