#[inline(always)]
pub fn reduction(
    is_pv_node: bool,
    cut_node: bool,
    depth: u8,
    has_tt_move: bool,
    excluded: bool,
) -> u8 {
    if excluded || has_tt_move {
        return 0;
    }
    if is_pv_node {
        if depth >= 8 {
            return 2;
        }
        if depth >= 5 {
            return 1;
        }
        return 0;
    }
    if cut_node && depth >= 7 {
        return 1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pv_gets_reduction_without_tt_move() {
        assert_eq!(reduction(true, false, 4, false, false), 0);
        assert_eq!(reduction(true, false, 5, false, false), 1);
        assert_eq!(reduction(true, false, 8, false, false), 2);
        assert_eq!(reduction(true, false, 9, true, false), 0);
        assert_eq!(reduction(true, false, 9, false, true), 0);
    }

    #[test]
    fn cut_node_needs_depth_7() {
        assert_eq!(reduction(false, true, 6, false, false), 0);
        assert_eq!(reduction(false, true, 7, false, false), 1);
        assert_eq!(reduction(false, false, 9, false, false), 0);
    }
}
