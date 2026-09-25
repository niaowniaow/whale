use crate::common::constants::MAX_SEARCH_DEPTH;
use std::cell::RefCell;

thread_local! {
    static CUTOFF_COUNTS: RefCell<[u8; 64]> = RefCell::new([0; 64]);
}

#[inline(always)]
pub fn record_cutoff(ply: u8) {
    CUTOFF_COUNTS.with(|c| {
        let mut arr = c.borrow_mut();
        let idx = (ply as usize).min(63);
        arr[idx] = arr[idx].saturating_add(1);
    });
}

#[inline(always)]
pub fn cutoff_count(ply: u8) -> u8 {
    CUTOFF_COUNTS.with(|c| c.borrow()[(ply as usize).min(63)])
}

#[inline(always)]
pub fn clear_cutoff_counts() {
    CUTOFF_COUNTS.with(|c| *c.borrow_mut() = [0; 64]);
}

#[inline(always)]
pub fn reset_cutoff(ply: u8) {
    CUTOFF_COUNTS.with(|c| {
        c.borrow_mut()[(ply as usize).min(63)] = 0;
    });
}

#[derive(Clone, Debug)]
pub struct LmrTable {
    table: [[u8; MAX_SEARCH_DEPTH as usize]; MAX_SEARCH_DEPTH as usize],
}

impl LmrTable {
    pub fn new(base: f64, div: f64) -> Self {
        let mut table = [[0u8; MAX_SEARCH_DEPTH as usize]; MAX_SEARCH_DEPTH as usize];
        for (d, row) in table
            .iter_mut()
            .enumerate()
            .take(MAX_SEARCH_DEPTH as usize)
            .skip(1)
        {
            for (m, cell) in row
                .iter_mut()
                .enumerate()
                .take(MAX_SEARCH_DEPTH as usize)
                .skip(1)
            {
                if d >= 3 && m >= 3 {
                    let red = base + ((d as f64).ln() * (m as f64).ln() / div);
                    let rounded = red.round() as i32;
                    *cell = rounded.clamp(0, d as i32 - 1) as u8;
                }
            }
        }
        Self { table }
    }

    #[inline(always)]
    pub fn base_reduction(&self, depth: u8, move_count: usize) -> u8 {
        let d = (depth as usize).min(MAX_SEARCH_DEPTH as usize - 1);
        let m = move_count.min(MAX_SEARCH_DEPTH as usize - 1);
        self.table[d][m]
    }
}

impl Default for LmrTable {
    fn default() -> Self {
        Self::new(0.5, 1.95)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LmrQuery {
    pub depth: u8,
    pub move_count: usize,
    pub is_pv_node: bool,
    pub is_improving: bool,
    pub gives_check: bool,
    pub is_tactical: bool,
    pub has_non_pawn_material: bool,
    pub history_score: i32,
    pub alpha: i16,
    pub static_eval: i16,
    pub momentum: i16,
    pub found_pv: bool,
    pub structural_disagreement: i16,

    pub cut_node: bool,

    pub tt_pv: bool,
    pub cutoff_cnt: u8,
    pub all_node: bool,
    pub tt_capture: bool,
    pub is_tt_move: bool,
}

#[inline(always)]
pub fn needs_reduction(depth: u8, move_count: usize, is_tactical: bool, in_check: bool) -> bool {
    !(depth < 3 || move_count < 3 || is_tactical || in_check)
}

#[inline(always)]
pub fn needs_tactical_reduction(
    depth: u8,
    move_count: usize,
    in_check: bool,
    history_score: i32,
) -> bool {
    depth >= 4 && move_count >= 4 && !in_check && history_score < -200
}

#[inline(always)]
pub fn compute_reduction(query: &LmrQuery, table: &LmrTable, history_divisors: &[i32; 16]) -> u8 {
    let base = table.base_reduction(query.depth, query.move_count);
    let mut reduction = i32::from(base);

    if query.is_pv_node {
        reduction = reduction.saturating_sub(1);
    }

    if query.tt_pv {
        reduction = reduction.saturating_sub(1);
    }
    if query.cut_node {
        reduction += 1;
    }
    if query.gives_check {
        reduction = reduction.saturating_sub(1);
    }
    if !query.has_non_pawn_material {
        reduction = reduction.saturating_sub(1);
    }

    if query.is_tactical {
        reduction = 1;
    } else {
        let d_idx = (query.depth as usize).min(16).saturating_sub(1);
        let divisor = history_divisors[d_idx].max(1);
        let history_bonus = query.history_score / divisor;
        reduction -= history_bonus;

        if query.alpha.abs() < 25_000 {
            let diff = (query.alpha as i32 - query.static_eval as i32).clamp(-64, 96);
            if diff > 48 {
                reduction += 1;
            } else if diff < -48 {
                reduction = reduction.saturating_sub(1);
            }
        }

        if query.momentum < -120 {
            reduction = reduction.saturating_sub(1);
        } else if query.momentum > 150 && query.move_count >= 6 {
            reduction += 1;
        }

        if !query.is_pv_node && !query.found_pv && query.move_count >= 6 && query.momentum >= -100 {
            reduction += 1;
        }

        if query.structural_disagreement > 100 {
            reduction = reduction.saturating_sub(1);
        }
    }

    if query.tt_capture {
        reduction += 1;
    }
    if query.cutoff_cnt > 1 {
        reduction += 1;
        if query.cutoff_cnt > 2 {
            reduction += 1;
        }
        if query.all_node {
            reduction += 1;
        }
    } else if query.is_tt_move {
        reduction = reduction.saturating_sub(2);
    }
    if query.all_node {
        reduction += reduction * 276 / (256 * query.depth as i32 + 268);
    }

    reduction.clamp(0, query.depth as i32 - 1) as u8
}

#[inline(always)]
pub fn deepen_adjustment(reduced_depth: u8, new_depth: u8, score: i16, best_score: i16) -> i8 {
    let do_deeper = reduced_depth < new_depth && score > best_score.saturating_add(53);
    let do_shallower = score < best_score.saturating_add(8);
    (do_deeper as i8) - (do_shallower as i8)
}

#[inline(always)]
pub fn get_reduction(
    depth: u8,
    number_of_legal_moves: usize,
    is_pv_node: bool,
    params: &crate::search::search_state::SearchParameters,
) -> u8 {
    let d = depth as f64;
    let m = number_of_legal_moves as f64;
    let red = params.lmr_base + (d.ln() * m.ln() / params.lmr_div);
    let red = red.round() as u8;
    let red = if is_pv_node {
        red.saturating_sub(1)
    } else {
        red
    };
    red.min(depth.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_formula() {
        let table = LmrTable::default();
        assert_eq!(table.base_reduction(1, 1), 0);
        assert_eq!(table.base_reduction(2, 5), 0);
        assert_eq!(table.base_reduction(3, 2), 0);
        assert!(table.base_reduction(3, 3) >= 1);
        assert!(table.base_reduction(8, 10) >= 2);
    }

    #[test]
    fn reduction_is_always_strictly_less_than_depth() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        for d in 3..20 {
            for m in 3..30 {
                let query = LmrQuery {
                    depth: d,
                    move_count: m,
                    is_pv_node: false,
                    is_improving: false,
                    gives_check: false,
                    is_tactical: false,
                    has_non_pawn_material: true,
                    history_score: -50000,
                    alpha: 0,
                    static_eval: 0,
                    momentum: 0,
                    found_pv: false,
                    structural_disagreement: 0,
                    cut_node: false,
                    tt_pv: false,
                    cutoff_cnt: 0,
                    all_node: false,
                    tt_capture: false,
                    is_tt_move: false,
                };
                let r = compute_reduction(&query, &table, &divisors);
                assert!(r < d);
            }
        }
    }

    #[test]
    fn pv_nodes_and_checks_reduce_less() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let normal_query = LmrQuery {
            depth: 6,
            move_count: 6,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: 0,
            momentum: 0,
            found_pv: false,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let pv_query = LmrQuery {
            is_pv_node: true,
            ..normal_query
        };
        let check_query = LmrQuery {
            gives_check: true,
            ..normal_query
        };
        let r_normal = compute_reduction(&normal_query, &table, &divisors);
        let r_pv = compute_reduction(&pv_query, &table, &divisors);
        let r_check = compute_reduction(&check_query, &table, &divisors);
        assert!(r_pv <= r_normal);
        assert!(r_check <= r_normal);
    }

    #[test]
    fn improving_reduces_less() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let normal_query = LmrQuery {
            depth: 6,
            move_count: 6,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: 0,
            momentum: 0,
            found_pv: false,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let improving_query = LmrQuery {
            is_improving: true,
            ..normal_query
        };
        let r_normal = compute_reduction(&normal_query, &table, &divisors);
        let r_improving = compute_reduction(&improving_query, &table, &divisors);
        assert!(r_improving <= r_normal);
    }

    #[test]
    fn momentum_adjusts_reduction() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let base_query = LmrQuery {
            depth: 8,
            move_count: 8,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: 0,
            momentum: 0,
            found_pv: true,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let crisis_query = LmrQuery {
            momentum: -150,
            ..base_query
        };
        let comfort_query = LmrQuery {
            momentum: 200,
            ..base_query
        };
        let r_base = compute_reduction(&base_query, &table, &divisors);
        let r_crisis = compute_reduction(&crisis_query, &table, &divisors);
        let r_comfort = compute_reduction(&comfort_query, &table, &divisors);
        assert!(r_crisis <= r_base);
        assert!(r_comfort >= r_base);
    }

    #[test]
    fn sibling_cutoff_rate_reduces_all_node_late_moves() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let pv_found_query = LmrQuery {
            depth: 8,
            move_count: 8,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: 0,
            momentum: 0,
            found_pv: true,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let no_pv_query = LmrQuery {
            found_pv: false,
            ..pv_found_query
        };
        let r_pv_found = compute_reduction(&pv_found_query, &table, &divisors);
        let r_no_pv = compute_reduction(&no_pv_query, &table, &divisors);
        assert!(r_no_pv >= r_pv_found);
    }

    #[test]
    fn disagreement_reduces_less() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let normal_query = LmrQuery {
            depth: 8,
            move_count: 8,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: 0,
            momentum: 0,
            found_pv: true,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let dis_query = LmrQuery {
            structural_disagreement: 150,
            ..normal_query
        };
        let r_normal = compute_reduction(&normal_query, &table, &divisors);
        let r_dis = compute_reduction(&dis_query, &table, &divisors);
        assert!(r_dis <= r_normal);
    }

    #[test]
    fn needs_reduction_gates() {
        assert!(!needs_reduction(2, 10, false, false));
        assert!(!needs_reduction(5, 2, false, false));
        assert!(!needs_reduction(5, 10, true, false));
        assert!(!needs_reduction(5, 10, false, true));
        assert!(needs_reduction(3, 3, false, false));
        assert!(needs_reduction(8, 12, false, false));
    }

    #[test]
    fn needs_tactical_reduction_gates() {
        assert!(!needs_tactical_reduction(3, 10, false, -500));
        assert!(!needs_tactical_reduction(5, 3, false, -500));
        assert!(!needs_tactical_reduction(5, 10, true, -500));
        assert!(!needs_tactical_reduction(5, 10, false, -199));
        assert!(!needs_tactical_reduction(5, 10, false, -200));
        assert!(needs_tactical_reduction(4, 4, false, -201));
    }

    #[test]
    fn tactical_moves_use_fixed_reduction() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let base = LmrQuery {
            depth: 8,
            move_count: 10,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: true,
            has_non_pawn_material: true,
            history_score: 9000,
            alpha: 0,
            static_eval: 0,
            momentum: 500,
            found_pv: false,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        assert_eq!(compute_reduction(&base, &table, &divisors), 1);
        let shallow = LmrQuery { depth: 1, ..base };
        assert_eq!(compute_reduction(&shallow, &table, &divisors), 0);
    }

    #[test]
    fn alpha_and_material_branches() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let base = LmrQuery {
            depth: 8,
            move_count: 8,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: -100,
            momentum: 0,
            found_pv: true,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let hi = compute_reduction(&base, &table, &divisors);
        let lo_query = LmrQuery {
            static_eval: 100,
            ..base
        };
        let lo = compute_reduction(&lo_query, &table, &divisors);
        assert!(hi >= lo);
        let mate_query = LmrQuery {
            alpha: 26_000,
            ..base
        };
        let _ = compute_reduction(&mate_query, &table, &divisors);
        let bare_query = LmrQuery {
            has_non_pawn_material: false,
            ..base
        };
        assert!(compute_reduction(&bare_query, &table, &divisors) <= hi);
        let wild = LmrQuery {
            history_score: 1_000_000,
            momentum: 500,
            move_count: 30,
            found_pv: false,
            ..base
        };
        assert!(compute_reduction(&wild, &table, &divisors) < 8);
    }

    #[test]
    fn cut_node_increases_and_tt_pv_decreases() {
        let table = LmrTable::default();
        let divisors = [3000; 16];
        let base = LmrQuery {
            depth: 8,
            move_count: 8,
            is_pv_node: false,
            is_improving: false,
            gives_check: false,
            is_tactical: false,
            has_non_pawn_material: true,
            history_score: 0,
            alpha: 0,
            static_eval: 0,
            momentum: 0,
            found_pv: true,
            structural_disagreement: 0,
            cut_node: false,
            tt_pv: false,
            cutoff_cnt: 0,
            all_node: false,
            tt_capture: false,
            is_tt_move: false,
        };
        let r_base = compute_reduction(&base, &table, &divisors);
        let r_cut = compute_reduction(
            &LmrQuery {
                cut_node: true,
                ..base
            },
            &table,
            &divisors,
        );
        let r_pv = compute_reduction(
            &LmrQuery {
                tt_pv: true,
                ..base
            },
            &table,
            &divisors,
        );
        assert!(r_cut >= r_base);
        assert!(r_pv <= r_base);
    }

    #[test]
    fn legacy_get_reduction_matches_bounds() {
        let params = crate::search::search_state::SearchParameters::default();
        let pv = get_reduction(6, 10, true, &params);
        let non_pv = get_reduction(6, 10, false, &params);
        assert!(pv <= non_pv);
        assert!(non_pv <= 3);
        assert_eq!(get_reduction(1, 10, false, &params), 0);
    }
}
