use crate::common::constants::MAX_SEARCH_DEPTH;

pub const MAX_REDUCTION: u8 = 3;

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
                    *cell = rounded.clamp(0, (d as i32 - 1).min(MAX_REDUCTION as i32)) as u8;
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
    }

    reduction.clamp(0, query.depth as i32 - 1) as u8
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
    red.min(3).min(depth - 1)
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
        };
        let improving_query = LmrQuery {
            is_improving: true,
            ..normal_query
        };
        let r_normal = compute_reduction(&normal_query, &table, &divisors);
        let r_improving = compute_reduction(&improving_query, &table, &divisors);
        assert!(r_improving <= r_normal);
    }
}
