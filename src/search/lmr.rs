use std::sync::OnceLock;

static LMR_TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();

pub fn get_lmr_table() -> &'static [[u8; 64]; 64] {
    LMR_TABLE.get_or_init(|| {
        let mut table = [[0u8; 64]; 64];
        for d in 1..64 {
            for m in 1..64 {
                let d_f = d as f64;
                let m_f = m as f64;
                let r = 0.5 + (d_f.ln() * m_f.ln() / 1.95);
                table[d][m] = r.round().max(0.0) as u8;
            }
        }
        table
    })
}

#[inline(always)]
pub fn needs_reduction(
    depth: u8,
    number_of_legal_moves: usize,
    is_tactical: bool,
    in_check: bool,
) -> bool {
    !(depth < 3 || number_of_legal_moves < 3 || is_tactical || in_check)
}

#[inline(always)]
pub fn get_reduction(
    depth: u8,
    number_of_legal_moves: usize,
    is_pv_node: bool,
    improving: bool,
    history_score: i32,
) -> u8 {
    let table = get_lmr_table();
    let d_idx = (depth as usize).min(63);
    let m_idx = number_of_legal_moves.min(63);
    let mut red = table[d_idx][m_idx] as i32;

    if is_pv_node {
        red -= 1;
    }

    if !improving {
        red += 1;
    }

    let history_bonus = history_score / 4096;
    red -= history_bonus;

    red.clamp(0, depth.saturating_sub(2) as i32) as u8
}
