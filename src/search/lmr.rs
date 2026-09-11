#[inline(always)]
pub fn needs_reduction(
    depth: u8,
    number_of_legal_moves: usize,
    is_tactical: bool,
    in_check: bool,
) -> bool {
    if depth < 3 || number_of_legal_moves < 3 || is_tactical || in_check {
        return false;
    }

    true
}

#[inline(always)]
pub fn get_reduction(depth: u8, number_of_legal_moves: usize, is_pv_node: bool) -> u8 {
    let d = depth as f64;
    let m = number_of_legal_moves as f64;
    // TODO: tune
    let red = 0.5 + (d.ln() * m.ln() / 1.95);
    let red = red.round() as u8;
    let red = if is_pv_node { red.saturating_sub(1) } else { red };
    red.min(3).min(depth - 1)
}
