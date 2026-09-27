use super::*;

pub(super) fn search_excluded_root(
    board_state: &mut BoardState,
    depth: u8,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
    excluded: &[Move],
) -> Option<(Move, i16, Vec<Move>)> {
    let mut all = MoveList::new();
    board_state.generate_moves(&mut all);
    let nt = NodeThreats::compute(board_state);
    let saved = std::mem::take(&mut search_state.searchmoves);
    let uci_active = !saved.is_empty();
    let mut allowed = Vec::new();
    for i in 0..all.len() {
        let m = all[i].mv;
        if excluded.contains(&m) {
            continue;
        }
        if uci_active && !saved.contains(&m) {
            continue;
        }
        if board_state.is_legal_with(m, nt.checkers, nt.pinned) {
            allowed.push(m);
        }
    }
    if allowed.is_empty() {
        search_state.searchmoves = saved;
        return None;
    }
    search_state.searchmoves = allowed;
    let mut pv = PvTable::new();
    let score = negamax::search(
        board_state,
        depth,
        i16::MIN + 1,
        i16::MAX - 1,
        cancellation_token,
        &[],
        &mut pv,
        search_state,
    );
    let line = pv.line().to_vec();
    let bm = line.first().copied().unwrap_or(Move::NO_MOVE);
    search_state.searchmoves = saved;
    if cancellation_token.load(Ordering::Relaxed) || bm == Move::NO_MOVE {
        return None;
    }
    Some((bm, score, line))
}

pub(super) fn search_single_root(
    board_state: &mut BoardState,
    depth: u8,
    candidate: Move,
    cancellation_token: &AtomicBool,
    search_state: &mut SearchState,
) -> Option<(i16, Vec<Move>)> {
    let saved = std::mem::take(&mut search_state.searchmoves);
    search_state.searchmoves = vec![candidate];
    let mut pv = PvTable::new();
    let score = negamax::search(
        board_state,
        depth,
        i16::MIN + 1,
        i16::MAX - 1,
        cancellation_token,
        &[],
        &mut pv,
        search_state,
    );
    let line = pv.line().to_vec();
    search_state.searchmoves = saved;
    if cancellation_token.load(Ordering::Relaxed) {
        return None;
    }
    if line.first().copied().unwrap_or(Move::NO_MOVE) != candidate {
        return None;
    }
    Some((score, line))
}
