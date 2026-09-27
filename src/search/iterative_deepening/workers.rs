use super::*;

pub(super) fn worker_search(
    mut board: BoardState,
    max_depth: u8,
    cancellation_token: &AtomicBool,
    mut search_state: SearchState,
    thread_id: usize,
) -> (u64, u64, u32) {
    let mut previous_pv = Vec::new();
    let mut pv_table = PvTable::new();

    let start_depth = if thread_id % 2 == 1 { 1 } else { 2 };
    for current_depth in start_depth..=max_depth {
        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if search_state.max_nodes > 0 && search_state.nodes >= search_state.max_nodes {
            break;
        }

        let score = negamax::search(
            &mut board,
            current_depth,
            i16::MIN + 1,
            i16::MAX - 1,
            cancellation_token,
            &previous_pv,
            &mut pv_table,
            &mut search_state,
        );

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        if search_state.max_nodes > 0 && search_state.nodes >= search_state.max_nodes {
            break;
        }
        let _ = score;
        let current_pv = pv_table.line().to_vec();
        if !current_pv.is_empty() {
            previous_pv = current_pv;
        }
    }
    (
        search_state.nodes,
        search_state.tbhits,
        search_state.seldepth,
    )
}

pub(super) fn search_split_root(
    board_state: &mut BoardState,
    depth: u8,
    cancellation_token: &AtomicBool,
    debug_mode: &mut bool,
    search_state: &mut SearchState,
    num_threads: usize,
) {
    let timer = Instant::now();
    let mut all = MoveList::new();
    board_state.generate_moves(&mut all);
    let nt = NodeThreats::compute(board_state);
    let filter = !search_state.searchmoves.is_empty();
    let mut root_moves = Vec::new();
    for i in 0..all.len() {
        let m = all[i].mv;
        if filter && !search_state.searchmoves.contains(&m) {
            continue;
        }
        if board_state.is_legal_with(m, nt.checkers, nt.pinned) {
            root_moves.push(m);
        }
    }
    if root_moves.is_empty() {
        return;
    }
    let depth = depth.max(1);
    let parts = split_root::partition(&root_moves, num_threads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for (tid, subset) in parts.into_iter().enumerate() {
            if subset.is_empty() {
                continue;
            }
            let mut board = board_state.clone();
            let mut state = search_state.clone_for_worker(tid);
            let cancel = cancellation_token;
            handles.push(
                std::thread::Builder::new()
                    .stack_size(SEARCH_THREAD_STACK_SIZE)
                    .spawn_scoped(s, move || {
                        let (bm, bs) = split_root::search_subset(
                            &mut board, &subset, depth, cancel, &mut state,
                        );
                        (bm, bs, state.nodes, state.tbhits, state.seldepth)
                    })
                    .unwrap(),
            );
        }
        let mut best_move = Move::NO_MOVE;
        let mut best_score = i16::MIN + 1;
        let mut nodes = 0u64;
        let mut tbhits = 0u64;
        let mut seldepth = 0u32;
        for h in handles {
            if let Ok((bm, bs, n, tb, sd)) = h.join() {
                nodes += n;
                tbhits += tb;
                seldepth = seldepth.max(sd);
                if bm != Move::NO_MOVE && bs > best_score {
                    best_score = bs;
                    best_move = bm;
                }
            }
        }
        search_state.nodes += nodes;
        search_state.tbhits += tbhits;
        search_state.seldepth = search_state.seldepth.max(seldepth);
        if best_move != Move::NO_MOVE {
            search_state.best_move = best_move;
            search_state.score = best_score;
        }
    });
    if *debug_mode && search_state.best_move != Move::NO_MOVE {
        let bm = search_state.best_move;
        let promo = bm
            .promotion_char()
            .map(|c| c.to_string())
            .unwrap_or_default();
        let time_ms = timer.elapsed().as_millis().max(1);
        let nps = (search_state.nodes as f64 / time_ms as f64 * 1000.0) as u64;
        println!(
            "info depth {} seldepth {} score {} nodes {} time {} nps {} pv {}{}{}",
            depth,
            search_state.seldepth,
            format_score(search_state.score),
            search_state.nodes,
            time_ms,
            nps,
            bm.source,
            bm.target,
            promo
        );
        println!("info string splitroot threads {}", num_threads);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}
