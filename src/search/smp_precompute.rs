use crate::board::state::BoardState;
use crate::common::constants::ASPIRATION_WINDOW_MARGIN;
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::tt::TranspositionTable;
use crate::search::negamax;
use crate::search::pv_table::PvTable;
use crate::search::search_state::SearchState;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

pub fn select_speculative_replies(
    board: &BoardState,
    tt: &TranspositionTable,
    max_count: usize,
) -> Vec<Move> {
    let mut move_list = MoveList::new();
    board.generate_moves(&mut move_list);

    let mut scored_moves: Vec<(Move, i32)> = Vec::with_capacity(move_list.len());
    let tt_move = tt.probe(board.board_hash).map(|e| e.best_move);

    for i in 0..move_list.len() {
        let m = move_list[i].mv;
        if !board.is_legal(m) {
            continue;
        }

        let mut score = 0;
        if Some(m) == tt_move {
            score += 100_000;
        }
        if m.is_capture() {
            let see_val = board.see(m);
            if see_val >= 0 {
                score += 50_000 + see_val as i32;
            } else {
                score += 1_000;
            }
        }
        if m.is_promotion() {
            score += 40_000;
        }

        scored_moves.push((m, score));
    }

    scored_moves.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
    scored_moves
        .into_iter()
        .take(max_count)
        .map(|(m, _)| m)
        .collect()
}

pub fn run_precomputation(
    mut board: BoardState,
    best_move: Move,
    search_state_mutex: Arc<Mutex<SearchState>>,
    cancel_token: Arc<AtomicBool>,
    max_depth: u8,
) {
    if !board.is_legal(best_move) {
        return;
    }

    board.make_move(best_move);

    let replies = {
        let state_guard = search_state_mutex.lock().unwrap();
        select_speculative_replies(&board, &state_guard.tt, 3)
    };

    let mut local_search_state = {
        let state_guard = search_state_mutex.lock().unwrap();
        state_guard.clone_for_worker(1)
    };

    for reply in replies {
        if cancel_token.load(Ordering::Relaxed) {
            break;
        }

        let mut reply_board = board.clone();
        if !reply_board.is_legal(reply) {
            continue;
        }
        reply_board.make_move(reply);

        let mut pv_table = PvTable::new();
        let mut previous_pv = Vec::new();

        for current_depth in 1..=max_depth {
            if cancel_token.load(Ordering::Relaxed) {
                break;
            }

            let alpha = -ASPIRATION_WINDOW_MARGIN;
            let beta = ASPIRATION_WINDOW_MARGIN;

            let score = negamax::search(
                &mut reply_board,
                current_depth,
                alpha,
                beta,
                &cancel_token,
                &previous_pv,
                &mut pv_table,
                &mut local_search_state,
            );

            if cancel_token.load(Ordering::Relaxed) {
                break;
            }

            let line = pv_table.line().to_vec();
            if !line.is_empty() {
                previous_pv = line;
            }

            if score <= alpha || score >= beta {
                negamax::search(
                    &mut reply_board,
                    current_depth,
                    -30_000,
                    30_000,
                    &cancel_token,
                    &previous_pv,
                    &mut pv_table,
                    &mut local_search_state,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_select_speculative_replies_count() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let tt = TranspositionTable::new(1024);
        let replies = select_speculative_replies(&board, &tt, 3);
        assert_eq!(replies.len(), 3);
    }

    #[test]
    fn test_select_speculative_replies_prioritizes_tt() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let tt = TranspositionTable::new(1024);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let test_move = move_list[0].mv;

        tt.submit_entry(
            board.board_hash,
            100,
            4,
            test_move,
            crate::common::tt::TranspositionEntryType::Exact,
        );

        let replies = select_speculative_replies(&board, &tt, 3);
        assert!(!replies.is_empty());
        assert_eq!(replies[0], test_move);
    }

    #[test]
    fn test_run_precomputation_cancels_immediately() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let test_move = move_list[0].mv;

        let state = Arc::new(Mutex::new(SearchState::new()));
        let cancel = Arc::new(AtomicBool::new(true));

        run_precomputation(board, test_move, state, cancel, 4);
    }
}
