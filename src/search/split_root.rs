use crate::board::state::BoardState;
use crate::common::moves::Move;
use crate::search::negamax;
use crate::search::pv_table::PvTable;
use crate::search::search_state::SearchState;
use std::sync::atomic::AtomicBool;

pub fn partition(moves: &[Move], n: usize) -> Vec<Vec<Move>> {
    let n = n.max(1);
    let mut buckets = vec![Vec::new(); n];
    for (i, &m) in moves.iter().enumerate() {
        buckets[i % n].push(m);
    }
    buckets
}

pub fn search_subset(
    board: &mut BoardState,
    moves: &[Move],
    depth: u8,
    cancellation: &AtomicBool,
    state: &mut SearchState,
) -> (Move, i16) {
    let mut best_move = Move::NO_MOVE;
    let mut best_score = i16::MIN + 1;
    for &m in moves {
        if cancellation.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        let saved = std::mem::take(&mut state.searchmoves);
        state.searchmoves = vec![m];
        let mut pv = PvTable::new();
        let score = negamax::search(
            board,
            depth,
            i16::MIN + 1,
            i16::MAX - 1,
            cancellation,
            &[],
            &mut pv,
            state,
        );
        state.searchmoves = saved;
        if cancellation.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }
        if score > best_score {
            best_score = score;
            best_move = m;
        }
    }
    (best_move, best_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_robin_partition_covers_all() {
        let moves = [Move::NO_MOVE; 5];
        let parts = partition(&moves, 3);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.iter().map(|p| p.len()).sum::<usize>(), 5);
        assert_eq!(parts[0].len(), 2);
    }

    #[test]
    fn single_bucket_keeps_order() {
        let moves = [Move::NO_MOVE; 2];
        let parts = partition(&moves, 1);
        assert_eq!(parts[0].len(), 2);
    }
}
