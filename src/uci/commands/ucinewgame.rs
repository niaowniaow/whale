use crate::uci::UciClient;

impl UciClient {
    pub(crate) fn run_ucinewgame(&mut self, _parameters: &[&str]) {
        // Stockfish uci.cpp:142-143 + engine.cpp:161-169: ucinewgame does NOT
        // touch the board (the GUI keeps its position); it stops any running
        // search and clears TT/heuristics. Cancelling first, then taking the
        // blocking lock, waits for the detached search thread to release the
        // guard (it holds it for the whole search) before clearing.
        self.precompute_cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(cancel) = &self.current_search {
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(cancel) = &self.current_search {
            // Brief grace period so a mid-search ucinewgame does not block the
            // UCI loop for long; the lock below still guarantees the clear
            // happens only after the search thread drops its guard.
            for _ in 0..200 {
                if cancel.load(std::sync::atomic::Ordering::Relaxed)
                    && self.search_state.try_lock().is_ok()
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        let mut state = self.search_state.lock().unwrap();
        state.tt.clear();
        state.reset_heuristics();
        self.is_ready = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::move_type::MoveType;
    use crate::common::moves::Move;
    use crate::common::square::Square;

    #[test]
    fn should_reset_program() {
        // NOTE (UCI wave fix): ucinewgame no longer resets the board — the
        // position belongs to the GUI (Stockfish uci.cpp:142-143). It clears
        // TT/heuristics and cancels any running search instead.
        let mut uci_client = UciClient::new();
        let fen = "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        uci_client.board = std::sync::Arc::new(std::sync::Mutex::new(
            crate::board::state::BoardState::parse_fen(fen),
        ));

        uci_client
            .search_state
            .lock()
            .unwrap()
            .move_ordering
            .add_killer_move(Move::new(Square::E2, Square::E3, MoveType::Quiet), 0);

        {
            let mut board = uci_client.board.lock().unwrap();
            board.history.save(
                crate::common::piece::Piece::None,
                Square::NoSquare,
                crate::common::castle::Castle::NONE,
                0,
                0,
            );
        }

        assert_ne!(
            *uci_client.board.lock().unwrap(),
            crate::board::state::BoardState::default()
        );
        assert!(
            !uci_client
                .search_state
                .lock()
                .unwrap()
                .move_ordering
                .is_move_heuristic_empty()
        );
        assert!(!uci_client.board.lock().unwrap().history.is_empty());

        uci_client.run_ucinewgame(&[]);

        // Board (and its history) is preserved; heuristics are cleared.
        assert_eq!(
            *uci_client.board.lock().unwrap(),
            crate::board::state::BoardState::parse_fen(fen)
        );
        assert!(
            uci_client
                .search_state
                .lock()
                .unwrap()
                .move_ordering
                .is_move_heuristic_empty()
        );
        assert!(!uci_client.board.lock().unwrap().history.is_empty());
    }

    #[test]
    fn should_be_ready_after_reset() {
        let mut uci_client = UciClient::new();

        uci_client.run_ucinewgame(&[]);

        assert!(uci_client.is_ready);
    }

    #[test]
    fn should_restore_ready_state_after_reset() {
        let mut uci_client = UciClient::new();
        uci_client.is_ready = true;

        assert!(uci_client.is_ready);

        uci_client.run_ucinewgame(&[]);

        assert!(uci_client.is_ready);
    }

    #[test]
    fn ucinewgame_cancels_search_and_precompute() {
        use std::sync::atomic::Ordering;

        let mut uci_client = UciClient::new();
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        uci_client.current_search = Some(std::sync::Arc::clone(&cancel));
        uci_client.precompute_cancel.store(false, Ordering::Relaxed);

        uci_client.run_ucinewgame(&[]);

        assert!(cancel.load(Ordering::Relaxed));
        assert!(uci_client.precompute_cancel.load(Ordering::Relaxed));
        assert!(uci_client.is_ready);
    }

    #[test]
    fn ucinewgame_without_search_still_clears_heuristics() {
        let mut uci_client = UciClient::new();
        assert!(uci_client.current_search.is_none());
        uci_client
            .search_state
            .lock()
            .unwrap()
            .move_ordering
            .add_killer_move(Move::new(Square::E2, Square::E3, MoveType::Quiet), 0);
        assert!(
            !uci_client
                .search_state
                .lock()
                .unwrap()
                .move_ordering
                .is_move_heuristic_empty()
        );

        uci_client.run_ucinewgame(&[]);

        assert!(
            uci_client
                .search_state
                .lock()
                .unwrap()
                .move_ordering
                .is_move_heuristic_empty()
        );
        assert!(uci_client.is_ready);
    }
}
