use crate::common::move_list::MoveList;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::uci::UciClient;

impl UciClient {
    pub(crate) fn run_position(&mut self, parameters: &[&str]) {
        self.precompute_cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if parameters.is_empty() {
            return;
        }

        match parameters[0] {
            "startpos" => {
                let moves = if parameters.len() > 2 && parameters[1] == "moves" {
                    &parameters[2..]
                } else {
                    &[][..]
                };
                self.parse_startpos(moves);
            }
            "fen" => {
                // FEN runs until the `moves` token (if any), not a hardcoded
                // 6 fields, so halfmove/fullmove-omitting GUIs keep working.
                let moves_idx = parameters[1..]
                    .iter()
                    .position(|&t| t == "moves")
                    .map(|i| i + 1);
                let fen_end = moves_idx.unwrap_or(parameters.len());
                if fen_end <= 1 {
                    return;
                }
                let fen = parameters[1..fen_end].join(" ");
                let moves = match moves_idx {
                    Some(i) if i + 1 < parameters.len() => &parameters[i + 1..],
                    _ => &[][..],
                };
                self.parse_fen(&fen, moves);
            }
            _ => {}
        }
    }

    fn parse_fen(&mut self, fen: &str, moves: &[&str]) {
        *self.board.lock().unwrap() = crate::board::state::BoardState::parse_fen(fen);
        self.parse_moves(moves);
        self.is_ready = true;
    }

    fn parse_startpos(&mut self, moves: &[&str]) {
        *self.board.lock().unwrap() = crate::board::state::BoardState::default();
        self.parse_moves(moves);
        self.is_ready = true;
    }

    fn parse_moves(&mut self, moves: &[&str]) {
        for &move_string in moves {
            // Illegal/unparseable moves are skipped silently: stdout must stay
            // valid UCI (Reckless uci.rs silently ignores them too).
            let move_obj = match Move::parse_long_algebraic(move_string) {
                Some(m) => m,
                None => return,
            };

            let found_move = self.find_move_from_move_list(move_obj);
            if found_move == Move::NO_MOVE {
                return;
            }

            self.board.lock().unwrap().make_move(found_move);
        }
    }

    fn find_move_from_move_list(&mut self, move_obj: Move) -> Move {
        let mut board = self.board.lock().unwrap();
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);

        for m in move_list.iter() {
            if m.mv.source == move_obj.source
                && m.mv.target == move_obj.target
                && (move_obj.move_type == MoveType::Quiet
                    || ((m.mv.move_type.value() & !8) == move_obj.move_type.value()))
            {
                let candidate = m.mv;
                board.make_move(candidate);
                let illegal = board.is_in_check(board.side_to_move.other());
                board.unmake_move(candidate);
                if !illegal {
                    return candidate;
                }
            }
        }

        Move::NO_MOVE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;

    fn find_move_from_move_list(board: &mut BoardState, move_obj: Move) -> Move {
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        for m in move_list.iter() {
            if m.mv.source == move_obj.source
                && m.mv.target == move_obj.target
                && (move_obj.move_type == MoveType::Quiet
                    || ((m.mv.move_type.value() & !8) == move_obj.move_type.value()))
            {
                return m.mv;
            }
        }
        Move::NO_MOVE
    }

    #[test]
    fn should_set_position_from_fen() {
        let mut uci_client = UciClient::new();
        let fen = "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

        uci_client.run_position(&[
            "fen",
            "rnbqkb1r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "w",
            "KQkq",
            "-",
            "0",
            "1",
        ]);

        assert_eq!(
            BoardState::parse_fen(fen),
            *uci_client.board.lock().unwrap()
        );
    }

    #[test]
    fn should_set_position_to_start_pos() {
        let mut uci_client = UciClient::new();

        uci_client.run_position(&["startpos"]);

        assert_eq!(BoardState::default(), *uci_client.board.lock().unwrap());
    }

    #[test]
    fn should_set_position_to_start_pos_and_apply_moves() {
        let mut uci_client = UciClient::new();

        uci_client.run_position(&["startpos", "moves", "e2e4", "e7e5"]);

        let mut expected_state = BoardState::default();
        let white_move = find_move_from_move_list(
            &mut expected_state,
            Move::parse_long_algebraic("e2e4").unwrap(),
        );
        expected_state.make_move(white_move);
        let black_move = find_move_from_move_list(
            &mut expected_state,
            Move::parse_long_algebraic("e7e5").unwrap(),
        );
        expected_state.make_move(black_move);

        assert_eq!(expected_state, *uci_client.board.lock().unwrap());
    }
}
