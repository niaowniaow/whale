use super::*;

pub(super) fn update_history_stats(
    board_state: &BoardState,
    search_state: &mut SearchState,
    best_move: Move,
    depth: u8,
    previous_move: Option<Move>,
    tried_quiets: &[Move],
) {
    let bonus = (133 * depth as i32 - 81).clamp(0, 1487);
    let malus = (968 * depth as i32 - 235).clamp(0, 2244);
    let piece = board_state.get_piece_on(best_move.source);
    let side = board_state.side_to_move;

    if piece >= 0 {
        let quiet_bonus = bonus * 899 / 1024;
        search_state
            .move_ordering
            .update_history(piece as usize, best_move, quiet_bonus);
        search_state
            .move_ordering
            .update_quiet_history(side, best_move, quiet_bonus);
        update_continuation(
            search_state,
            board_state,
            previous_move,
            best_move,
            quiet_bonus,
        );
    }

    let mut actual_malus = malus * 1159 / 1024;
    for &quiet_move in tried_quiets {
        if quiet_move != best_move {
            actual_malus = actual_malus * 921 / 1024;
            let q_piece = board_state.get_piece_on(quiet_move.source);
            if q_piece >= 0 {
                search_state.move_ordering.update_history(
                    q_piece as usize,
                    quiet_move,
                    -actual_malus,
                );
                search_state
                    .move_ordering
                    .update_quiet_history(side, quiet_move, -actual_malus);
                update_continuation(
                    search_state,
                    board_state,
                    previous_move,
                    quiet_move,
                    -actual_malus,
                );
            }
        }
    }
}

fn update_continuation(
    search_state: &mut SearchState,
    board_state: &BoardState,
    previous_move: Option<Move>,
    move_obj: Move,
    bonus: i32,
) {
    if let Some(previous_move) = previous_move {
        let prev_target = previous_move.target as usize;
        if prev_target < crate::common::constants::SQUARES {
            let previous_piece = board_state.piece_mapping[prev_target];
            if previous_piece != Piece::None {
                search_state.move_ordering.update_continuation_history(
                    previous_piece,
                    previous_move.target,
                    move_obj,
                    bonus,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn beta_cutoff(
    score: i16,
    move_obj: Move,
    ply: usize,
    board_state: &BoardState,
    depth: u8,
    previous_move: Option<Move>,
    search_state: &mut SearchState,
    cancellation_token: &AtomicBool,
    tried_quiets: &[Move],
    tried_captures: &[Move],
    excluded_move: Option<Move>,
) -> i16 {
    if cancellation_token.load(Ordering::Relaxed) {
        return score;
    }
    if excluded_move.is_some() {
        return score;
    }
    if search_state.tt_store_allowed(ply as u8) {
        search_state.tt.submit_entry(
            board_state.board_hash,
            tt::TranspositionTable::adjust_score(score, ply as i32, board_state.half_move_clock),
            depth,
            move_obj,
            TranspositionEntryType::Beta,
        );
    }

    if !move_obj.is_capture() {
        search_state.move_ordering.add_killer_move(move_obj, ply);

        update_history_stats(
            board_state,
            search_state,
            move_obj,
            depth,
            previous_move,
            tried_quiets,
        );

        if let Some(prev_mv) = previous_move {
            let prev_target = prev_mv.target as usize;
            if prev_target < crate::common::constants::SQUARES {
                let prev_side = board_state.side_to_move.other();
                let prev_piece = board_state.piece_mapping[prev_target];
                if prev_piece != Piece::None {
                    search_state.move_ordering.add_counter_move(
                        prev_side,
                        prev_piece,
                        prev_mv.target,
                        move_obj,
                    );
                }
            }
        }
    } else {
        let src = move_obj.source as usize;
        let tgt = move_obj.target as usize;
        if src < crate::common::constants::SQUARES && tgt < crate::common::constants::SQUARES {
            let piece = board_state.piece_mapping[src];
            if piece != Piece::None {
                let moved_piece = board_state.get_piece_on(move_obj.source);
                let captured_piece = if move_obj.move_type == MoveType::EnPassant {
                    Piece::Pawn
                } else {
                    board_state.piece_mapping[tgt]
                };
                if moved_piece >= 0 && captured_piece != Piece::None {
                    let bonus = (133 * depth as i32 - 81).clamp(0, 1487);
                    let malus = (968 * depth as i32 - 235).clamp(0, 2244);
                    search_state.move_ordering.update_capture_history(
                        moved_piece as usize,
                        move_obj.target,
                        captured_piece,
                        bonus * 1427 / 1024,
                    );
                    for &prev_cap in tried_captures {
                        let prev_tgt = prev_cap.target as usize;
                        if prev_tgt < crate::common::constants::SQUARES {
                            let prev_moved = board_state.get_piece_on(prev_cap.source);
                            let prev_captured = if prev_cap.move_type == MoveType::EnPassant {
                                Piece::Pawn
                            } else {
                                board_state.piece_mapping[prev_tgt]
                            };
                            if prev_moved >= 0 && prev_captured != Piece::None {
                                search_state.move_ordering.update_capture_history(
                                    prev_moved as usize,
                                    prev_cap.target,
                                    prev_captured,
                                    -malus * 1332 / 1024,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    score
}
