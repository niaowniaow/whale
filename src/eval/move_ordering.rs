use crate::board::state::BoardState;
use crate::common::constants::{MAX_PLY, PIECES, SIDES, SQUARES};
use crate::common::move_list::ScoredMove;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

pub struct MoveOrdering {
    pub killer_moves: [[Move; MAX_PLY]; 2],
    pub history_moves: [[i32; SQUARES]; PIECES * 2],
    pub quiet_history: [[[i16; SQUARES]; SQUARES]; SIDES],
    pub continuation_history: [[[i16; SQUARES]; SQUARES]; PIECES * 2],
    pub counter_moves: [[[Move; SQUARES]; PIECES]; SIDES],
    pub capture_history: [[[i16; PIECES]; SQUARES]; PIECES * 2],
}

#[rustfmt::skip]
const MVV_LVA: [[i32; 7]; 7] = [
    
    [ 15_000, 14_000, 13_000, 12_000, 11_000, 10_000, 0 ], 
    [ 25_000, 24_000, 23_000, 22_000, 21_000, 20_000, 0 ], 
    [ 35_000, 34_000, 33_000, 32_000, 31_000, 30_000, 0 ], 
    [ 45_000, 44_000, 43_000, 42_000, 41_000, 40_000, 0 ], 
    [ 55_000, 54_000, 53_000, 52_000, 51_000, 50_000, 0 ], 
    [ 65_000, 64_000, 63_000, 62_000, 61_000, 60_000, 0 ], 
    [ 0, 0, 0, 0, 0, 0, 0 ] 
];

impl MoveOrdering {
    pub fn new() -> Self {
        Self {
            killer_moves: [[Move::NO_MOVE; MAX_PLY]; 2],
            history_moves: [[0; SQUARES]; PIECES * 2],
            quiet_history: [[[0; SQUARES]; SQUARES]; SIDES],
            continuation_history: [[[0; SQUARES]; SQUARES]; PIECES * 2],
            counter_moves: [[[Move::NO_MOVE; SQUARES]; PIECES]; SIDES],
            capture_history: [[[0; PIECES]; SQUARES]; PIECES * 2],
        }
    }

    pub fn add_killer_move(&mut self, move_obj: Move, ply: usize) {
        if ply >= MAX_PLY || self.killer_moves[0][ply] == move_obj {
            return;
        }

        self.killer_moves[1][ply] = self.killer_moves[0][ply];
        self.killer_moves[0][ply] = move_obj;
    }

    pub fn update_history(&mut self, piece: usize, move_obj: Move, bonus: i32) {
        const MAX_HISTORY: i32 = 16384;
        let target = move_obj.target as usize;
        if piece < PIECES * 2 && target < SQUARES {
            Self::update_gravity(&mut self.history_moves[piece][target], bonus, MAX_HISTORY);
        }
    }

    pub fn reset(&mut self) {
        self.killer_moves = [[Move::NO_MOVE; MAX_PLY]; 2];
        self.history_moves = [[0; SQUARES]; PIECES * 2];
        self.quiet_history = [[[0; SQUARES]; SQUARES]; SIDES];
        self.continuation_history = [[[0; SQUARES]; SQUARES]; PIECES * 2];
        self.counter_moves = [[[Move::NO_MOVE; SQUARES]; PIECES]; SIDES];
        self.capture_history = [[[0; PIECES]; SQUARES]; PIECES * 2];
    }

    pub fn decay_history(&mut self) {
        for row in self.history_moves.iter_mut() {
            for score in row.iter_mut() {
                *score /= 2;
            }
        }
        for side in &mut self.quiet_history {
            for row in side {
                for score in row {
                    *score /= 2;
                }
            }
        }
        for piece in &mut self.continuation_history {
            for row in piece {
                for score in row {
                    *score /= 2;
                }
            }
        }
        for piece in &mut self.capture_history {
            for row in piece {
                for score in row {
                    *score /= 2;
                }
            }
        }
    }

    pub fn is_move_heuristic_empty(&self) -> bool {
        self.killer_moves
            .iter()
            .all(|row| row.iter().all(|&m| m == Move::NO_MOVE))
            && self
                .history_moves
                .iter()
                .all(|row| row.iter().all(|&s| s == 0))
            && self
                .quiet_history
                .iter()
                .all(|side| side.iter().all(|row| row.iter().all(|&s| s == 0)))
            && self
                .continuation_history
                .iter()
                .all(|piece| piece.iter().all(|row| row.iter().all(|&s| s == 0)))
            && self.counter_moves.iter().all(|side_row| {
                side_row
                    .iter()
                    .all(|piece_row| piece_row.iter().all(|&m| m == Move::NO_MOVE))
            })
            && self
                .capture_history
                .iter()
                .all(|piece| piece.iter().all(|row| row.iter().all(|&s| s == 0)))
    }

    #[inline(always)]
    pub fn sort_next_best_move(moves: &mut [ScoredMove], starting_index: usize) {
        let mut best_index = starting_index;
        for index in (starting_index + 1)..moves.len() {
            if moves[index].score > moves[best_index].score {
                best_index = index;
            }
        }
        if best_index != starting_index {
            moves.swap(best_index, starting_index);
        }
    }

    pub fn get_quiet_history_score(
        &self,
        board_state: &BoardState,
        move_obj: Move,
        previous_move: Option<Move>,
    ) -> i32 {
        let source = move_obj.source as usize;
        let target = move_obj.target as usize;
        if source >= SQUARES || target >= SQUARES {
            return 0;
        }
        let piece = board_state.get_piece_on(move_obj.source);
        if piece < 0 || piece as usize >= PIECES * 2 {
            return 0;
        }
        let history_score = self.history_moves[piece as usize][target];
        let from_to_score = self.quiet_history[board_state.side_to_move as usize][source][target];
        let continuation_score = previous_move
            .and_then(|prev_mv| {
                let prev_target = prev_mv.target as usize;
                if prev_target < SQUARES {
                    let prev_piece = board_state.piece_mapping[prev_target];
                    if (prev_piece as usize) < PIECES {
                        return Some(
                            self.continuation_history[prev_piece as usize][prev_target][target],
                        );
                    }
                }
                None
            })
            .unwrap_or(0) as i32;
        history_score + i32::from(from_to_score) + continuation_score
    }

    pub fn populate_quiet_scores(
        &self,
        moves: &mut [ScoredMove],
        board_state: &BoardState,
        ply: usize,
        previous_move: Option<Move>,
        threats: &[u64; 6],
        check_squares: &[u64; 6],
    ) {
        let counter_move = if let Some(prev_mv) = previous_move {
            let prev_target = prev_mv.target as usize;
            if prev_target < SQUARES {
                let prev_side = board_state.side_to_move.other();
                let prev_piece = board_state.piece_mapping[prev_target];
                if (prev_piece as usize) < PIECES {
                    self.counter_moves[prev_side as usize][prev_piece as usize][prev_target]
                } else {
                    Move::NO_MOVE
                }
            } else {
                Move::NO_MOVE
            }
        } else {
            Move::NO_MOVE
        };

        for move_obj in moves.iter_mut() {
            let prom_piece = move_obj.mv.move_type.promotion_piece();
            if prom_piece == Piece::Queen {
                move_obj.score = 25000;
            } else if ply < MAX_PLY && move_obj.mv == self.killer_moves[0][ply] {
                move_obj.score = 22000;
            } else if ply < MAX_PLY && move_obj.mv == self.killer_moves[1][ply] {
                move_obj.score = 21000;
            } else if counter_move != Move::NO_MOVE && move_obj.mv == counter_move {
                move_obj.score = 20000;
            } else if prom_piece != Piece::None {
                move_obj.score = -20000;
            } else {
                let source = move_obj.mv.source as usize;
                let target = move_obj.mv.target as usize;
                if source >= SQUARES || target >= SQUARES {
                    continue;
                }
                let piece = board_state.get_piece_on(move_obj.mv.source);
                if piece >= 0 && (piece as usize) < PIECES * 2 {
                    let history_score =
                        self.history_moves[piece as usize][target];
                    let from_to_score = self.quiet_history[board_state.side_to_move as usize]
                        [source][target];
                    let continuation_score = previous_move
                        .and_then(|prev_mv| {
                            let prev_target = prev_mv.target as usize;
                            if prev_target < SQUARES {
                                let prev_piece = board_state.piece_mapping[prev_target];
                                if (prev_piece as usize) < PIECES {
                                    return Some(
                                        self.continuation_history[prev_piece as usize]
                                            [prev_target]
                                            [target],
                                    );
                                }
                            }
                            None
                        })
                        .unwrap_or(0) as i32;
                    let mut score =
                        2 * history_score + i32::from(from_to_score) + continuation_score;

                    let pt = (piece as usize) % PIECES;
                    let to_mask = 1u64 << target;
                    let from_mask = 1u64 << source;

                    if (check_squares[pt] & to_mask) != 0 && board_state.see_ge(move_obj.mv, -75) {
                        score += 16384;
                    }

                    let was_threatened = (threats[pt] & from_mask) != 0;
                    let is_threatened = (threats[pt] & to_mask) != 0;
                    let v = 20 * (was_threatened as i32 - is_threatened as i32);
                    let pt_val = match pt {
                        0 => 100,
                        1 => 300,
                        2 => 300,
                        3 => 500,
                        4 => 900,
                        _ => 0,
                    };
                    score += pt_val * v;

                    move_obj.score = score;
                }
            }
        }
    }

    #[inline(always)]
    pub fn update_quiet_history(&mut self, side: Side, move_obj: Move, bonus: i32) {
        let source = move_obj.source as usize;
        let target = move_obj.target as usize;
        if source < SQUARES && target < SQUARES {
            Self::update_gravity_i16(
                &mut self.quiet_history[side as usize][source][target],
                bonus,
                16384,
            );
        }
    }

    #[inline(always)]
    pub fn update_capture_history(
        &mut self,
        moved_piece: usize,
        target_square: Square,
        captured_piece: Piece,
        bonus: i32,
    ) {
        let target = target_square as usize;
        if moved_piece < PIECES * 2 && target < SQUARES && captured_piece != Piece::None {
            let cap_idx = captured_piece as usize;
            if cap_idx < PIECES {
                Self::update_gravity_i16(
                    &mut self.capture_history[moved_piece][target][cap_idx],
                    bonus,
                    16384,
                );
            }
        }
    }

    #[inline(always)]
    pub fn update_continuation_history(
        &mut self,
        previous_piece: Piece,
        previous_target: Square,
        move_obj: Move,
        bonus: i32,
    ) {
        let target = move_obj.target as usize;
        let prev_target = previous_target as usize;
        let prev_piece = previous_piece as usize;
        if prev_piece < PIECES && prev_target < SQUARES && target < SQUARES {
            Self::update_gravity_i16(
                &mut self.continuation_history[prev_piece][prev_target][target],
                bonus,
                i16::MAX as i32,
            );
        }
    }

    #[inline(always)]
    fn update_gravity(score: &mut i32, bonus: i32, limit: i32) {
        let bonus = bonus.clamp(-limit, limit);
        *score += bonus - *score * bonus.abs() / limit;
    }

    #[inline(always)]
    fn update_gravity_i16(score: &mut i16, bonus: i32, limit: i32) {
        let mut value = i32::from(*score);
        Self::update_gravity(&mut value, bonus, limit);
        *score = value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
    pub fn add_counter_move(
        &mut self,
        prev_side: Side,
        prev_piece: Piece,
        prev_square: Square,
        counter_move: Move,
    ) {
        let prev_p = prev_piece as usize;
        let prev_sq = prev_square as usize;
        if prev_p < PIECES && prev_sq < SQUARES {
            self.counter_moves[prev_side as usize][prev_p][prev_sq] = counter_move;
        }
    }
}

impl Default for MoveOrdering {
    fn default() -> Self {
        Self::new()
    }
}

pub fn populate_capture_scores(
    moves: &mut [ScoredMove],
    board_state: &BoardState,
    move_ordering: &MoveOrdering,
) {
    for move_obj in moves.iter_mut() {
        let target = move_obj.mv.target as usize;
        let source = move_obj.mv.source as usize;
        if target >= SQUARES || source >= SQUARES {
            continue;
        }
        let source_piece =
            board_state.get_piece_on_side(move_obj.mv.source, board_state.side_to_move);
        let target_piece = if move_obj.mv.move_type == MoveType::EnPassant {
            Piece::Pawn
        } else {
            board_state.piece_mapping[target]
        };

        let mut score = if target_piece != Piece::None && source_piece < 7 {
            MVV_LVA[target_piece as usize][source_piece]
        } else {
            0
        };
        let prom_piece = move_obj.mv.move_type.promotion_piece();
        if prom_piece == Piece::Queen {
            score += 50000;
        } else if prom_piece != Piece::None {
            score -= 20000;
        }
        let moved_piece = board_state.get_piece_on(move_obj.mv.source);
        if moved_piece >= 0
            && (moved_piece as usize) < PIECES * 2
            && target_piece != Piece::None
            && (target_piece as usize) < PIECES
        {
            let hist = move_ordering.capture_history[moved_piece as usize][target][target_piece as usize];
            score += hist as i32;
        }
        move_obj.score = score;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::square::Square;

    #[test]
    fn should_sort_moves_by_score() {
        let mut moves = vec![
            ScoredMove {
                mv: Move {
                    source: Square::E2,
                    target: Square::E4,
                    move_type: MoveType::Quiet,
                },
                score: 100,
            },
            ScoredMove {
                mv: Move {
                    source: Square::D2,
                    target: Square::D4,
                    move_type: MoveType::Quiet,
                },
                score: 300,
            },
            ScoredMove {
                mv: Move {
                    source: Square::G1,
                    target: Square::F3,
                    move_type: MoveType::Quiet,
                },
                score: 200,
            },
        ];

        MoveOrdering::sort_next_best_move(&mut moves, 0);

        assert_eq!(moves[0].score, 300);
    }

    #[test]
    fn should_not_change_order_if_already_sorted() {
        let mut moves = vec![
            ScoredMove {
                mv: Move {
                    source: Square::D2,
                    target: Square::D4,
                    move_type: MoveType::Quiet,
                },
                score: 300,
            },
            ScoredMove {
                mv: Move {
                    source: Square::G1,
                    target: Square::F3,
                    move_type: MoveType::Quiet,
                },
                score: 200,
            },
            ScoredMove {
                mv: Move {
                    source: Square::E2,
                    target: Square::E4,
                    move_type: MoveType::Quiet,
                },
                score: 100,
            },
        ];

        MoveOrdering::sort_next_best_move(&mut moves, 1);

        assert_eq!(moves[0].score, 300);
        assert_eq!(moves[1].score, 200);
        assert_eq!(moves[2].score, 100);
    }

    #[test]
    fn should_prioritize_queen_promotions_and_penalize_under_promotions() {
        let board = BoardState::parse_fen("1r5k/P1Q5/8/8/8/8/8/K7 w - - 0 1");
        let mut move_ordering = MoveOrdering::new();

        let mut quiet_moves = vec![
            ScoredMove {
                mv: Move {
                    source: Square::A7,
                    target: Square::A8,
                    move_type: MoveType::QueenPromotion,
                },
                score: 0,
            },
            ScoredMove {
                mv: Move {
                    source: Square::A7,
                    target: Square::A8,
                    move_type: MoveType::RookPromotion,
                },
                score: 0,
            },
            ScoredMove {
                mv: Move {
                    source: Square::C7,
                    target: Square::C6,
                    move_type: MoveType::Quiet,
                },
                score: 0,
            },
        ];

        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        move_ordering.populate_quiet_scores(
            &mut quiet_moves,
            &board,
            0,
            None,
            &nt.threats_us,
            &nt.check_squares,
        );

        assert_eq!(quiet_moves[0].score, 25000);
        assert_eq!(quiet_moves[1].score, -20000);
        assert_eq!(quiet_moves[2].score, 0);

        let under_prom = Move {
            source: Square::A7,
            target: Square::A8,
            move_type: MoveType::RookPromotion,
        };
        move_ordering.add_killer_move(under_prom, 0);

        let mut killer_quiet_moves = vec![ScoredMove {
            mv: under_prom,
            score: 0,
        }];
        move_ordering.populate_quiet_scores(
            &mut killer_quiet_moves,
            &board,
            0,
            None,
            &nt.threats_us,
            &nt.check_squares,
        );
        assert_eq!(killer_quiet_moves[0].score, 22000);

        let mut capture_moves = vec![
            ScoredMove {
                mv: Move {
                    source: Square::A7,
                    target: Square::B8,
                    move_type: MoveType::QueenPromotionCapture,
                },
                score: 0,
            },
            ScoredMove {
                mv: Move {
                    source: Square::A7,
                    target: Square::B8,
                    move_type: MoveType::RookPromotionCapture,
                },
                score: 0,
            },
            ScoredMove {
                mv: Move {
                    source: Square::C7,
                    target: Square::B8,
                    move_type: MoveType::Capture,
                },
                score: 0,
            },
        ];

        populate_capture_scores(&mut capture_moves, &board, &move_ordering);

        assert_eq!(capture_moves[0].score, 95000);
        assert_eq!(capture_moves[1].score, 25000);
        assert_eq!(capture_moves[2].score, 41000);
    }

    #[test]
    fn quiet_history_is_separated_by_side_and_from_to() {
        let mut ordering = MoveOrdering::new();
        let move_obj = Move::new(Square::E2, Square::E4, MoveType::Quiet);

        ordering.update_quiet_history(Side::White, move_obj, 1000);

        assert!(
            ordering.quiet_history[Side::White as usize][Square::E2 as usize][Square::E4 as usize]
                > 0
        );
        assert_eq!(
            ordering.quiet_history[Side::Black as usize][Square::E2 as usize][Square::E4 as usize],
            0
        );
        assert_eq!(
            ordering.quiet_history[Side::White as usize][Square::E2 as usize][Square::E3 as usize],
            0
        );
    }

    #[test]
    fn continuation_history_uses_previous_piece_and_target() {
        let mut ordering = MoveOrdering::new();
        let current = Move::new(Square::E2, Square::E4, MoveType::Quiet);

        ordering.update_continuation_history(Piece::Knight, Square::F3, current, 1000);

        assert!(
            ordering.continuation_history[Piece::Knight as usize][Square::F3 as usize]
                [Square::E4 as usize]
                > 0
        );
        assert_eq!(
            ordering.continuation_history[Piece::Bishop as usize][Square::F3 as usize]
                [Square::E4 as usize],
            0
        );
    }

    #[test]
    fn capture_history_distinguishes_captured_piece_type() {
        let mut ordering = MoveOrdering::new();
        let knight_idx = Piece::Knight as usize;

        ordering.update_capture_history(knight_idx, Square::D5, Piece::Queen, 1000);

        assert!(
            ordering.capture_history[knight_idx][Square::D5 as usize][Piece::Queen as usize] > 0
        );
        assert_eq!(
            ordering.capture_history[knight_idx][Square::D5 as usize][Piece::Pawn as usize],
            0
        );
        assert_eq!(
            ordering.capture_history[knight_idx][Square::E5 as usize][Piece::Queen as usize],
            0
        );
    }

    #[test]
    fn killer_duplicate_is_ignored_and_second_killer_scores() {
        let mut ordering = MoveOrdering::new();
        let first = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let second = Move::new(Square::D2, Square::D4, MoveType::Quiet);
        ordering.add_killer_move(first, 0);
        ordering.add_killer_move(first, 0);
        assert_eq!(ordering.killer_moves[0][0], first);
        assert_eq!(ordering.killer_moves[1][0], Move::NO_MOVE);
        ordering.add_killer_move(second, 0);
        assert_eq!(ordering.killer_moves[0][0], second);
        assert_eq!(ordering.killer_moves[1][0], first);

        let board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let mut moves = vec![
            ScoredMove {
                mv: first,
                score: 0,
            },
            ScoredMove {
                mv: second,
                score: 0,
            },
        ];
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        ordering.populate_quiet_scores(
            &mut moves,
            &board,
            0,
            None,
            &nt.threats_us,
            &nt.check_squares,
        );
        assert_eq!(moves[0].score, 21000);
        assert_eq!(moves[1].score, 22000);
        let _ = MoveOrdering::default();
    }

    #[test]
    fn history_reset_decay_and_empty_probe() {
        let mut ordering = MoveOrdering::new();
        assert!(ordering.is_move_heuristic_empty());
        let mv = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        ordering.update_history(0, mv, 1000);
        ordering.update_quiet_history(Side::White, mv, 1000);
        assert!(!ordering.is_move_heuristic_empty());
        let before = ordering.history_moves[0][Square::E4 as usize];
        ordering.decay_history();
        assert_eq!(ordering.history_moves[0][Square::E4 as usize], before / 2);
        ordering.reset();
        assert!(ordering.is_move_heuristic_empty());
    }

    #[test]
    fn quiet_score_handles_empty_and_continuation() {
        let mut board = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let ordering = MoveOrdering::new();
        let empty = Move::new(Square::E3, Square::E4, MoveType::Quiet);
        assert_eq!(ordering.get_quiet_history_score(&board, empty, None), 0);
        let prev_empty = Move::new(Square::A3, Square::A4, MoveType::Quiet);
        let mv = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        assert_eq!(
            ordering.get_quiet_history_score(&board, mv, Some(prev_empty)),
            0
        );

        let e4 = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
        board.make_move(e4);
        let mut ordering2 = MoveOrdering::new();
        let reply = Move::new(Square::D7, Square::D5, MoveType::DoublePush);
        ordering2.add_counter_move(Side::White, Piece::Pawn, Square::E4, reply);
        let mut scored = vec![ScoredMove {
            mv: reply,
            score: 0,
        }];
        let nt = crate::board::node_threats::NodeThreats::compute(&board);
        ordering2.populate_quiet_scores(
            &mut scored,
            &board,
            0,
            Some(e4),
            &nt.threats_us,
            &nt.check_squares,
        );
        assert_eq!(scored[0].score, 20000);
    }

    #[test]
    fn capture_history_guards_and_en_passant() {
        let mut ordering = MoveOrdering::new();
        ordering.update_capture_history(999, Square::D5, Piece::Queen, 1000);
        ordering.update_capture_history(0, Square::D5, Piece::None, 1000);
        assert!(ordering.is_move_heuristic_empty());
        ordering.update_continuation_history(
            Piece::None,
            Square::E5,
            Move::new(Square::E2, Square::E4, MoveType::Quiet),
            1000,
        );
        assert!(ordering.is_move_heuristic_empty());
        ordering.add_counter_move(
            Side::White,
            Piece::Pawn,
            Square::E4,
            Move::new(Square::E2, Square::E4, MoveType::Quiet),
        );
        assert!(!ordering.is_move_heuristic_empty());

        let board =
            BoardState::parse_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
        let mut moves = vec![ScoredMove {
            mv: Move::new(Square::E5, Square::D6, MoveType::EnPassant),
            score: 0,
        }];
        populate_capture_scores(&mut moves, &board, &ordering);
        assert!(moves[0].score > 0);
        let start = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
        let mut quiets = vec![ScoredMove {
            mv: Move::new(Square::E2, Square::E3, MoveType::Quiet),
            score: 77,
        }];
        populate_capture_scores(&mut quiets, &start, &ordering);
        assert_eq!(quiets[0].score, 0);
    }
}
