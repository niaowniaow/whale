use crate::board::state::BoardState;
use crate::common::move_list::MoveList;
use crate::common::moves::Move;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlunderType {
    HangingPiece,
    BadExchange,
}

#[derive(Debug, Clone)]
pub struct CounterfactualSample {
    pub board: BoardState,
    pub blunder_move: Move,
    pub penalized_eval: i16,
    pub blunder_type: BlunderType,
    pub see_score: i16,
}

#[derive(Debug, Clone, Copy)]
pub struct CmaConfig {
    pub min_blunder_loss: i16,
    pub penalty_offset: i16,
    pub max_samples_per_position: usize,
}

impl Default for CmaConfig {
    fn default() -> Self {
        Self {
            min_blunder_loss: -200,
            penalty_offset: 450,
            max_samples_per_position: 2,
        }
    }
}

pub fn generate_counterfactual_samples(
    board: &BoardState,
    best_move: Move,
    base_eval: i16,
    config: CmaConfig,
) -> Vec<CounterfactualSample> {
    let mut move_list = MoveList::new();
    board.generate_moves(&mut move_list);

    let mut candidates: Vec<(Move, i16, BlunderType)> = Vec::new();

    for i in 0..move_list.len() {
        let m = move_list[i].mv;
        if m == best_move || !board.is_legal(m) {
            continue;
        }

        let see_score = board.see(m);
        if see_score <= config.min_blunder_loss {
            let blunder_type = if see_score <= -500 {
                BlunderType::HangingPiece
            } else {
                BlunderType::BadExchange
            };
            candidates.push((m, see_score, blunder_type));
        }
    }

    candidates.sort_unstable_by_key(|a| a.1);

    candidates
        .into_iter()
        .take(config.max_samples_per_position)
        .map(|(m, see_score, blunder_type)| {
            let mut sample_board = board.clone();
            sample_board.make_move(m);

            let penalty =
                (config.penalty_offset as i32 + (-see_score as i32)).clamp(300, 1500) as i16;
            let penalized_eval = (-base_eval).saturating_sub(penalty).min(-400);

            CounterfactualSample {
                board: sample_board,
                blunder_move: m,
                penalized_eval,
                blunder_type,
                see_score,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_cma_ignores_best_move() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);
        let first_move = move_list[0].mv;

        let samples = generate_counterfactual_samples(
            &board,
            first_move,
            20,
            CmaConfig {
                min_blunder_loss: 10_000,
                penalty_offset: 400,
                max_samples_per_position: 5,
            },
        );

        for sample in samples {
            assert_ne!(sample.blunder_move, first_move);
        }
    }

    #[test]
    fn test_cma_detects_bad_hanging_move() {
        let board =
            BoardState::parse_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
        let mut move_list = MoveList::new();
        board.generate_moves(&mut move_list);

        let config = CmaConfig {
            min_blunder_loss: -100,
            penalty_offset: 450,
            max_samples_per_position: 3,
        };

        let samples = generate_counterfactual_samples(&board, Move::NO_MOVE, 0, config);
        for sample in &samples {
            assert!(sample.see_score <= config.min_blunder_loss);
            assert!(sample.penalized_eval <= -400);
        }
    }

    #[test]
    fn test_cma_max_samples_limit() {
        let board =
            BoardState::parse_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
        let config = CmaConfig {
            min_blunder_loss: 0,
            penalty_offset: 400,
            max_samples_per_position: 1,
        };

        let samples = generate_counterfactual_samples(&board, Move::NO_MOVE, 0, config);
        assert!(samples.len() <= 1);
    }
}
