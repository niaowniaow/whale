use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_aspiration(
    board_state: &mut BoardState,
    current_depth: u8,
    last_score: i16,
    previous_pv: &[Move],
    cancellation_token: &AtomicBool,
    pv_table: &mut PvTable,
    search_state: &mut SearchState,
) -> (i16, bool) {
    let mut alpha = i16::MIN + 1;
    let mut beta = i16::MAX - 1;

    if current_depth > 1 {
        if last_score.abs() as i32 > MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32 {
            alpha = i16::MIN + 1;
            beta = i16::MAX - 1;
        } else {
            alpha = last_score
                .saturating_sub(ASPIRATION_WINDOW_MARGIN)
                .max(i16::MIN + 1);
            beta = last_score
                .saturating_add(ASPIRATION_WINDOW_MARGIN)
                .min(i16::MAX - 1);
        }
    }

    let mut current_score = last_score;
    let mut completed = false;
    let mut delta = ASPIRATION_WINDOW_MARGIN;

    loop {
        let score = negamax::search(
            board_state,
            current_depth,
            alpha,
            beta,
            cancellation_token,
            previous_pv,
            &mut *pv_table,
            search_state,
        );

        if cancellation_token.load(Ordering::Relaxed) {
            break;
        }

        current_score = score;
        if current_score <= alpha {
            if alpha == i16::MIN + 1 {
                completed = true;
                break;
            }
            beta = alpha;
            alpha = current_score.saturating_sub(delta).max(i16::MIN + 1);
        } else if current_score >= beta {
            if beta == i16::MAX - 1 {
                completed = true;
                break;
            }
            alpha = beta.saturating_sub(delta).max(alpha);
            beta = current_score.saturating_add(delta).min(i16::MAX - 1);
        } else {
            completed = true;
            break;
        }
        delta = delta
            .saturating_add(delta.saturating_mul(47) / 128)
            .min(1500);
    }
    (current_score, completed)
}
