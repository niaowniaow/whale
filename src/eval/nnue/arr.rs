use crate::board::state::BoardState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrConfig {
    pub enabled: bool,
    pub lambda_arr: f32,
    pub max_allowed_divergence: i16,
}

impl Default for ArrConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lambda_arr: 0.08,
            max_allowed_divergence: 30,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdversarialRobustnessReport {
    pub original_eval: i16,
    pub mirror_eval: i16,
    pub symmetry_divergence: i16,
    pub robustness_score: f32,
    pub is_robust: bool,
}

#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn compute_arr_loss(main_out: f32, aux_out: f32, target: f32, lambda: f32) -> f32 {
    let p_main = sigmoid(main_out);
    let p_aux = sigmoid(aux_out);
    let main_loss = (p_main - target).powi(2);
    let aux_loss = (p_aux - target).powi(2);
    let coherence_penalty = (p_main - p_aux).powi(2);
    main_loss + 0.1 * aux_loss + lambda * coherence_penalty
}

pub fn create_mirrored_board(board: &BoardState) -> BoardState {
    let mut mirrored_fen = String::new();
    let fen_str = board.to_fen();
    let ranks: Vec<&str> = fen_str.split_whitespace().collect();
    if ranks.is_empty() {
        return board.clone();
    }

    let board_part = ranks[0];
    let mut mirrored_rows = Vec::new();

    for row in board_part.split('/') {
        let mut expanded = Vec::new();
        for ch in row.chars() {
            if let Some(digit) = ch.to_digit(10) {
                expanded.extend(std::iter::repeat_n(' ', digit as usize));
            } else {
                expanded.push(ch);
            }
        }
        expanded.reverse();

        let mut collapsed = String::new();
        let mut empty_count = 0;
        for ch in expanded {
            if ch == ' ' {
                empty_count += 1;
            } else {
                if empty_count > 0 {
                    collapsed.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                collapsed.push(ch);
            }
        }
        if empty_count > 0 {
            collapsed.push_str(&empty_count.to_string());
        }
        mirrored_rows.push(collapsed);
    }

    mirrored_fen.push_str(&mirrored_rows.join("/"));
    // A horizontal (file) mirror keeps STM but swaps kingside/queenside
    // castling rights and mirrors the en-passant file.
    if let Some(stm) = ranks.get(1) {
        mirrored_fen.push(' ');
        mirrored_fen.push_str(stm);
    }
    if let Some(castling) = ranks.get(2) {
        mirrored_fen.push(' ');
        if *castling == "-" {
            mirrored_fen.push('-');
        } else {
            for ch in castling.chars() {
                mirrored_fen.push(match ch {
                    'K' => 'Q',
                    'Q' => 'K',
                    'k' => 'q',
                    'q' => 'k',
                    other => other,
                });
            }
        }
    }
    if let Some(ep) = ranks.get(3) {
        mirrored_fen.push(' ');
        if *ep == "-" {
            mirrored_fen.push('-');
        } else {
            let mut chars = ep.chars();
            match (chars.next(), chars.next()) {
                (Some(file), Some(rank)) => {
                    let mirrored_file = (b'h' - file as u8 + b'a') as char;
                    mirrored_fen.push(mirrored_file);
                    mirrored_fen.push(rank);
                }
                _ => mirrored_fen.push_str(ep),
            }
        }
    }
    for part in ranks.iter().skip(4) {
        mirrored_fen.push(' ');
        mirrored_fen.push_str(part);
    }

    BoardState::parse_fen(&mirrored_fen)
}

pub fn audit_adversarial_robustness(
    board: &mut BoardState,
    config: ArrConfig,
) -> AdversarialRobustnessReport {
    let network = crate::eval::nnue::loader::Network::get_embedded();
    board.ensure_accumulators_fresh();
    let original_eval = crate::eval::nnue::evaluate_internal(board, network);

    let mut mirrored_board = create_mirrored_board(board);
    mirrored_board.ensure_accumulators_fresh();
    let mirror_eval = crate::eval::nnue::evaluate_internal(&mirrored_board, network);

    let symmetry_divergence = (original_eval as i32 - mirror_eval as i32).abs() as i16;
    let is_robust = symmetry_divergence <= config.max_allowed_divergence;

    let divergence_ratio =
        (symmetry_divergence as f32 / (config.max_allowed_divergence as f32).max(1.0)).min(2.0);
    let robustness_score = (1.0 - divergence_ratio * 0.5).clamp(0.0, 1.0);

    AdversarialRobustnessReport {
        original_eval,
        mirror_eval,
        symmetry_divergence,
        robustness_score,
        is_robust,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_sigmoid_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-5);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_compute_arr_loss_penalizes_divergence() {
        let baseline_loss = compute_arr_loss(0.0, 0.0, 0.5, 0.5);
        let divergent_loss = compute_arr_loss(0.0, 2.0, 0.5, 0.5);
        assert!(divergent_loss > baseline_loss);
    }

    #[test]
    fn test_create_mirrored_board_starting_position() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mirrored = create_mirrored_board(&board);
        let count1 = board.occupancies[crate::common::side::Side::White].count_ones()
            + board.occupancies[crate::common::side::Side::Black].count_ones();
        let count2 = mirrored.occupancies[crate::common::side::Side::White].count_ones()
            + mirrored.occupancies[crate::common::side::Side::Black].count_ones();
        assert_eq!(count1, count2);
        assert_eq!(board.side_to_move, mirrored.side_to_move);
    }

    #[test]
    fn test_audit_adversarial_robustness_starting_pos() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let report = audit_adversarial_robustness(&mut board, ArrConfig::default());
        assert!(report.robustness_score >= 0.0 && report.robustness_score <= 1.0);
    }

    #[test]
    fn test_arr_config_defaults() {
        let cfg = ArrConfig::default();
        assert!(cfg.enabled);
        assert!((cfg.lambda_arr - 0.08).abs() < 1e-6);
        assert_eq!(cfg.max_allowed_divergence, 30);
    }

    #[test]
    fn test_compute_arr_loss_lambda_zero_ignores_coherence() {
        // With lambda 0 the coherence term drops out, but aux loss still matters.
        let same = compute_arr_loss(0.0, 0.0, 0.5, 0.0);
        let divergent = compute_arr_loss(0.0, 2.0, 0.5, 0.0);
        assert!(divergent > same);
        // Identical heads give identical loss regardless of lambda.
        assert_eq!(
            compute_arr_loss(0.5, 0.5, 0.5, 0.0),
            compute_arr_loss(0.5, 0.5, 0.5, 1.0)
        );
    }

    #[test]
    fn test_create_mirrored_board_swaps_castling_and_ep() {
        let board = BoardState::parse_fen("r3k2r/ppp1pppp/8/3pP3/8/8/PPPP1PPP/R3K2R w Kq d6 0 2");
        let mirrored = create_mirrored_board(&board);
        let fen = mirrored.to_fen();
        // K <-> Q swap on each side: "Kq" mirrors to "Qk".
        assert!(fen.contains(" Qk "), "castling not mirrored: {fen}");
        // d6 mirrors to e6.
        assert!(fen.contains(" e6 "), "en-passant not mirrored: {fen}");
        assert_eq!(board.side_to_move, mirrored.side_to_move);
    }

    #[test]
    fn test_create_mirrored_board_dash_fields_preserved() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mirrored = create_mirrored_board(&board);
        let fen = mirrored.to_fen();
        assert!(fen.contains(" KQkq "));
        assert!(fen.contains(" - "));
        // Halfmove/fullmove suffix preserved.
        assert!(fen.ends_with("0 1"));
    }

    #[test]
    fn test_audit_report_fields_are_consistent() {
        let mut board = BoardState::parse_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        );
        let config = ArrConfig::default();
        let report = audit_adversarial_robustness(&mut board, config);
        let expected_div = (report.original_eval as i32 - report.mirror_eval as i32).abs() as i16;
        assert_eq!(report.symmetry_divergence, expected_div);
        assert_eq!(
            report.is_robust,
            report.symmetry_divergence <= config.max_allowed_divergence
        );
        assert!((0.0..=1.0).contains(&report.robustness_score));
        // Zero divergence yields a perfect score.
        if report.symmetry_divergence == 0 {
            assert_eq!(report.robustness_score, 1.0);
        }
    }
}
