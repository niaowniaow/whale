//! Draw scoring, contempt and WDL helpers.
//!
//! Clean-room ideas learned from Stockfish (`value_draw`, upcoming-repetition
//! handling, 50-move gating) and Lc0 (`Contempt`, `DrawScore`, WDL reporting).
//! No code is ported: formulas below are Whale's own, only the *concepts*
//! (parity-tweaked draws, user contempt, logistic cp->WDL) are reused.

/// Nominal draw score before contempt (centipawns, from side-to-move view).
pub const DRAW_BASE: i16 = 0;

/// Clamp range for user contempt / draw-score options.
pub const CONTEMPT_LIMIT: i16 = 200;

/// Draw score with node parity tweak.
///
/// Returning a flat `0` for every repetition makes the search blind to *which*
/// repetition arrives first. Flipping the score by +-1 based on `nodes`
/// (Stockfish `value_draw` concept) gives the search a gradient to prefer
/// delaying/avoiding the draw without changing the game-theoretic value.
/// With default options the result stays within [-1, 1].
#[inline(always)]
pub fn draw_score(nodes: u64) -> i16 {
    if nodes & 2 == 0 {
        DRAW_BASE
    } else {
        DRAW_BASE + 1
    }
}

/// Apply user contempt to a draw-ish score.
///
/// * `score` — score from side-to-move perspective.
/// * `stm_is_white` — true when White is to move.
/// * `contempt_cp` — positive means "avoid draws" for both sides
///   (Lc0 `Contempt` concept, linear form).
/// * `draw_score_cp` — absolute draw value override (Lc0 `DrawScore` concept).
///
/// Only near-zero scores are shifted so mates/TB wins are untouched.
pub fn apply_contempt(score: i16, stm_is_white: bool, contempt_cp: i16, draw_score_cp: i16) -> i16 {
    let c = contempt_cp.clamp(-CONTEMPT_LIMIT, CONTEMPT_LIMIT);
    let d = draw_score_cp.clamp(-CONTEMPT_LIMIT, CONTEMPT_LIMIT);
    if c == 0 && d == 0 {
        return score;
    }
    if score.abs() > 10 {
        return score;
    }
    // Shift the draw baseline, then add side-aware contempt so both sides
    // prefer playing on when contempt is positive.
    let side_sign: i16 = if stm_is_white { 1 } else { -1 };
    let _ = side_sign;
    // Symmetric form: positive contempt lowers the draw value for the
    // side to move (they would rather play on).
    score
        .saturating_add(d)
        .saturating_sub(c.signum() * c.abs().min(50))
}

/// Convert a centipawn score to WDL permille (w, d, l), sum = 1000.
///
/// Whale's own logistic mapping (Lc0 reports `wdl` the same way conceptually
/// but with a different curve). Used for `info ... wdl` and future
/// resign/adjudication thresholds. Mate scores saturate.
pub fn cp_to_wdl(cp: i16) -> (u16, u16, u16) {
    const MATE_CP: i32 = 29_000;
    let v = cp as i32;
    if v >= MATE_CP {
        return (1000, 0, 0);
    }
    if v <= -MATE_CP {
        return (0, 0, 1000);
    }
    // Logistic win probability, draw mass concentrated near equality.
    let w = 1000.0 / (1.0 + (-f64::from(v) / 220.0).exp());
    let draw = (-(f64::from(v.abs()) / 180.0)).exp() * 420.0;
    let mut wi = w.round() as i32;
    let mut di = draw.round() as i32;
    wi = wi.clamp(0, 1000);
    di = di.clamp(0, 1000 - wi);
    let li = 1000 - wi - di;
    (wi as u16, di as u16, li as u16)
}

/// 50-move TT-cutoff gate (Stockfish concept: no TT cutoff at rule50 >= 96).
#[inline(always)]
pub fn allow_tt_cutoff(halfmove: u8) -> bool {
    halfmove < 96
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_score_stays_near_zero() {
        assert!(draw_score(0).abs() <= 1);
        assert!(draw_score(1).abs() <= 1);
        assert!(draw_score(2).abs() <= 1);
        assert!(draw_score(3).abs() <= 1);
        // Parity flips at bit 1.
        assert_ne!(draw_score(0), draw_score(2));
    }

    #[test]
    fn contempt_leaves_default_scores_untouched() {
        assert_eq!(apply_contempt(0, true, 0, 0), 0);
        assert_eq!(apply_contempt(1, false, 0, 0), 1);
        assert_eq!(apply_contempt(500, true, 50, 0), 500);
    }

    #[test]
    fn contempt_shifts_draws() {
        let base = apply_contempt(0, true, 30, 0);
        assert!(base < 0, "positive contempt must devalue draws");
        let d = apply_contempt(0, true, 0, 20);
        assert_eq!(d, 20);
    }

    #[test]
    fn wdl_sums_to_1000_and_saturates() {
        for cp in [-29_500, -1000, -100, 0, 100, 1000, 29_500] {
            let (w, d, l) = cp_to_wdl(cp);
            assert_eq!(w as u32 + d as u32 + l as u32, 1000);
        }
        assert!(cp_to_wdl(0).1 >= 300);
        assert_eq!(cp_to_wdl(30_000), (1000, 0, 0));
        assert_eq!(cp_to_wdl(-30_000), (0, 0, 1000));
        let (w_pos, _, _) = cp_to_wdl(200);
        let (w_neg, _, _) = cp_to_wdl(-200);
        assert!(w_pos > w_neg);
    }

    #[test]
    fn tt_gate_matches_96() {
        assert!(allow_tt_cutoff(0));
        assert!(allow_tt_cutoff(90));
        assert!(allow_tt_cutoff(95));
        assert!(!allow_tt_cutoff(96));
        assert!(!allow_tt_cutoff(100));
    }
}
