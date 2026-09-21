use crate::common::side::Side;

pub const DRAW_BASE: i16 = 0;

pub const CONTEMPT_LIMIT: i16 = 200;

#[inline(always)]
pub fn draw_score(nodes: u64) -> i16 {
    if nodes & 2 == 0 {
        DRAW_BASE
    } else {
        DRAW_BASE + 1
    }
}

pub fn apply_contempt(
    score: i16,
    stm: Side,
    engine_side: Side,
    contempt_cp: i16,
    draw_score_cp: i16,
) -> i16 {
    let c = contempt_cp.clamp(-CONTEMPT_LIMIT, CONTEMPT_LIMIT);
    let d = draw_score_cp.clamp(-CONTEMPT_LIMIT, CONTEMPT_LIMIT);
    if c == 0 && d == 0 {
        return score;
    }
    if score.abs() > 10 {
        return score;
    }

    let contempt = if stm == engine_side { c } else { -c };
    score.saturating_add(d).saturating_sub(contempt)
}

pub fn cp_to_wdl(cp: i16) -> (u16, u16, u16) {
    const MATE_CP: i32 = 29_000;
    let v = cp as i32;
    if v >= MATE_CP {
        return (1000, 0, 0);
    }
    if v <= -MATE_CP {
        return (0, 0, 1000);
    }

    let w = 1000.0 / (1.0 + (-f64::from(v) / 220.0).exp());
    let draw = (-(f64::from(v.abs()) / 180.0)).exp() * 420.0;
    let mut wi = w.round() as i32;
    let mut di = draw.round() as i32;
    wi = wi.clamp(0, 1000);
    di = di.clamp(0, 1000 - wi);
    let li = 1000 - wi - di;
    (wi as u16, di as u16, li as u16)
}

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

        assert_ne!(draw_score(0), draw_score(2));
    }

    #[test]
    fn contempt_leaves_default_scores_untouched() {
        assert_eq!(apply_contempt(0, Side::White, Side::White, 0, 0), 0);
        assert_eq!(apply_contempt(1, Side::Black, Side::White, 0, 0), 1);
        assert_eq!(apply_contempt(500, Side::White, Side::White, 50, 0), 500);
    }

    #[test]
    fn contempt_shifts_draws() {
        let ours = apply_contempt(0, Side::White, Side::White, 30, 0);
        let theirs = apply_contempt(0, Side::Black, Side::White, 30, 0);
        assert!(ours < 0, "engine must devalue draws when contempt > 0");
        assert_eq!(theirs, -ours, "contempt must be zero-sum across sides");

        let d = apply_contempt(0, Side::White, Side::White, 0, 20);
        assert_eq!(d, 20);
    }

    #[test]
    fn contempt_follows_the_engine_side() {
        assert_eq!(
            apply_contempt(0, Side::White, Side::Black, 40, 0),
            -apply_contempt(0, Side::White, Side::White, 40, 0)
        );

        assert!(apply_contempt(0, Side::Black, Side::Black, -40, 0) > 0);

        assert_eq!(
            apply_contempt(0, Side::White, Side::Black, -40, 0),
            -apply_contempt(0, Side::White, Side::White, -40, 0)
        );
    }

    #[test]
    fn contempt_uses_the_full_option_range() {
        assert_eq!(apply_contempt(0, Side::White, Side::White, 50, 0), -50);
        assert_eq!(
            apply_contempt(0, Side::White, Side::White, 200, 0),
            -CONTEMPT_LIMIT
        );

        assert_eq!(
            apply_contempt(0, Side::White, Side::White, i16::MAX, 0),
            -CONTEMPT_LIMIT
        );
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
