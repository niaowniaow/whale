use crate::common::side::Side;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

pub const DRAW_BASE: i16 = 0;

pub const CONTEMPT_LIMIT: i16 = 200;

pub const ATTENUATION_START_CP: i32 = 10;

pub const ATTENUATION_END_CP: i32 = 250;

pub const DEFAULT_DRAW_RATE_MILLIS: u32 = 420;

static DRAW_RATE_MILLIS: AtomicU32 = AtomicU32::new(DEFAULT_DRAW_RATE_MILLIS);

#[inline(always)]
pub fn draw_score(nodes: u64) -> i16 {
    if nodes & 2 == 0 {
        DRAW_BASE
    } else {
        DRAW_BASE + 1
    }
}

#[inline(always)]
pub fn draw_score_for_side(stm: Side, nodes: u64) -> i16 {
    let base = draw_score(nodes);
    if stm == Side::White {
        base
    } else if stm == Side::Black {
        base.saturating_neg()
    } else {
        base
    }
}

#[inline(always)]
pub fn attenuation_factor(score: i16) -> f32 {
    let a = (score as i32).abs() as f32;
    let start = ATTENUATION_START_CP as f32;
    let end = ATTENUATION_END_CP as f32;
    if a <= start {
        1.0
    } else if a >= end {
        0.0
    } else {
        1.0 - (a - start) / (end - start)
    }
}

#[inline(always)]
pub fn elo_to_contempt_cp(elo_diff: i32) -> i16 {
    let v = elo_diff / 4;
    v.clamp(-CONTEMPT_LIMIT as i32, CONTEMPT_LIMIT as i32) as i16
}

#[inline(always)]
pub fn elo_win_probability(elo_diff: i32) -> f32 {
    let e = elo_diff as f32;
    1.0 / (1.0 + 10.0f32.powf(-e / 400.0))
}

#[inline(always)]
pub fn contempt_from_elo_diff(score: i16, elo_diff: i32) -> i16 {
    let c = elo_to_contempt_cp(elo_diff);
    let att = attenuation_factor(score);
    ((c as f32) * att).round() as i16
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
    let att = attenuation_factor(score);
    if att <= 0.0 {
        return score;
    }
    let contempt = if stm == engine_side { c } else { -c };
    let scaled_contempt = ((contempt as f32) * att).round() as i16;
    let scaled_draw = ((d as f32) * att).round() as i16;
    score.saturating_add(scaled_draw).saturating_sub(scaled_contempt)
}

#[inline(always)]
pub fn apply_contempt_search(
    score: i16,
    stm: Side,
    engine_side: Side,
    contempt_cp: i16,
    draw_score_cp: i16,
) -> i16 {
    apply_contempt(score, stm, engine_side, contempt_cp, draw_score_cp)
}

#[inline(always)]
pub fn apply_contempt_display(
    search_score: i16,
    stm: Side,
    engine_side: Side,
    contempt_cp: i16,
) -> i16 {
    let c = contempt_cp.clamp(-CONTEMPT_LIMIT, CONTEMPT_LIMIT);
    if c == 0 {
        return search_score;
    }
    let att = attenuation_factor(search_score);
    if att <= 0.0 {
        return search_score;
    }
    let contempt = if stm == engine_side { c } else { -c };
    let scaled = ((contempt as f32) * att).round() as i16;
    search_score.saturating_add(scaled)
}

#[inline(always)]
pub fn get_draw_score(
    stm: Side,
    engine_side: Side,
    nodes: u64,
    contempt_cp: i16,
    draw_score_cp: i16,
) -> i16 {
    let base = draw_score_for_side(stm, nodes);
    apply_contempt_search(base, stm, engine_side, contempt_cp, draw_score_cp)
}

#[inline(always)]
pub fn get_display_draw_score(stm: Side, nodes: u64, draw_score_cp: i16) -> i16 {
    let d = draw_score_cp.clamp(-CONTEMPT_LIMIT, CONTEMPT_LIMIT);
    if stm == Side::White {
        draw_score(nodes).saturating_add(d)
    } else if stm == Side::Black {
        draw_score(nodes).saturating_neg().saturating_add(d)
    } else {
        draw_score(nodes).saturating_add(d)
    }
}

#[inline(always)]
pub fn search_draw_value(
    stm: Side,
    engine_side: Side,
    nodes: u64,
    contempt_cp: i16,
    draw_score_cp: i16,
) -> i16 {
    get_draw_score(stm, engine_side, nodes, contempt_cp, draw_score_cp)
}

#[inline(always)]
pub fn display_draw_value(stm: Side, nodes: u64, draw_score_cp: i16) -> i16 {
    get_display_draw_score(stm, nodes, draw_score_cp)
}

pub fn set_draw_rate(draw_rate: f32) {
    let v = (draw_rate.clamp(0.0, 1.0) * 1000.0).round() as u32;
    DRAW_RATE_MILLIS.store(v, Ordering::Relaxed);
}

#[inline(always)]
pub fn get_draw_rate() -> f32 {
    DRAW_RATE_MILLIS.load(Ordering::Relaxed) as f32 / 1000.0
}

pub fn cp_to_wdl_with_draw_rate(cp: i16, draw_rate: f32) -> (u16, u16, u16) {
    const MATE_CP: i32 = 29_000;
    let v = cp as i32;
    if v >= MATE_CP {
        return (1000, 0, 0);
    }
    if v <= -MATE_CP {
        return (0, 0, 1000);
    }
    let w = 1000.0 / (1.0 + (-f64::from(v) / 220.0).exp());
    let rate = f64::from(draw_rate.clamp(0.0, 1.0));
    let draw = (-(f64::from(v.abs()) / 180.0)).exp() * rate * 1000.0;
    let mut wi = w.round() as i32;
    let mut di = draw.round() as i32;
    wi = wi.clamp(0, 1000);
    di = di.clamp(0, 1000 - wi);
    let li = 1000 - wi - di;
    (wi as u16, di as u16, li as u16)
}

pub fn cp_to_wdl(cp: i16) -> (u16, u16, u16) {
    let rate = get_draw_rate();
    cp_to_wdl_with_draw_rate(cp, rate)
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
