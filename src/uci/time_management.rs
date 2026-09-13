use crate::common::constants;

pub fn calculate_move_time(clock: i32, increment: i32) -> i32 {
    calculate_move_time_with_moves(clock, increment, -1)
}

pub fn calculate_move_time_with_moves(clock: i32, increment: i32, movestogo: i32) -> i32 {
    calculate_optimum_with_ply(clock, increment, movestogo, 0).0
}

pub fn calculate_optimum_with_ply(
    clock: i32,
    increment: i32,
    movestogo: i32,
    ply: i32,
) -> (i32, i32) {
    if clock <= 0 {
        return (10, 10);
    }
    let move_overhead = constants::BUFFER_TIME as i32;
    let scaled_time = clock.max(1);
    let mut mtg = if movestogo > 0 { movestogo.min(50) } else { 50 };
    if scaled_time < 1000 && movestogo <= 0 {
        mtg = (scaled_time as f64 * 0.05) as i32;
        if mtg < 1 {
            mtg = 1;
        }
    }
    let time_left = (clock + increment * (mtg - 1) - move_overhead * (2 + mtg)).max(1);
    let (opt_scale, max_scale) = if movestogo <= 0 {
        let log_time = (scaled_time as f64 / 1000.0).log10();
        let opt_constant = (0.0029869 + 0.00033554 * log_time).min(0.004905);
        let max_constant = (3.3744 + 3.0608 * log_time).max(3.1441);
        let mut opt = (0.012112 + (ply as f64 + 3.22713).powf(0.46866) * opt_constant)
            .min(0.19404 * clock as f64 / time_left as f64);
        // originalTimeAdjust
        let mut original_adjust = 0.3272 * (time_left as f64).log10() - 0.4141;
        if original_adjust < 0.0 {
            original_adjust = 0.0;
        }
        if original_adjust != 0.0 {
            opt *= original_adjust;
        }
        let max = (6.873_f64).min(max_constant + ply as f64 / 12.352);
        (opt, max)
    } else {
        let opt =
            ((0.88 + ply as f64 / 116.4) / mtg as f64).min(0.88 * clock as f64 / time_left as f64);
        let max = 1.3 + 0.11 * mtg as f64;
        (opt, max)
    };
    let mut optimum = (opt_scale * time_left as f64).max(1.0) as i32;
    let mut maximum = (max_scale * optimum as f64)
        .min(0.8097 * clock as f64 - move_overhead as f64)
        .max(optimum as f64) as i32;
    optimum = optimum.max(10).min(clock - move_overhead);
    maximum = maximum.max(optimum).min(clock - move_overhead).max(10);
    (optimum, maximum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_manage_time_without_exhausting(starting_time: i32, increment: i32) {
        let max_moves = if increment > 0 { 400 } else { 75 };
        let position_parse_delay = 5;
        let network_delay = 20;
        let engine_cancel_delay = 1;

        let mut remaining_time = starting_time;

        for _move_number in 1..=max_moves {
            let move_time = calculate_move_time(remaining_time, increment);
            assert!(
                move_time >= 10,
                "Allocated time {move_time}ms is less than minimum 10ms"
            );

            remaining_time -=
                move_time + position_parse_delay + network_delay + engine_cancel_delay;
            remaining_time += increment;
            assert!(
                remaining_time > 0,
                "Ran out of time. Remaining: {remaining_time}ms"
            );
        }
    }

    macro_rules! time_case {
        ($name:ident, $starting_time:expr, $increment:expr) => {
            #[test]
            fn $name() {
                assert_manage_time_without_exhausting($starting_time, $increment);
            }
        };
    }

    time_case!(case_180000_2000, 180000, 2000);
    time_case!(case_300000_0, 300000, 0);
    time_case!(case_600000_5000, 600000, 5000);
    time_case!(case_60000_0, 60000, 0);
    time_case!(case_30000_0, 30000, 0);
    time_case!(case_15000_100, 15000, 100);
    time_case!(case_30000_100, 30000, 100);
    time_case!(case_10000_10000, 10000, 10000);
    time_case!(case_5000_20000, 5000, 20000);
    time_case!(case_60000_60000, 60000, 60000);
    time_case!(case_0_10000, 0, 10000);
}
