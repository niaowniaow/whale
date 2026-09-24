pub fn calculate_move_time(clock: i32, increment: i32) -> i32 {
    calculate_move_time_with_moves(clock, increment, -1)
}

pub fn calculate_move_time_with_moves(clock: i32, increment: i32, movestogo: i32) -> i32 {
    calculate_optimum_with_ply(clock, increment, movestogo, 0, -1, 50, false).0
}

pub fn calculate_optimum_with_ply(
    clock: i32,
    increment: i32,
    movestogo: i32,
    ply: i32,
    opp_clock: i32,
    move_overhead: i32,
    ponder: bool,
) -> (i32, i32) {
    if clock <= move_overhead {
        return (10, 10);
    }

    let available_clock = (clock - move_overhead).max(10);
    let (format_opt_cap, format_max_cap) = if available_clock <= 15_000 {
        (
            available_clock / 15 + increment / 2,
            available_clock / 8 + increment,
        )
    } else if available_clock <= 60_000 {
        (
            1_500.min(available_clock / 20) + increment / 2,
            3_000.min(available_clock / 10) + increment,
        )
    } else if available_clock <= 300_000 {
        (
            4_500.min(available_clock / 30) + increment / 2,
            8_000.min(available_clock / 15) + increment,
        )
    } else if available_clock <= 900_000 {
        (
            10_000.min(available_clock / 35) + increment / 2,
            18_000.min(available_clock / 20) + increment,
        )
    } else {
        (
            20_000.min(available_clock / 40) + increment / 2,
            35_000.min(available_clock / 25) + increment,
        )
    };

    let scaled_time = clock.max(1);
    let mut mtg = if movestogo > 0 { movestogo.min(50) } else { 50 };
    if scaled_time < 1000 && movestogo <= 0 {
        mtg = ((scaled_time as f64 * 0.05) as i32).max(1);
    }
    let time_left = (clock + increment * (mtg - 1) - move_overhead * (2 + mtg)).max(1);
    let (mut opt_scale, max_scale) = if movestogo <= 0 {
        let log_time = (scaled_time as f64 / 1000.0).log10();
        let opt_constant = (0.0029869 + 0.00033554 * log_time).min(0.004905);
        let max_constant = (3.3744 + 3.0608 * log_time).max(3.1441);
        let mut opt = (0.012112 + (ply as f64 + 3.22713).powf(0.46866) * opt_constant)
            .min(0.19404 * clock as f64 / time_left as f64);
        let original_adjust = 0.3272 * (time_left as f64).log10() - 0.4141;
        opt *= original_adjust;
        let max = (2.5_f64).min(max_constant + ply as f64 / 12.352);
        (opt, max)
    } else {
        let opt =
            ((0.88 + ply as f64 / 116.4) / mtg as f64).min(0.88 * clock as f64 / time_left as f64);
        let max = (1.3 + 0.11 * mtg as f64).min(2.5);
        (opt, max)
    };

    if movestogo != 1 && opp_clock > 0 {
        let time_advantage =
            (clock as f64 - opp_clock as f64) / (1.0 + clock as f64 + opp_clock as f64);
        opt_scale *= 1.0 + 0.9 * time_advantage.min(0.0);
    }

    let raw_optimum = (opt_scale * time_left as f64).max(1.0) as i32;
    let base_optimum = raw_optimum.min(format_opt_cap.max(10)).max(10);
    let raw_max = (max_scale * base_optimum as f64) as i32;
    let maximum = raw_max
        .min(format_max_cap.max(base_optimum))
        .min((clock - move_overhead - 5).max(base_optimum))
        .max(base_optimum);

    let mut optimum = base_optimum;
    if ponder {
        optimum += optimum / 4;
    }
    (optimum.max(10), maximum.max(10))
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

    #[test]
    fn ponder_bonus_only_changes_optimum() {
        for movestogo in [-1, 1, 30] {
            let (optimum, maximum) =
                calculate_optimum_with_ply(60000, 1000, movestogo, 20, 60000, 10, false);
            let (ponder_optimum, ponder_maximum) =
                calculate_optimum_with_ply(60000, 1000, movestogo, 20, 60000, 10, true);
            assert_eq!(ponder_optimum, optimum + optimum / 4);
            assert_eq!(ponder_maximum, maximum);
        }
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
