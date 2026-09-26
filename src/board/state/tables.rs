use crate::common::constants::{SIDES, SQUARES};

#[rustfmt::skip]
pub const CASTLING_CONSTANTS: [u8; SQUARES] = [
     7, 15, 15, 15,  3, 15, 15, 11,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    13, 15, 15, 15, 12, 15, 15, 14,
];

const fn generate_passed_pawn_masks() -> [[u64; SQUARES]; SIDES] {
    let mut masks = [[0u64; SQUARES]; SIDES];
    let mut sq = 0;
    while sq < 64 {
        let file = (sq % 8) as i32;
        let rank = (sq / 8) as i32;

        let mut w_mask = 0u64;
        let mut r = 0;
        while r < rank {
            let mut f = file - 1;
            while f <= file + 1 {
                if f >= 0 && f <= 7 {
                    w_mask |= 1u64 << (r * 8 + f);
                }
                f += 1;
            }
            r += 1;
        }
        masks[0][sq] = w_mask;

        let mut b_mask = 0u64;
        let mut r = rank + 1;
        while r <= 7 {
            let mut f = file - 1;
            while f <= file + 1 {
                if f >= 0 && f <= 7 {
                    b_mask |= 1u64 << (r * 8 + f);
                }
                f += 1;
            }
            r += 1;
        }
        masks[1][sq] = b_mask;

        sq += 1;
    }
    masks
}

pub const PASSED_PAWN_MASKS: [[u64; SQUARES]; SIDES] = generate_passed_pawn_masks();

const fn generate_ray_tables() -> ([[u64; SQUARES]; SQUARES], [[u64; SQUARES]; SQUARES]) {
    let mut between = [[0u64; SQUARES]; SQUARES];
    let mut line = [[0u64; SQUARES]; SQUARES];
    let mut sq1 = 0;
    while sq1 < 64 {
        let r1 = sq1 as i32 / 8;
        let f1 = sq1 as i32 % 8;
        let mut sq2 = 0;
        while sq2 < 64 {
            if sq1 != sq2 {
                let r2 = sq2 as i32 / 8;
                let f2 = sq2 as i32 % 8;
                let dr = if r2 > r1 {
                    1
                } else if r2 < r1 {
                    -1
                } else {
                    0
                };
                let df = if f2 > f1 {
                    1
                } else if f2 < f1 {
                    -1
                } else {
                    0
                };
                let dy = if r2 >= r1 { r2 - r1 } else { r1 - r2 };
                let dx = if f2 >= f1 { f2 - f1 } else { f1 - f2 };
                if dr == 0 || df == 0 || dy == dx {
                    let mut curr_r = r1 + dr;
                    let mut curr_f = f1 + df;
                    let mut b_mask = 0u64;
                    while curr_r != r2 || curr_f != f2 {
                        b_mask |= 1u64 << (curr_r * 8 + curr_f);
                        curr_r += dr;
                        curr_f += df;
                    }
                    between[sq1][sq2] = b_mask;

                    let mut l_mask = 0u64;
                    let mut r = r1;
                    let mut f = f1;
                    while r >= 0 && r < 8 && f >= 0 && f < 8 {
                        l_mask |= 1u64 << (r * 8 + f);
                        r += dr;
                        f += df;
                    }
                    r = r1 - dr;
                    f = f1 - df;
                    while r >= 0 && r < 8 && f >= 0 && f < 8 {
                        l_mask |= 1u64 << (r * 8 + f);
                        r -= dr;
                        f -= df;
                    }
                    line[sq1][sq2] = l_mask;
                }
            }
            sq2 += 1;
        }
        sq1 += 1;
    }
    (between, line)
}

pub static RAY_TABLES: ([[u64; SQUARES]; SQUARES], [[u64; SQUARES]; SQUARES]) =
    generate_ray_tables();
pub static BETWEEN_BB: &[[u64; SQUARES]; SQUARES] = &RAY_TABLES.0;
pub static LINE_BB: &[[u64; SQUARES]; SQUARES] = &RAY_TABLES.1;
