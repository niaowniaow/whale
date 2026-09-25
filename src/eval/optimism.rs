pub const OPTIMISM_GAIN: i32 = 114;
pub const OPTIMISM_DIVISOR: i32 = 85;
pub const INTERNAL_TO_SF_NUM: i32 = 208;
pub const INTERNAL_TO_SF_DEN: i32 = 100;

pub fn optimism_from_average(avg_sf: i32) -> i32 {
    if avg_sf == 0 {
        0
    } else {
        OPTIMISM_GAIN * avg_sf / (avg_sf.abs() + OPTIMISM_DIVISOR)
    }
}

pub fn to_sf_scale(internal_cp: i32) -> i32 {
    internal_cp * INTERNAL_TO_SF_NUM / INTERNAL_TO_SF_DEN
}

pub fn mirrored_pair(opt_us: i32) -> [i32; 2] {
    [opt_us, -opt_us]
}

#[derive(Default)]
pub struct ScoreAverage {
    sum: i64,
    count: u32,
}

impl ScoreAverage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, score_cp: i32) {
        self.sum += score_cp as i64;
        self.count = self.count.saturating_add(1);
    }

    pub fn average(&self) -> i32 {
        if self.count == 0 {
            0
        } else {
            (self.sum / self.count as i64) as i32
        }
    }

    pub fn optimism(&self) -> i32 {
        optimism_from_average(to_sf_scale(self.average()))
    }

    pub fn pair(&self) -> [i32; 2] {
        mirrored_pair(self.optimism())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_average_gives_zero() {
        assert_eq!(optimism_from_average(0), 0);
    }

    #[test]
    fn symmetric_around_zero() {
        for avg in [1, 85, 200, 1500, 30000] {
            assert_eq!(optimism_from_average(avg), -optimism_from_average(-avg));
        }
    }

    #[test]
    fn saturates_below_gain() {
        assert_eq!(optimism_from_average(100000), 113);
        assert_eq!(optimism_from_average(-100000), -113);
    }

    #[test]
    fn small_values() {
        assert_eq!(optimism_from_average(1), 1);
        assert_eq!(optimism_from_average(85), 57);
    }

    #[test]
    fn scale_conversion() {
        assert_eq!(to_sf_scale(100), 208);
        assert_eq!(to_sf_scale(-50), -104);
        assert_eq!(to_sf_scale(0), 0);
    }

    #[test]
    fn pair_mirrors() {
        assert_eq!(mirrored_pair(37), [37, -37]);
        assert_eq!(mirrored_pair(0), [0, 0]);
    }

    #[test]
    fn tracker_empty() {
        let tracker = ScoreAverage::new();
        assert_eq!(tracker.average(), 0);
        assert_eq!(tracker.optimism(), 0);
        assert_eq!(tracker.pair(), [0, 0]);
    }

    #[test]
    fn tracker_averages_scores() {
        let mut tracker = ScoreAverage::new();
        tracker.push(100);
        tracker.push(100);
        tracker.push(100);
        assert_eq!(tracker.average(), 100);
        assert_eq!(tracker.optimism(), optimism_from_average(208));
    }

    #[test]
    fn tracker_negative_mirrors_positive() {
        let mut pos = ScoreAverage::new();
        pos.push(200);
        pos.push(100);
        let mut neg = ScoreAverage::new();
        neg.push(-200);
        neg.push(-100);
        assert_eq!(pos.average(), -neg.average());
        assert_eq!(pos.optimism(), -neg.optimism());
        assert_eq!(pos.pair(), [pos.optimism(), neg.optimism()]);
    }
}
