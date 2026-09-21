#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PerfSample {
    pub nodes: u64,
    pub tbhits: u64,
    pub elapsed_ms: u128,
    pub behavior_us: u64,
    pub depth: u8,
}

impl PerfSample {
    pub fn nps(&self) -> u64 {
        if self.elapsed_ms == 0 {
            0
        } else {
            (self.nodes as f64 * 1000.0 / self.elapsed_ms as f64) as u64
        }
    }

    pub fn tt_hit_pct(&self) -> f64 {
        if self.nodes == 0 {
            0.0
        } else {
            self.tbhits as f64 * 100.0 / self.nodes as f64
        }
    }

    pub fn behavior_overhead_pct(&self) -> f64 {
        let total_us = self.elapsed_ms as f64 * 1000.0;
        if total_us <= 0.0 {
            0.0
        } else {
            self.behavior_us as f64 * 100.0 / total_us
        }
    }

    pub fn branching_estimate(&self) -> f64 {
        if self.depth <= 1 || self.nodes == 0 {
            0.0
        } else {
            (self.nodes as f64).powf(1.0 / self.depth as f64)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PerfBudget {
    pub max_behavior_overhead_pct: f64,
    pub min_nps: u64,
    pub max_branching: f64,
}

impl Default for PerfBudget {
    fn default() -> Self {
        Self {
            max_behavior_overhead_pct: 20.0,
            min_nps: 1,
            max_branching: 40.0,
        }
    }
}

pub fn within_budget(sample: &PerfSample, budget: &PerfBudget) -> bool {
    sample.behavior_overhead_pct() <= budget.max_behavior_overhead_pct
        && sample.nps() >= budget.min_nps
        && (sample.depth <= 1 || sample.branching_estimate() <= budget.max_branching)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_math() {
        let s = PerfSample {
            nodes: 2000,
            tbhits: 100,
            elapsed_ms: 1000,
            behavior_us: 50_000,
            depth: 4,
        };
        assert_eq!(s.nps(), 2000);
        assert!((s.tt_hit_pct() - 5.0).abs() < 1e-9);
        assert!((s.behavior_overhead_pct() - 5.0).abs() < 1e-9);
        assert!(s.branching_estimate() > 1.0 && s.branching_estimate() < 40.0);
    }

    #[test]
    fn budget_gate() {
        let ok = PerfSample {
            nodes: 5000,
            tbhits: 50,
            elapsed_ms: 1000,
            behavior_us: 10_000,
            depth: 4,
        };
        assert!(within_budget(&ok, &PerfBudget::default()));
        let heavy = PerfSample {
            behavior_us: 900_000,
            ..ok
        };
        assert!(!within_budget(&heavy, &PerfBudget::default()));
        let stalled = PerfSample { nodes: 0, ..ok };
        assert!(!within_budget(&stalled, &PerfBudget::default()));
    }

    #[test]
    fn empty_sample_is_zero() {
        let s = PerfSample::default();
        assert_eq!(s.nps(), 0);
        assert_eq!(s.tt_hit_pct(), 0.0);
        assert_eq!(s.behavior_overhead_pct(), 0.0);
        assert_eq!(s.branching_estimate(), 0.0);
    }
}
