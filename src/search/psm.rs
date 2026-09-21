use crate::common::constants::MAX_PLY;

pub const PSM_HIDDEN_DIM: usize = 128;

#[derive(Clone, Copy, Debug)]
pub struct PsmHiddenState {
    pub hidden: [i16; PSM_HIDDEN_DIM],
    pub consecutive_fail_lows: u8,
}

impl Default for PsmHiddenState {
    fn default() -> Self {
        Self {
            hidden: [0; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 0,
        }
    }
}

impl PsmHiddenState {
    pub const fn new() -> Self {
        Self {
            hidden: [0; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 0,
        }
    }

    pub fn reset(&mut self) {
        self.hidden.fill(0);
        self.consecutive_fail_lows = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PsmFeatures {
    pub static_eval: i16,
    pub depth: u8,
    pub alpha: i16,
    pub beta: i16,
    pub move_history: i32,
    pub is_capture: bool,
    pub sibling_index: usize,
    pub failed_low: bool,
}

pub struct PsmTracker;

impl PsmTracker {
    pub fn step(parent: &PsmHiddenState, features: &PsmFeatures) -> PsmHiddenState {
        let tilt = ((features.static_eval.saturating_sub(features.alpha)) as i32 / 32)
            .clamp(-64, 64)
            + ((features.static_eval.saturating_sub(features.beta)) as i32 / 32).clamp(-64, 64);
        let load = ((features.beta.saturating_sub(features.alpha)) as i32 / 32).clamp(0, 64)
            + (features.depth.min(16) as i32) * 2
            + (features.sibling_index.min(32) as i32);
        let base = tilt - load / 2
            + if features.is_capture { 16 } else { 0 }
            + (features.move_history / 1024).clamp(-16, 16);

        let mut next_hidden = [0i16; PSM_HIDDEN_DIM];
        for (i, h) in next_hidden.iter_mut().enumerate() {
            let lane = ((i % 16) as i32 - 8) * 2;
            *h = ((parent.hidden[i] as i32 * 3 + base + lane) / 4).clamp(-256, 256) as i16;
        }

        let fails = if features.failed_low {
            parent.consecutive_fail_lows.saturating_add(1).min(32)
        } else {
            0
        };

        PsmHiddenState {
            hidden: next_hidden,
            consecutive_fail_lows: fails,
        }
    }

    pub fn readout(state: &PsmHiddenState) -> i16 {
        let mut sum = 0i32;
        for &val in state.hidden.iter() {
            sum += val as i32;
        }
        (sum / PSM_HIDDEN_DIM as i32 / 4).clamp(-60, 60) as i16
    }

    pub fn should_prune_sibling(
        parent: &PsmHiddenState,
        sibling_index: usize,
        depth: u8,
        consecutive_fail_lows: u8,
    ) -> bool {
        if depth > 4 || sibling_index < 6 {
            return false;
        }

        let total_fails = parent.consecutive_fail_lows.max(consecutive_fail_lows);
        if total_fails >= 4 && sibling_index >= 8 {
            return true;
        }

        if total_fails >= 2 && sibling_index >= 12 {
            let readout = Self::readout(parent);
            if readout < -15 {
                return true;
            }
        }

        false
    }
}

pub use PsmTracker as PsmEngine;

pub struct PsmStack {
    pub stack: [PsmHiddenState; MAX_PLY],
}

impl Default for PsmStack {
    fn default() -> Self {
        Self {
            stack: [PsmHiddenState::new(); MAX_PLY],
        }
    }
}

impl PsmStack {
    pub const fn new() -> Self {
        Self {
            stack: [PsmHiddenState::new(); MAX_PLY],
        }
    }

    pub fn reset(&mut self) {
        for s in self.stack.iter_mut() {
            s.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psm_initial_state() {
        let state = PsmHiddenState::new();
        assert_eq!(state.consecutive_fail_lows, 0);
        assert_eq!(state.hidden[0], 0);
        assert_eq!(PsmEngine::readout(&state), 0);
    }

    #[test]
    fn test_psm_step_and_readout() {
        let parent = PsmHiddenState::new();
        let features = PsmFeatures {
            static_eval: 100,
            depth: 6,
            alpha: 50,
            beta: 150,
            move_history: 300,
            is_capture: true,
            sibling_index: 1,
            failed_low: false,
        };

        let next = PsmEngine::step(&parent, &features);
        let readout = PsmEngine::readout(&next);
        assert!(readout.abs() <= 60);
    }

    #[test]
    fn test_psm_prune_sibling_trigger() {
        let mut parent = PsmHiddenState::new();
        parent.consecutive_fail_lows = 5;

        assert!(PsmEngine::should_prune_sibling(&parent, 9, 3, 5));
        assert!(!PsmEngine::should_prune_sibling(&parent, 3, 3, 5));
        assert!(!PsmEngine::should_prune_sibling(&parent, 9, 8, 5));
    }

    #[test]
    fn hidden_state_reset_and_defaults() {
        let mut state = PsmHiddenState {
            hidden: [7; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 9,
        };
        state.reset();
        assert_eq!(state.hidden, [0; PSM_HIDDEN_DIM]);
        assert_eq!(state.consecutive_fail_lows, 0);
        assert_eq!(PsmHiddenState::default().hidden, [0; PSM_HIDDEN_DIM]);
        let mut stack = PsmStack::new();
        stack.stack[0].hidden[0] = 5;
        stack.reset();
        assert_eq!(stack.stack[0].hidden[0], 0);
        let _ = PsmStack::default();
    }

    #[test]
    fn step_tracks_fail_lows_with_saturation() {
        let parent = PsmHiddenState {
            hidden: [0; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 31,
        };
        let base = PsmFeatures {
            static_eval: 0,
            depth: 3,
            alpha: 0,
            beta: 50,
            move_history: 0,
            is_capture: false,
            sibling_index: 0,
            failed_low: true,
        };
        assert_eq!(PsmEngine::step(&parent, &base).consecutive_fail_lows, 32);
        let saturated = PsmHiddenState {
            hidden: [0; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 32,
        };
        assert_eq!(PsmEngine::step(&saturated, &base).consecutive_fail_lows, 32);
        let ok = PsmFeatures {
            failed_low: false,
            ..base
        };
        assert_eq!(PsmEngine::step(&parent, &ok).consecutive_fail_lows, 0);
    }

    #[test]
    fn prune_second_branch_needs_negative_readout() {
        let negative = PsmHiddenState {
            hidden: [-256; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 2,
        };
        assert!(PsmEngine::readout(&negative) < -15);
        assert!(PsmEngine::should_prune_sibling(&negative, 12, 3, 0));
        let neutral = PsmHiddenState::new();
        assert_eq!(PsmEngine::readout(&neutral), 0);
        assert!(!PsmEngine::should_prune_sibling(&neutral, 12, 3, 2));
        assert!(!PsmEngine::should_prune_sibling(&negative, 11, 3, 2));
        let cold = PsmHiddenState {
            hidden: [-256; PSM_HIDDEN_DIM],
            consecutive_fail_lows: 0,
        };
        assert!(!PsmEngine::should_prune_sibling(&cold, 12, 3, 1));
    }

    #[test]
    fn readout_stays_bounded_on_extremes() {
        let parent = PsmHiddenState::new();
        let lo = PsmFeatures {
            static_eval: -30_000,
            depth: 64,
            alpha: -30_000,
            beta: 30_000,
            move_history: -1_000_000,
            is_capture: true,
            sibling_index: 500,
            failed_low: false,
        };
        let hi = PsmFeatures {
            static_eval: 30_000,
            depth: 64,
            alpha: -30_000,
            beta: 30_000,
            move_history: 1_000_000,
            is_capture: true,
            sibling_index: 500,
            failed_low: true,
        };
        for next in [PsmEngine::step(&parent, &lo), PsmEngine::step(&parent, &hi)] {
            assert!(PsmEngine::readout(&next).abs() <= 60);
            assert!(next.hidden.iter().all(|&v| (-256..=256).contains(&v)));
        }
    }

    #[test]
    fn lanes_converge_toward_signal() {
        let parent = PsmHiddenState::new();
        let hot = PsmFeatures {
            static_eval: 500,
            depth: 2,
            alpha: 0,
            beta: 100,
            move_history: 0,
            is_capture: false,
            sibling_index: 0,
            failed_low: false,
        };
        let warm = PsmEngine::step(&parent, &hot);
        let warmer = PsmEngine::step(&warm, &hot);
        assert!(PsmEngine::readout(&warmer) >= PsmEngine::readout(&warm));
    }
}
