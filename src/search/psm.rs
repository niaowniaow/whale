use crate::common::constants::MAX_PLY;

pub const PSM_HIDDEN_DIM: usize = 128;
pub const PSM_INPUT_DIM: usize = 8;

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

pub struct PsmEngine;

impl PsmEngine {
    pub fn step(parent: &PsmHiddenState, features: &PsmFeatures) -> PsmHiddenState {
        let norm_eval = (features.static_eval / 32).clamp(-128, 128) as i32;
        let norm_depth = (features.depth as i32).clamp(0, 64);
        let norm_window =
            ((features.beta.saturating_sub(features.alpha)) / 32).clamp(0, 128) as i32;
        let norm_history = (features.move_history / 512).clamp(-128, 128);
        let is_cap = if features.is_capture { 64 } else { 0 };
        let norm_sibling = (features.sibling_index as i32).clamp(0, 64);
        let alpha_diff =
            ((features.static_eval.saturating_sub(features.alpha)) / 32).clamp(-128, 128) as i32;
        let beta_diff =
            ((features.static_eval.saturating_sub(features.beta)) / 32).clamp(-128, 128) as i32;

        let x = [
            norm_eval,
            norm_depth,
            norm_window,
            norm_history,
            is_cap,
            norm_sibling,
            alpha_diff,
            beta_diff,
        ];

        let mut next_hidden = [0i16; PSM_HIDDEN_DIM];

        for i in 0..PSM_HIDDEN_DIM {
            let p_val = parent.hidden[i] as i32;
            let mut gate_input = 0i32;
            for j in 0..PSM_INPUT_DIM {
                let weight = PSM_GATE_WEIGHTS[(i + j * 7) % PSM_GATE_WEIGHTS.len()];
                gate_input += x[j] * weight;
            }

            let update_gate = ((gate_input + p_val * 3) / 16).clamp(-128, 128);
            let candidate = ((gate_input * 2 - p_val) / 20).clamp(-128, 128);

            let z = (update_gate + 128).clamp(0, 256);
            let blended = (p_val * (256 - z) + candidate * z) / 256;

            next_hidden[i] = blended.clamp(-256, 256) as i16;
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
        for (i, &val) in state.hidden.iter().enumerate() {
            let w = PSM_READOUT_WEIGHTS[i % PSM_READOUT_WEIGHTS.len()];
            sum += (val as i32) * w;
        }

        (sum / 2048).clamp(-60, 60) as i16
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

const PSM_GATE_WEIGHTS: [i32; 32] = [
    3, -2, 4, 1, -3, 2, 5, -1, 2, -4, 3, 2, -2, 4, 1, -3, 4, -1, 2, 3, -4, 2, 1, -2, 3, 1, -3, 2,
    4, -2, 3, 1,
];

const PSM_READOUT_WEIGHTS: [i32; 16] = [5, -3, 6, 2, -4, 3, 7, -2, 4, -5, 3, 4, -3, 6, 2, -4];

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
}
