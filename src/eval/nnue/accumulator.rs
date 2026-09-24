use super::ACC_SIZE;
use super::loader::Network;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Accumulator {
    pub state: [i16; ACC_SIZE],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Accumulators {
    pub white: Accumulator,
    pub black: Accumulator,
}

impl Accumulators {
    #[inline(always)]
    pub fn push_from(&mut self, parent: &Self) {
        self.white.push_from(&parent.white);
        self.black.push_from(&parent.black);
    }
    #[inline(always)]
    pub fn is_accurate_flag(&self, flag: bool) -> bool {
        flag
    }
}

impl Accumulator {
    pub fn new() -> Self {
        Self {
            state: [0; ACC_SIZE],
        }
    }

    #[inline(always)]
    pub fn copy_from_other(&mut self, other: &Self) {
        self.state.copy_from_slice(&other.state);
    }

    #[inline(always)]
    pub fn push_from(&mut self, parent: &Self) {
        self.copy_from_other(parent);
    }

    #[inline(always)]
    pub fn pop_to(&mut self, parent: &mut Self) {
        parent.copy_from_other(self);
    }

    #[inline(always)]
    pub fn init_with_biases(&mut self, network: &Network) {
        self.state.copy_from_slice(&network.transformer_biases);
    }

    #[inline(always)]
    pub fn add_feature(&mut self, feature_idx: usize, network: &Network) {
        let start = feature_idx * ACC_SIZE;
        let weights = &network.transformer_weights[start..start + ACC_SIZE];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::add_feature_avx2(&mut self.state, weights);
                }
                return;
            }
        }
        for (state, weight) in self.state.iter_mut().zip(weights) {
            *state += *weight;
        }
    }

    #[inline(always)]
    pub fn remove_feature(&mut self, feature_idx: usize, network: &Network) {
        let start = feature_idx * ACC_SIZE;
        let weights = &network.transformer_weights[start..start + ACC_SIZE];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::remove_feature_avx2(&mut self.state, weights);
                }
                return;
            }
        }
        for (state, weight) in self.state.iter_mut().zip(weights) {
            *state -= *weight;
        }
    }

    #[inline(always)]
    pub fn add_1_sub_1(&mut self, add_idx: usize, remove_idx: usize, network: &Network) {
        let add_start = add_idx * ACC_SIZE;
        let remove_start = remove_idx * ACC_SIZE;
        let add_weights = &network.transformer_weights[add_start..add_start + ACC_SIZE];
        let remove_weights = &network.transformer_weights[remove_start..remove_start + ACC_SIZE];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::add_1_sub_1_avx2(&mut self.state, add_weights, remove_weights);
                }
                return;
            }
        }
        for i in 0..ACC_SIZE {
            self.state[i] += add_weights[i] - remove_weights[i];
        }
    }

    #[inline(always)]
    pub fn add_1_sub_2(
        &mut self,
        add_idx: usize,
        remove_idx1: usize,
        remove_idx2: usize,
        network: &Network,
    ) {
        let add_start = add_idx * ACC_SIZE;
        let remove1_start = remove_idx1 * ACC_SIZE;
        let remove2_start = remove_idx2 * ACC_SIZE;
        let add_weights = &network.transformer_weights[add_start..add_start + ACC_SIZE];
        let remove1_weights = &network.transformer_weights[remove1_start..remove1_start + ACC_SIZE];
        let remove2_weights = &network.transformer_weights[remove2_start..remove2_start + ACC_SIZE];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::add_1_sub_2_avx2(
                        &mut self.state,
                        add_weights,
                        remove1_weights,
                        remove2_weights,
                    );
                }
                return;
            }
        }
        for i in 0..ACC_SIZE {
            self.state[i] += add_weights[i] - remove1_weights[i] - remove2_weights[i];
        }
    }

    #[inline(always)]
    pub fn add_2_sub_2(
        &mut self,
        add_idx1: usize,
        add_idx2: usize,
        remove_idx1: usize,
        remove_idx2: usize,
        network: &Network,
    ) {
        let add1_start = add_idx1 * ACC_SIZE;
        let add2_start = add_idx2 * ACC_SIZE;
        let remove1_start = remove_idx1 * ACC_SIZE;
        let remove2_start = remove_idx2 * ACC_SIZE;
        let a1 = &network.transformer_weights[add1_start..add1_start + ACC_SIZE];
        let a2 = &network.transformer_weights[add2_start..add2_start + ACC_SIZE];
        let r1 = &network.transformer_weights[remove1_start..remove1_start + ACC_SIZE];
        let r2 = &network.transformer_weights[remove2_start..remove2_start + ACC_SIZE];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    Self::add_2_sub_2_avx2(&mut self.state, a1, a2, r1, r2);
                }
                return;
            }
        }
        for i in 0..ACC_SIZE {
            self.state[i] += a1[i] + a2[i] - r1[i] - r2[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_feature_avx2(state: &mut [i16; ACC_SIZE], weights: &[i16]) {
        unsafe {
            use std::arch::x86_64::*;
            let s_ptr = state.as_mut_ptr() as *mut __m256i;
            let w_ptr = weights.as_ptr() as *const __m256i;
            for i in 0..16 {
                let s = _mm256_load_si256(s_ptr.add(i));
                let w = _mm256_loadu_si256(w_ptr.add(i));
                _mm256_store_si256(s_ptr.add(i), _mm256_add_epi16(s, w));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn remove_feature_avx2(state: &mut [i16; ACC_SIZE], weights: &[i16]) {
        unsafe {
            use std::arch::x86_64::*;
            let s_ptr = state.as_mut_ptr() as *mut __m256i;
            let w_ptr = weights.as_ptr() as *const __m256i;
            for i in 0..16 {
                let s = _mm256_load_si256(s_ptr.add(i));
                let w = _mm256_loadu_si256(w_ptr.add(i));
                _mm256_store_si256(s_ptr.add(i), _mm256_sub_epi16(s, w));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_1_sub_1_avx2(
        state: &mut [i16; ACC_SIZE],
        add_weights: &[i16],
        remove_weights: &[i16],
    ) {
        unsafe {
            use std::arch::x86_64::*;
            let s_ptr = state.as_mut_ptr() as *mut __m256i;
            let a_ptr = add_weights.as_ptr() as *const __m256i;
            let r_ptr = remove_weights.as_ptr() as *const __m256i;
            for i in 0..16 {
                let s = _mm256_load_si256(s_ptr.add(i));
                let a = _mm256_loadu_si256(a_ptr.add(i));
                let r = _mm256_loadu_si256(r_ptr.add(i));
                let diff = _mm256_sub_epi16(a, r);
                _mm256_store_si256(s_ptr.add(i), _mm256_add_epi16(s, diff));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_1_sub_2_avx2(
        state: &mut [i16; ACC_SIZE],
        add_weights: &[i16],
        rem1_weights: &[i16],
        rem2_weights: &[i16],
    ) {
        unsafe {
            use std::arch::x86_64::*;
            let s_ptr = state.as_mut_ptr() as *mut __m256i;
            let a_ptr = add_weights.as_ptr() as *const __m256i;
            let r1_ptr = rem1_weights.as_ptr() as *const __m256i;
            let r2_ptr = rem2_weights.as_ptr() as *const __m256i;
            for i in 0..16 {
                let s = _mm256_load_si256(s_ptr.add(i));
                let a = _mm256_loadu_si256(a_ptr.add(i));
                let r1 = _mm256_loadu_si256(r1_ptr.add(i));
                let r2 = _mm256_loadu_si256(r2_ptr.add(i));
                let rem = _mm256_add_epi16(r1, r2);
                let diff = _mm256_sub_epi16(a, rem);
                _mm256_store_si256(s_ptr.add(i), _mm256_add_epi16(s, diff));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_2_sub_2_avx2(
        state: &mut [i16; ACC_SIZE],
        a1_weights: &[i16],
        a2_weights: &[i16],
        r1_weights: &[i16],
        r2_weights: &[i16],
    ) {
        unsafe {
            use std::arch::x86_64::*;
            let s_ptr = state.as_mut_ptr() as *mut __m256i;
            let a1_ptr = a1_weights.as_ptr() as *const __m256i;
            let a2_ptr = a2_weights.as_ptr() as *const __m256i;
            let r1_ptr = r1_weights.as_ptr() as *const __m256i;
            let r2_ptr = r2_weights.as_ptr() as *const __m256i;
            for i in 0..16 {
                let s = _mm256_load_si256(s_ptr.add(i));
                let a1 = _mm256_loadu_si256(a1_ptr.add(i));
                let a2 = _mm256_loadu_si256(a2_ptr.add(i));
                let r1 = _mm256_loadu_si256(r1_ptr.add(i));
                let r2 = _mm256_loadu_si256(r2_ptr.add(i));
                let adds = _mm256_add_epi16(a1, a2);
                let rems = _mm256_add_epi16(r1, r2);
                let diff = _mm256_sub_epi16(adds, rems);
                _mm256_store_si256(s_ptr.add(i), _mm256_add_epi16(s, diff));
            }
        }
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::state::BoardState;
    use crate::common::side::Side;

    #[test]
    fn test_accumulator_new() {
        let acc = Accumulator::new();
        assert_eq!(acc.state, [0i16; ACC_SIZE]);
    }

    #[test]
    fn test_accumulator_incremental_updates() {
        let mut network = Network::new_boxed();
        network.transformer_biases.fill(10);

        for i in 0..ACC_SIZE {
            network.transformer_weights[5 * ACC_SIZE + i] = 1;
        }

        let mut acc = Accumulator::new();
        acc.init_with_biases(&network);
        assert_eq!(acc.state, [10i16; ACC_SIZE]);

        acc.add_feature(5, &network);
        assert_eq!(acc.state, [11i16; ACC_SIZE]);

        acc.remove_feature(5, &network);
        assert_eq!(acc.state, [10i16; ACC_SIZE]);
    }

    #[test]
    fn test_accumulator_refresh_starting_position() {
        let mut network = Network::new_boxed();
        network.transformer_biases.fill(5);
        network.transformer_weights.fill(1);

        let mut board = BoardState::default();
        board.refresh_accumulator(Side::White, &network);

        let idx = board.history.index;
        assert_ne!(board.history.accumulators[idx].white.state[0], 5);
        assert_ne!(board.history.accumulators[idx].white.state[0], 0);
    }

    #[test]
    fn test_accumulator_add_2_sub_2() {
        let mut network = Network::new_boxed();
        network.transformer_biases.fill(100);

        for i in 0..ACC_SIZE {
            network.transformer_weights[ACC_SIZE + i] = 5;
            network.transformer_weights[2 * ACC_SIZE + i] = 12;
            network.transformer_weights[3 * ACC_SIZE + i] = -3;
            network.transformer_weights[4 * ACC_SIZE + i] = 8;
        }

        let mut acc_expected = Accumulator::new();
        acc_expected.init_with_biases(&network);
        acc_expected.add_feature(1, &network);
        acc_expected.add_feature(2, &network);
        acc_expected.remove_feature(3, &network);
        acc_expected.remove_feature(4, &network);

        let mut acc_actual = Accumulator::new();
        acc_actual.init_with_biases(&network);
        acc_actual.add_2_sub_2(1, 2, 3, 4, &network);

        assert_eq!(acc_expected.state, acc_actual.state);
    }

    #[test]
    fn test_accumulator_add_1_sub_1_matches_sequential() {
        let mut network = Network::new_boxed();
        network.transformer_biases.fill(7);
        for i in 0..ACC_SIZE {
            network.transformer_weights[3 * ACC_SIZE + i] = 9;
            network.transformer_weights[4 * ACC_SIZE + i] = -4;
        }

        let mut expected = Accumulator::new();
        expected.init_with_biases(&network);
        expected.add_feature(3, &network);
        expected.remove_feature(4, &network);

        let mut actual = Accumulator::new();
        actual.init_with_biases(&network);
        actual.add_1_sub_1(3, 4, &network);

        assert_eq!(expected.state, actual.state);
    }

    #[test]
    fn test_accumulator_add_1_sub_2_matches_sequential() {
        let mut network = Network::new_boxed();
        network.transformer_biases.fill(-3);
        for i in 0..ACC_SIZE {
            network.transformer_weights[ACC_SIZE + i] = 6;
            network.transformer_weights[2 * ACC_SIZE + i] = 2;
            network.transformer_weights[3 * ACC_SIZE + i] = 1;
        }

        let mut expected = Accumulator::new();
        expected.init_with_biases(&network);
        expected.add_feature(1, &network);
        expected.remove_feature(2, &network);
        expected.remove_feature(3, &network);

        let mut actual = Accumulator::new();
        actual.init_with_biases(&network);
        actual.add_1_sub_2(1, 2, 3, &network);

        assert_eq!(expected.state, actual.state);
    }

    #[test]
    fn test_accumulator_default_and_copy() {
        let acc = Accumulator::default();
        assert_eq!(acc, Accumulator::new());
        let copied = acc;
        assert_eq!(copied.state, [0i16; ACC_SIZE]);
        let accs = Accumulators::default();
        assert_eq!(accs.white.state, [0i16; ACC_SIZE]);
        assert_eq!(accs.black.state, [0i16; ACC_SIZE]);
    }

    #[test]
    fn test_accumulator_refresh_black_perspective() {
        let mut network = Network::new_boxed();
        network.transformer_biases.fill(5);
        network.transformer_weights.fill(1);

        let mut board = BoardState::default();
        board.refresh_accumulator(Side::Black, &network);

        let idx = board.history.index;
        assert_ne!(board.history.accumulators[idx].black.state[0], 5);
        assert!(board.history.computed[idx]);
    }
}
