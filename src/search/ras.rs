pub struct RuntimeAnnealer {
    cooling_lut: [u16; 64],
}

impl RuntimeAnnealer {
    pub fn new() -> Self {
        let mut cooling_lut = [0u16; 64];
        let mut t = 1.0f64;
        for cell in cooling_lut.iter_mut() {
            *cell = (t * 256.0).round() as u16;
            t *= 0.88;
        }
        Self { cooling_lut }
    }

    #[inline(always)]
    pub fn temperature_q8(&self, ply: usize) -> u16 {
        if ply < 64 { self.cooling_lut[ply] } else { 0 }
    }

    #[inline(always)]
    pub fn rfp_margin_adjustment(&self, ply: usize, depth: u8) -> i16 {
        if depth >= 4 {
            let t = self.temperature_q8(ply) as i32;
            ((t * 36) / 256) as i16
        } else {
            0
        }
    }

    #[inline(always)]
    pub fn lmr_perturbation(&self, ply: usize, depth: u8, hash: u64) -> i8 {
        if depth >= 5 && ply <= 6 {
            let t = self.temperature_q8(ply);
            if t > 100 {
                let pseudo_rand =
                    ((hash ^ ((ply as u64).wrapping_mul(0x9e3779b97f4a7c15))) >> 56) as u16;
                if pseudo_rand < (t / 4) {
                    return -1;
                }
            }
        }
        0
    }
}

impl Default for RuntimeAnnealer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cooling_monotonic() {
        let ras = RuntimeAnnealer::new();
        let mut prev = 300;
        for ply in 0..30 {
            let t = ras.temperature_q8(ply);
            assert!(t <= prev);
            prev = t;
        }
    }

    #[test]
    fn test_margin_adjustment_near_root() {
        let ras = RuntimeAnnealer::new();
        let adj_root = ras.rfp_margin_adjustment(0, 8);
        let adj_deep = ras.rfp_margin_adjustment(20, 8);
        assert!(adj_root > adj_deep);
    }
}
