#[derive(Debug, Clone, Copy)]
pub struct ConversionParams {
    pub convert_min_cp: i16,

    pub crush_min_cp: i16,

    pub max_opp_cpi: i32,
}

impl Default for ConversionParams {
    fn default() -> Self {
        Self {
            convert_min_cp: 250,
            crush_min_cp: 600,
            max_opp_cpi: 40,
        }
    }
}

pub fn should_convert(score: i16, opp_cpi: i32, params: &ConversionParams) -> bool {
    if score >= params.crush_min_cp {
        return true;
    }
    score >= params.convert_min_cp && opp_cpi <= params.max_opp_cpi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_gates() {
        let p = ConversionParams::default();
        assert!(should_convert(700, 100, &p));
        assert!(should_convert(300, 10, &p));
        assert!(!should_convert(300, 100, &p));
        assert!(!should_convert(70, 0, &p));
    }
}
