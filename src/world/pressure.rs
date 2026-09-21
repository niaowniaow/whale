#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PressureState {
    pub pressure: i32,

    pub sustained_plies: u32,
}

impl Default for PressureState {
    fn default() -> Self {
        Self {
            pressure: 0,
            sustained_plies: 0,
        }
    }
}

pub fn update_pressure(
    prev: &PressureState,
    momentum: i16,
    cpi_delta_opp: i32,
    freedom_delta_opp: i32,
) -> PressureState {
    update_pressure_with_plans(prev, momentum, cpi_delta_opp, freedom_delta_opp, 0)
}

pub fn update_pressure_with_plans(
    prev: &PressureState,
    momentum: i16,
    cpi_delta_opp: i32,
    freedom_delta_opp: i32,
    plans_denied_opp: i32,
) -> PressureState {
    let step = momentum as i32 / 2 - cpi_delta_opp - freedom_delta_opp * 2 + plans_denied_opp * 3;
    let pressure = prev.pressure + step.clamp(-60, 60);
    let sustained_plies = if step >= 0 {
        prev.sustained_plies + 1
    } else {
        0
    };
    PressureState {
        pressure,
        sustained_plies,
    }
}

pub fn pressure_efficiency_x100(cpi_reduction: i32, risk_increase: i32) -> i32 {
    if risk_increase <= 0 {
        return if cpi_reduction > 0 { 10_000 } else { 0 };
    }
    (cpi_reduction * 100 / risk_increase).clamp(-10_000, 10_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_accumulates_on_denial() {
        let s0 = PressureState::default();
        let s1 = update_pressure(&s0, 10, -8, -2);
        assert!(s1.pressure > 0);
        assert_eq!(s1.sustained_plies, 1);
        let s2 = update_pressure(&s1, 0, 30, 5);
        assert_eq!(s2.sustained_plies, 0);
    }

    #[test]
    fn plan_denial_adds_pressure() {
        let s0 = PressureState::default();
        let a = update_pressure(&s0, 0, 0, 0);
        let b = update_pressure_with_plans(&s0, 0, 0, 0, 2);
        assert!(b.pressure > a.pressure);

        assert_eq!(
            update_pressure_with_plans(&s0, 10, -8, -2, 0),
            update_pressure(&s0, 10, -8, -2)
        );
    }

    #[test]
    fn efficiency_math() {
        assert_eq!(pressure_efficiency_x100(10, 1), 1000);
        assert_eq!(pressure_efficiency_x100(0, 5), 0);
        assert_eq!(pressure_efficiency_x100(5, 0), 10_000);
    }
}
