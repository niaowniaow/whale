use std::sync::atomic::AtomicBool;
use whale::board::state::BoardState;
use whale::common::helpers::STARTING_FEN;
use whale::search::iterative_deepening;
use whale::search::perf::{PerfBudget, PerfSample, within_budget};
use whale::search::search_state::SearchState;

#[test]
fn root_behavior_overhead_within_budget() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let mut state = SearchState::new();
    let token = AtomicBool::new(false);
    let mut debug = false;
    let start = std::time::Instant::now();
    iterative_deepening::search(&mut board, 5, &token, &mut debug, &mut state, 1);
    let elapsed = start.elapsed();
    assert!(board.is_legal(state.best_move));
    assert!(state.nodes > 0);
    let sample = PerfSample {
        nodes: state.nodes,
        tbhits: state.tbhits,
        elapsed_ms: elapsed.as_millis().max(1),
        behavior_us: state.behavior_us,
        depth: 5,
    };
    assert!(sample.nps() > 0);
    assert!(sample.tt_hit_pct() >= 0.0 && sample.tt_hit_pct() <= 100.0);
    assert!(within_budget(&sample, &PerfBudget::default()));
}

#[test]
fn aprm_off_is_not_slower_than_budget_ceiling() {
    let token = AtomicBool::new(false);
    let mut debug = false;
    let mut on_nodes = 0u64;
    let mut off_nodes = 0u64;
    for enabled in [true, false] {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let mut state = SearchState::new();
        state.params.cpi_enabled = enabled;
        state.params.state_enabled = enabled;
        state.params.risk_enabled = enabled;
        state.params.pressure_enabled = enabled;
        state.params.attack_enabled = enabled;
        state.params.conversion_enabled = enabled;
        iterative_deepening::search(&mut board, 2, &token, &mut debug, &mut state, 1);
        assert!(board.is_legal(state.best_move));
        if enabled {
            on_nodes = state.nodes;
        } else {
            off_nodes = state.nodes;
        }
    }
    assert!(on_nodes > 0 && off_nodes > 0);
}
