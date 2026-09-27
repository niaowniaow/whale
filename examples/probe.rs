use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use whale::board::state::BoardState;
use whale::common::helpers::BENCH_FENS;
use whale::common::tt::TranspositionTable;
use whale::search::search_state::SearchState;

type Setup = Box<dyn Fn(&mut SearchState)>;

fn run_case(depth: u8, setup: &dyn Fn(&mut SearchState)) -> (u128, u64) {
    let cancel = AtomicBool::new(false);
    let mut debug = false;
    let mut total_ms = 0u128;
    let mut total_nodes = 0u64;
    for fen in BENCH_FENS.iter().take(3) {
        let mut state = SearchState::new();
        state.tt = Arc::new(TranspositionTable::new_mb(16));
        setup(&mut state);
        let mut board = BoardState::parse_fen(fen);
        let start = Instant::now();
        board.find_best_move(depth, &cancel, &mut debug, &mut state, 1);
        total_ms += start.elapsed().as_millis();
        total_nodes += state.nodes;
    }
    (total_ms, total_nodes)
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let (depth, want): (u8, Vec<String>) = match raw.first().and_then(|s| s.parse::<u8>().ok()) {
        Some(d) => (d, raw[1..].to_vec()),
        None => (10, raw),
    };
    let cases: Vec<(&str, Setup)> = vec![
        ("base", Box::new(|_: &mut SearchState| {})),
        (
            "no-cpi",
            Box::new(|s: &mut SearchState| s.params.cpi_enabled = false),
        ),
        (
            "no-state",
            Box::new(|s: &mut SearchState| s.params.state_enabled = false),
        ),
        (
            "no-risk",
            Box::new(|s: &mut SearchState| s.params.risk_enabled = false),
        ),
        (
            "no-pressure",
            Box::new(|s: &mut SearchState| s.params.pressure_enabled = false),
        ),
        (
            "no-attack",
            Box::new(|s: &mut SearchState| s.params.attack_enabled = false),
        ),
        (
            "no-conversion",
            Box::new(|s: &mut SearchState| s.params.conversion_enabled = false),
        ),
        (
            "no-learned",
            Box::new(|s: &mut SearchState| s.params.learned_enabled = false),
        ),
        (
            "atk-score-150",
            Box::new(|s: &mut SearchState| s.state_thresholds.attack_score = 150),
        ),
        (
            "atk-score-200",
            Box::new(|s: &mut SearchState| s.state_thresholds.attack_score = 200),
        ),
        (
            "atk-cpi-120",
            Box::new(|s: &mut SearchState| s.state_thresholds.attack_cpi = 120),
        ),
        (
            "urg-min-50",
            Box::new(|s: &mut SearchState| s.urgency_thresholds.min_gain = 50),
        ),
        (
            "urg-musttry-100",
            Box::new(|s: &mut SearchState| s.urgency_thresholds.musttry_gain = 100),
        ),
    ];
    for (name, setup) in &cases {
        if !want.is_empty() && !want.iter().any(|w| w == name) {
            continue;
        }
        let mut best_ms = u128::MAX;
        let mut best_nodes = 0u64;
        for _ in 0..2 {
            let (ms, nodes) = run_case(depth, &**setup);
            if ms < best_ms {
                best_ms = ms;
                best_nodes = nodes;
            }
        }
        let nps = if best_ms > 0 {
            (best_nodes as f64 / best_ms as f64 * 1000.0) as u64
        } else {
            0
        };
        println!("{name}: {best_ms} ms {best_nodes} nodes {nps} nps");
    }
}
