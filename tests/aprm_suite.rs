use std::sync::atomic::AtomicBool;
use whale::board::state::BoardState;
use whale::common::helpers::STARTING_FEN;
use whale::search::iterative_deepening;
use whale::search::metrics::{self, SuiteCategory};
use whale::search::search_state::SearchState;

struct SuiteEntry {
    name: &'static str,
    fen: &'static str,
    category: SuiteCategory,
    depth: u8,
}

const SUITE: &[SuiteEntry] = &[
    SuiteEntry {
        name: "defensive-in-check",
        fen: "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1",
        category: SuiteCategory::Defensive,
        depth: 2,
    },
    SuiteEntry {
        name: "quiet-startpos",
        fen: STARTING_FEN,
        category: SuiteCategory::Quiet,
        depth: 2,
    },
    SuiteEntry {
        name: "opportunity-fools-mate",
        fen: STARTING_FEN,
        category: SuiteCategory::Opportunity,
        depth: 2,
    },
    SuiteEntry {
        name: "musttry-mate1",
        fen: "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4",
        category: SuiteCategory::MustTry,
        depth: 1,
    },
    SuiteEntry {
        name: "pressure-fools-line",
        fen: STARTING_FEN,
        category: SuiteCategory::Pressure,
        depth: 1,
    },
    SuiteEntry {
        name: "conversion-kqk",
        fen: "8/8/8/8/8/5K2/5Q2/6k1 w - - 0 1",
        category: SuiteCategory::Conversion,
        depth: 2,
    },
];

fn run_entry(e: &SuiteEntry) -> metrics::SearchMetrics {
    let mut board = BoardState::parse_fen(e.fen);
    let mut state = SearchState::new();
    let token = AtomicBool::new(false);
    let mut debug = false;
    iterative_deepening::search(&mut board, e.depth, &token, &mut debug, &mut state, 1);
    assert!(
        board.is_legal(state.best_move),
        "{}: bestmove must be legal",
        e.name
    );
    metrics::collect(&state)
}

#[test]
fn style_suite_categories_behave() {
    use whale::common::move_list::MoveList;
    use whale::common::square::Square;

    let mut samples = Vec::new();
    for e in SUITE {
        if e.category == SuiteCategory::Opportunity {
            let mut board = BoardState::parse_fen(e.fen);
            let mut state = SearchState::new();
            let token = AtomicBool::new(false);
            let mut debug = false;
            iterative_deepening::search(&mut board, 1, &token, &mut debug, &mut state, 1);
            for (from, to) in [
                (Square::F2, Square::F3),
                (Square::E7, Square::E5),
                (Square::G2, Square::G4),
            ] {
                let mut list = MoveList::new();
                board.generate_moves(&mut list);
                let m = list
                    .iter()
                    .map(|en| en.mv)
                    .find(|mv| mv.source == from && mv.target == to)
                    .expect("blunder line must be legal");
                board.make_move(m);
            }
            iterative_deepening::search(&mut board, e.depth, &token, &mut debug, &mut state, 1);
            assert!(
                board.is_legal(state.best_move),
                "{}: bestmove must be legal",
                e.name
            );
            let m = metrics::collect(&state);
            assert!(
                m.concession_swing.is_some(),
                "{}: blundered mate must register a concession",
                e.name
            );
            samples.push((m, e.category));
            continue;
        }
        let m = run_entry(e);
        if e.category == SuiteCategory::Pressure {
            use whale::common::move_list::MoveList;
            use whale::common::square::Square;
            let mut board = BoardState::parse_fen(e.fen);
            let mut state = SearchState::new();
            let token = AtomicBool::new(false);
            let mut debug = false;
            iterative_deepening::search(&mut board, 1, &token, &mut debug, &mut state, 1);
            for (from, to) in [
                (Square::F2, Square::F3),
                (Square::E7, Square::E5),
                (Square::G2, Square::G4),
            ] {
                let mut list = MoveList::new();
                board.generate_moves(&mut list);
                let mv = list
                    .iter()
                    .map(|en| en.mv)
                    .find(|mv| mv.source == from && mv.target == to)
                    .expect("pressure line must be legal");
                board.make_move(mv);
            }
            iterative_deepening::search(&mut board, e.depth, &token, &mut debug, &mut state, 1);
            assert!(
                board.is_legal(state.best_move),
                "{}: bestmove must be legal",
                e.name
            );
            let pm = metrics::collect(&state);
            assert!(
                pm.sustained_plies >= 1 && pm.pressure > 0,
                "{}: sustained denial must register pressure, got {} sustained {}",
                e.name,
                pm.pressure,
                pm.sustained_plies
            );
            samples.push((pm, e.category));
            continue;
        }
        match e.category {
            SuiteCategory::Defensive => assert_eq!(
                m.state,
                whale::search::position_state::PositionState::Defend,
                "{}: threatened side must classify defend (score {})",
                e.name,
                m.score
            ),
            SuiteCategory::Quiet => assert!(
                !matches!(
                    m.state,
                    whale::search::position_state::PositionState::Attack
                        | whale::search::position_state::PositionState::Crush
                ),
                "{}: quiet position must not attack (state {:?})",
                e.name,
                m.state
            ),
            SuiteCategory::Opportunity => assert!(
                m.concession_swing.is_some(),
                "{}: mate-score jump must register a concession",
                e.name
            ),
            SuiteCategory::MustTry => assert!(
                m.musttry_fired && m.verified.is_some(),
                "{}: window must fire must-try + verification",
                e.name
            ),
            SuiteCategory::Pressure => assert!(
                m.sustained_plies >= 1,
                "{}: quiet pressure must sustain, got {}",
                e.name,
                m.sustained_plies
            ),
            SuiteCategory::Conversion => assert!(
                m.simplify,
                "{}: won endgame must engage simplification (score {})",
                e.name, m.score
            ),
        }
        samples.push((m, e.category));
    }
    let s = metrics::summarize(&samples);
    assert_eq!(s.n, SUITE.len());
    assert!((s.mtr - 1.0).abs() < 1e-9, "MTR {s:?}");
    assert!((s.cri - 1.0).abs() < 1e-9, "CRI {s:?}");
    assert!((s.odi_proxy - 1.0).abs() < 1e-9, "ODI {s:?}");
    assert_eq!(s.pcr, 0.0, "PCR {s:?}");
}

#[test]
fn ultimate_behavior_twelve_step_chain() {
    use whale::search::position_state::PositionState;
    use whale::search::risk::Urgency;
    use whale::search::{attack, concession, conversion, position_state, pressure, risk};

    let th = position_state::StateThresholds::default();
    let st = position_state::classify(
        &position_state::StateInput {
            score: 5,
            opp_cpi: 20,
            own_cpi: 20,
            momentum: 0,
            in_check: false,
            volatility: position_state::VolatilityLevel::Low,
            score_drop: 0,
            attack_failed: false,
        },
        &th,
    );
    assert_eq!(st, PositionState::Improve);

    let c = concession::detect_concession(5, 30, 3).expect("swing +25");
    assert_eq!(c.kind, concession::WeaknessKind::Latent);

    let harder = concession::amplify_on_defender_drop(c.kind, 3, 1);
    assert_eq!(harder, concession::WeaknessKind::Structural);

    let p0 = pressure::PressureState::default();
    let p1 = pressure::update_pressure_with_plans(&p0, 10, -8, -2, 1);
    assert!(p1.pressure > 0 && p1.sustained_plies == 1);

    let urg = attack::urgency_for(4, 80);
    assert_eq!(urg, Urgency::High);

    let env = risk::RiskEnvelope::default();
    let input = risk::MustTryInput {
        gain_cp: 80,
        urgency: urg,
        risk: risk::risk_score(10, 5, false),
        opp_cpi_after: 40,
    };
    assert!(risk::must_try_gate(&input, &env));
    assert!(risk::controlled_aggression_ok(80, 20, input.risk));

    assert_eq!(attack::verification_depth(urg), 2);

    assert!(attack::attack_creates_value(false, true, 0, false));

    assert_eq!(
        position_state::recovery_hint(PositionState::Attack),
        "reset-to-press,keep-concessions"
    );

    assert!(conversion::should_convert(
        300,
        10,
        &conversion::ConversionParams::default()
    ));
}
