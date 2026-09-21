use std::sync::atomic::AtomicBool;
use whale::board::state::BoardState;
use whale::common::constants::MAX_CENTIPAWN_EVAL;
use whale::common::helpers::STARTING_FEN;
use whale::common::move_list::MoveList;
use whale::common::square::Square;
use whale::search::iterative_deepening;
use whale::search::metrics::{self, SuiteCategory};
use whale::search::search_state::SearchState;
use whale::world::counterplay::count_freedom;
use whale::world::intent::SearchIntent;
use whale::world::position_state::PositionState;

fn go(board: &mut BoardState, state: &mut SearchState, depth: u8) -> metrics::SearchMetrics {
    let token = AtomicBool::new(false);
    let mut debug = false;
    iterative_deepening::search(board, depth, &token, &mut debug, state, 1);
    assert!(board.is_legal(state.best_move));
    metrics::collect(state)
}

fn play(board: &mut BoardState, from: Square, to: Square) {
    let mut list = MoveList::new();
    board.generate_moves(&mut list);
    let mv = list
        .iter()
        .map(|en| en.mv)
        .find(|m| m.source == from && m.target == to)
        .expect("scripted line must stay legal");
    assert!(board.is_legal(mv));
    board.make_move(mv);
}

fn lan_of(state: &SearchState) -> String {
    let m = state.best_move;
    let promo = m
        .promotion_char()
        .map(|c| c.to_string())
        .unwrap_or_default();
    format!("{}{}{}", m.source, m.target, promo)
}

#[test]
fn ultimate_twenty_step_full_lifecycle() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let mut main = SearchState::new();

    let m0 = go(&mut board, &mut main, 1);
    assert!(
        board.is_legal(main.best_move),
        "1/20 stable position yields a legal move"
    );
    assert!(
        matches!(
            m0.state,
            PositionState::Improve | PositionState::Stabilize | PositionState::Press
        ),
        "2/20 engine stays calm at start, got {:?}",
        m0.state
    );

    play(&mut board, Square::F2, Square::F3);
    let mut aux = SearchState::new();
    let m1 = go(&mut board, &mut aux, 1);
    assert!(
        !matches!(m1.state, PositionState::Attack | PositionState::Crush),
        "3/20 no premature attack after 1.f3, got {:?}",
        m1.state
    );
    assert!(!m1.musttry_fired, "4/20 no false must-try in a quiet line");

    play(&mut board, Square::E7, Square::E5);
    let mut aux2 = SearchState::new();
    let m2 = go(&mut board, &mut aux2, 1);
    assert!(
        board.is_legal(aux2.best_move),
        "5/20 engine answers 1...e5 legally"
    );
    assert!(
        m2.concession_swing.is_none() || m2.score.abs() < 200,
        "6/20 no phantom concession in a normal opening"
    );

    play(&mut board, Square::G2, Square::G4);
    let mm = go(&mut board, &mut main, 1);
    assert_eq!(lan_of(&main), "d8h4", "7/20 mate window is found");
    assert!(
        mm.score.abs() >= MAX_CENTIPAWN_EVAL - 64,
        "8/20 mate score is decisive, got {}",
        mm.score
    );
    assert!(
        mm.concession_swing.is_some(),
        "9/20 blunder registers as a concession"
    );
    assert!(mm.musttry_fired, "10/20 urgent window triggers must-try");
    assert!(mm.verified.is_some(), "11/20 attack is verified by search");
    assert!(
        matches!(mm.state, PositionState::Crush | PositionState::Attack),
        "12/20 state commits to the attack, got {:?}",
        mm.state
    );
    assert_eq!(
        mm.intent,
        SearchIntent::Verification,
        "13/20 intent routes to verification"
    );

    board.make_move(main.best_move);
    assert_eq!(
        count_freedom(&board),
        0,
        "14/20 window closes with no escape"
    );

    let mut end = BoardState::parse_fen("4k3/8/8/8/8/8/5Q2/4K3 w - - 0 1");
    let mut conv = SearchState::new();
    let mk = go(&mut end, &mut conv, 1);
    assert!(
        end.is_legal(conv.best_move),
        "15/20 converter finds a legal move"
    );
    assert!(mk.simplify, "16/20 won endgame engages simplification");
    assert!(
        matches!(mk.state, PositionState::Convert | PositionState::Crush),
        "17/20 state reaches conversion, got {:?}",
        mk.state
    );
    assert_eq!(
        mk.intent,
        SearchIntent::Conversion,
        "18/20 intent routes to conversion"
    );
    assert!(!mk.reset, "19/20 stable conversion needs no reset");

    let opp_sample = metrics::collect(&main);
    let samples = vec![
        (m0, SuiteCategory::Quiet),
        (mm, SuiteCategory::MustTry),
        (opp_sample, SuiteCategory::Opportunity),
        (mk, SuiteCategory::Conversion),
    ];
    let s = metrics::summarize(&samples);
    assert_eq!(s.n, 4, "20/20 lifecycle summary covers all chapters");
    assert!(
        (s.mtr - 1.0).abs() < 1e-9,
        "20/20 must-try recognized {s:?}"
    );
    assert!(
        (s.cri - 1.0).abs() < 1e-9,
        "20/20 conversion reliable {s:?}"
    );
    assert!(
        (s.odi_proxy - 1.0).abs() < 1e-9,
        "20/20 concession detected {s:?}"
    );
    assert_eq!(s.pcr, 0.0, "20/20 no premature attack {s:?}");
}
