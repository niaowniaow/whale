use whale::board::state::BoardState;
use whale::common::helpers::STARTING_FEN;
use whale::common::move_list::MoveList;
use whale::common::move_type::MoveType;
use whale::common::moves::Move;
use whale::common::square::Square;
use whale::common::tt::{TranspositionEntryType, TranspositionTable};

fn is_legal(board: &mut BoardState, m: Move) -> bool {
    let mut ml = MoveList::new();
    board.generate_moves(&mut ml);
    for e in ml.iter() {
        if e.mv.source == m.source && e.mv.target == m.target && e.mv.move_type == m.move_type {
            return true;
        }
    }
    false
}

#[test]
fn tt_best_move_is_legal() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let tt = TranspositionTable::new(1024);
    let m = Move::new(Square::E2, Square::E4, MoveType::DoublePush);
    assert!(is_legal(&mut board, m));
    tt.submit_entry(999001, 50, 5, m, TranspositionEntryType::Exact);
    let got = tt.probe(999001).expect("entry present").best_move;
    assert!(is_legal(&mut board, got));
    let (_, _, mv) = tt.get_entry(999001, -1000, 1000, 5, 0, 0);
    assert_eq!(mv, Some(m));
}

#[test]
fn tt_shallow_entry_returns_move_without_cutoff() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let tt = TranspositionTable::new(1024);
    let m = Move::new(Square::G1, Square::F3, MoveType::Quiet);
    assert!(is_legal(&mut board, m));
    tt.submit_entry(999002, 20, 5, m, TranspositionEntryType::Exact);
    let (found, _, mv) = tt.get_entry(999002, -1000, 1000, 6, 0, 0);
    assert!(!found);
    let got = mv.expect("move present");
    assert!(is_legal(&mut board, got));
}
