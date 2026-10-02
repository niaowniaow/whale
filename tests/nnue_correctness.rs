use whale::board::state::BoardState;
use whale::common::helpers::{ADVANCED_MOVE_FEN, ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN};
use whale::common::move_list::MoveList;
use whale::eval::nnue::v16::{SfnnPosition, ensure_sfnn16_fresh, evaluate_board, evaluate_board_detailed};

fn check_fen(fen: &str) {
    let mut board = BoardState::parse_fen(fen);
    let first = evaluate_board(&mut board);
    let second = evaluate_board(&mut board);
    assert_eq!(first, second);
    if let Some(v) = first {
        assert!(v.abs() <= 29000);
    }
    if let Some(d) = evaluate_board_detailed(&mut board) {
        assert!(d.combined.abs() <= 29000);
        if let Some(v) = first {
            assert_eq!(d.combined.clamp(-29000, 29000) as i16, v);
        }
    }
    let pos = SfnnPosition::from_board(&board);
    ensure_sfnn16_fresh(&mut board, &pos);
    let again = evaluate_board(&mut board);
    assert_eq!(first, again);
}

#[test]
fn nnue_fen_consistency() {
    let fens = [
        STARTING_FEN,
        KIWI_PETE_FEN,
        ENDGAME_FEN,
        ADVANCED_MOVE_FEN,
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1",
        "8/5P1k/8/8/8/8/6K1/8 w - - 0 1",
        "8/8/8/8/8/1k6/8/K7 w - - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1",
    ];
    for fen in fens {
        check_fen(fen);
    }
}

#[test]
fn nnue_make_unmake_stable() {
    let fens = [
        STARTING_FEN,
        KIWI_PETE_FEN,
        "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1",
        "8/5P1k/8/8/8/8/6K1/8 w - - 0 1",
    ];
    for fen in fens {
        let mut board = BoardState::parse_fen(fen);
        let base = evaluate_board(&mut board);
        let mut ml = MoveList::new();
        board.generate_moves(&mut ml);
        let mut count = 0;
        for e in ml.iter() {
            if count >= 12 {
                break;
            }
            count += 1;
            board.make_move(e.mv);
            let after = evaluate_board(&mut board);
            if let Some(v) = after {
                assert!(v.abs() <= 29000);
            }
            let rep = evaluate_board(&mut board);
            assert_eq!(after, rep);
            board.unmake_move(e.mv);
            let back = evaluate_board(&mut board);
            assert_eq!(base, back);
        }
        assert!(count > 0);
    }
}

#[test]
fn nnue_both_sides_deterministic() {
    let pair = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
    ];
    for fen in pair {
        let mut board = BoardState::parse_fen(fen);
        let a = evaluate_board(&mut board);
        let b = evaluate_board(&mut board);
        assert_eq!(a, b);
    }
}

fn next_rand(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

#[test]
fn nnue_random_walk_stable() {
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let mut rng: u64 = 0x123456789abcdef;
    let mut steps = 0;
    for _ in 0..512 {
        let before = evaluate_board(&mut board);
        let again = evaluate_board(&mut board);
        assert_eq!(before, again);
        let mut ml = MoveList::new();
        board.generate_moves(&mut ml);
        if ml.is_empty() {
            break;
        }
        let idx = (next_rand(&mut rng) as usize) % ml.len();
        let mv = ml[idx].mv;
        board.make_move(mv);
        let after = evaluate_board(&mut board);
        let rep = evaluate_board(&mut board);
        assert_eq!(after, rep);
        board.unmake_move(mv);
        let back = evaluate_board(&mut board);
        assert_eq!(before, back);
        board.make_move(mv);
        steps += 1;
        if board.history.index >= 240 {
            break;
        }
    }
    assert!(steps > 50);
}
