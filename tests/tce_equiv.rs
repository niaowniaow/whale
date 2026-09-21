use whale::board::node_threats::NodeThreats;
use whale::board::state::BoardState;
use whale::common::helpers::{ADVANCED_MOVE_FEN, ENDGAME_FEN, KIWI_PETE_FEN, STARTING_FEN};
use whale::common::move_list::MoveList;
use whale::common::moves::Move;
use whale::common::piece::Piece;
use whale::search::tce::{ThreatFeatures, extract_threat_features};

fn fresh_features(
    board: &BoardState,
    move_obj: Move,
    depth: u8,
    in_check: bool,
    previous_move: Option<Move>,
) -> ThreatFeatures {
    let us = board.side_to_move;
    let them = us.other();
    let num_pins = board.pinned_pieces(us).0.count_ones() as i32
        + board.pinned_pieces(them).0.count_ones() as i32;
    let threats = board.threat_by_lesser(us);
    let mut num_threatened_pieces = 0i32;
    for (piece_idx, &threat) in threats.iter().enumerate().take(5).skip(1) {
        let piece = match piece_idx {
            1 => Piece::Knight,
            2 => Piece::Bishop,
            3 => Piece::Rook,
            4 => Piece::Queen,
            _ => Piece::None,
        };
        let pieces_bb = board.get_pieces(us, piece).0;
        num_threatened_pieces += (pieces_bb & threat).count_ones() as i32;
    }
    let num_checkers = if in_check {
        board.checkers(us).0.count_ones() as i32
    } else {
        0
    };

    let _ = (depth, move_obj, previous_move);
    ThreatFeatures {
        num_pins,
        num_threatened_pieces,
        is_check: in_check,
        num_checkers,
        is_capture: move_obj.is_capture(),
        material_imbalance: 0,
        king_danger: 0,
        depth_remaining: depth as i32,
        is_recapture: false,
        is_pawn_advance: false,
    }
}

fn pseudo_random(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[test]
fn tce_features_match_fresh_on_random_walk() {
    let mut seed = 0x12345678u64;
    let mut board = BoardState::parse_fen(STARTING_FEN);
    let extra = [
        KIWI_PETE_FEN,
        ENDGAME_FEN,
        ADVANCED_MOVE_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "7k/8/2b2b2/3pp3/8/8/8/K2RQ3 w - - 0 1",
    ];
    let mut positions = vec![board.clone()];
    for fen in extra {
        positions.push(BoardState::parse_fen(fen));
    }

    for _ in 0..300 {
        let mut ml = MoveList::new();
        board.generate_moves(&mut ml);
        if ml.is_empty() {
            board = BoardState::parse_fen(STARTING_FEN);
            continue;
        }
        let idx = (pseudo_random(&mut seed) as usize) % ml.len();
        let m = ml[idx].mv;
        if !board.is_legal(m) {
            continue;
        }
        board.make_move(m);
        positions.push(board.clone());
        if positions.len() > 400 {
            break;
        }
    }
    assert!(positions.len() > 100);
    let mut checked = 0;
    for b in &positions {
        let nt = NodeThreats::compute(b);
        let mut ml = MoveList::new();
        b.generate_moves(&mut ml);
        for e in ml.iter().take(8) {
            let m = e.mv;
            for depth in [2u8, 6, 9] {
                let in_check = b.is_in_check(b.side_to_move);
                let fresh = fresh_features(b, m, depth, in_check, None);
                let got = extract_threat_features(b, &nt, m, depth, in_check, None);
                assert_eq!(fresh.num_pins, got.num_pins, "pins {m:?}");
                assert_eq!(
                    fresh.num_threatened_pieces, got.num_threatened_pieces,
                    "threats {m:?}"
                );
                assert_eq!(fresh.num_checkers, got.num_checkers, "checkers {m:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 1000);
}
