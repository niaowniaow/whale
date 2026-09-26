use super::BoardState;
use crate::common::castle::Castle;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;
#[test]
fn add_piece_sets_all_structures() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);

    let sq = Square::E4 as usize;

    assert_eq!(board.get_pieces(Side::White, Piece::Pawn).get_bit(sq), 1);
    assert_eq!(board.occupancies[Side::White].get_bit(sq), 1);
    assert_eq!(board.occupancy().get_bit(sq), 1);
    assert_eq!(board.occupancies[Side::Black].get_bit(sq), 0);
    assert_eq!(board.piece_mapping[sq], Piece::Pawn);
    assert_eq!(board.phase, 0);
}

#[test]
fn remove_piece_clears_all_structures() {
    let mut board = BoardState::new();
    board.add_piece(Square::D5, Side::Black, Piece::Queen, false);

    let sq = Square::D5 as usize;
    let removed = board.remove_piece(Square::D5, false);

    assert_eq!(removed, Piece::Queen);
    assert_eq!(board.get_pieces(Side::Black, Piece::Queen).get_bit(sq), 0);
    assert_eq!(board.occupancies[Side::Black].get_bit(sq), 0);
    assert_eq!(board.occupancy().get_bit(sq), 0);
    assert_eq!(board.piece_mapping[sq], Piece::None);
}

#[test]
fn get_piece_on_side_returns_correct_piece() {
    let mut board = BoardState::new();
    board.add_piece(Square::C3, Side::White, Piece::Knight, false);

    assert_eq!(
        board.get_piece_on_side(Square::C3, Side::White),
        Piece::Knight as usize
    );
    assert_eq!(
        board.get_piece_on_side(Square::C3, Side::Black),
        Piece::None as usize
    );
    assert_eq!(
        board.get_piece_on_side(Square::D4, Side::White),
        Piece::None as usize
    );
}

#[test]
fn get_piece_on_returns_signed_index() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E8, Side::Black, Piece::King, false);

    assert_eq!(board.get_piece_on(Square::E1), 5);
    assert_eq!(board.get_piece_on(Square::E8), 11);
    assert_eq!(board.get_piece_on(Square::D4), -1);
}

#[test]
fn equality_works_for_blank_boards() {
    let b1 = BoardState::new();
    let b2 = BoardState::new();
    assert_eq!(b1, b2);
}

#[test]
fn equality_detects_difference() {
    let mut b1 = BoardState::new();
    let b2 = BoardState::new();
    b1.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    assert_ne!(b1, b2);
}

#[test]
fn is_in_check_not_in_check_on_empty_board() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    assert!(!board.is_in_check(Side::White));
}

#[test]
fn is_in_check_handles_missing_king_without_panicking() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    assert!(!board.is_in_check(Side::Black));
}

#[test]
fn is_in_check_detects_rook_check() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
    assert!(board.is_in_check(Side::White));
}

#[test]
fn is_in_check_detects_knight_check() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::King, false);

    board.add_piece(Square::D6, Side::Black, Piece::Knight, false);
    assert!(board.is_in_check(Side::White));
}

#[test]
fn is_in_check_detects_bishop_check() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::King, false);
    board.add_piece(Square::H7, Side::Black, Piece::Bishop, false);
    assert!(board.is_in_check(Side::White));
}

#[test]
fn is_in_check_detects_queen_check() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::King, false);
    board.add_piece(Square::E8, Side::Black, Piece::Queen, false);
    assert!(board.is_in_check(Side::White));
}

#[test]
fn is_in_check_detects_pawn_check() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::King, false);
    board.add_piece(Square::D5, Side::Black, Piece::Pawn, false);
    assert!(board.is_in_check(Side::White));
}

#[test]
fn blocker_prevents_rook_check() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
    assert!(!board.is_in_check(Side::White));
}

#[test]
fn test_accumulator_make_unmake_consistency() {
    use crate::common::move_list::MoveList;
    use crate::eval::nnue::loader::Network;

    let network = Network::get_embedded();

    let mut board = BoardState::starting_position();

    let expected_white = board.history.accumulators[board.history.index].white;
    let expected_black = board.history.accumulators[board.history.index].black;
    board.refresh_accumulator(Side::White, network);
    board.refresh_accumulator(Side::Black, network);
    assert_eq!(
        board.history.accumulators[board.history.index].white,
        expected_white
    );
    assert_eq!(
        board.history.accumulators[board.history.index].black,
        expected_black
    );

    let mut move_history = Vec::new();
    for _ in 0..10 {
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);
        if moves.count == 0 {
            break;
        }

        let mut legal_move = None;
        for m_entry in moves.iter() {
            let m = m_entry.mv;
            board.make_move(m);
            let is_legal = !board.is_in_check(board.side_to_move.other());
            board.unmake_move(m);
            if is_legal {
                legal_move = Some(m);
                break;
            }
        }

        let Some(m) = legal_move else {
            break;
        };

        board.make_move(m);
        move_history.push(m);

        board.ensure_accumulators_fresh();
        let current_white = board.history.accumulators[board.history.index].white;
        let current_black = board.history.accumulators[board.history.index].black;

        board.refresh_accumulator(Side::White, network);
        board.refresh_accumulator(Side::Black, network);

        assert_eq!(
            board.history.accumulators[board.history.index].white, current_white,
            "Failed white accumulator check after move {:?}",
            m
        );
        assert_eq!(
            board.history.accumulators[board.history.index].black, current_black,
            "Failed black accumulator check after move {:?}",
            m
        );
    }

    while let Some(m) = move_history.pop() {
        board.unmake_move(m);

        board.ensure_accumulators_fresh();
        let current_white = board.history.accumulators[board.history.index].white;
        let current_black = board.history.accumulators[board.history.index].black;

        board.refresh_accumulator(Side::White, network);
        board.refresh_accumulator(Side::Black, network);

        assert_eq!(
            board.history.accumulators[board.history.index].white, current_white,
            "Failed white accumulator check after unmake {:?}",
            m
        );
        assert_eq!(
            board.history.accumulators[board.history.index].black, current_black,
            "Failed black accumulator check after unmake {:?}",
            m
        );
    }

    assert_eq!(
        board.history.accumulators[board.history.index].white,
        expected_white
    );
    assert_eq!(
        board.history.accumulators[board.history.index].black,
        expected_black
    );
}

#[test]
fn test_is_legal_matches_make_unmake() {
    use crate::common::move_list::MoveList;

    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/8/8/8/k2Pp2R/8/8/3K4 b - d3 0 1",
    ];

    for fen in fens {
        let mut board = BoardState::parse_fen(fen);
        let mut moves = MoveList::new();
        board.generate_moves(&mut moves);

        let stm = board.side_to_move;
        let node_checkers = board.checkers(stm).0;
        let node_pinned = board.pinned_pieces(stm).0;
        assert_eq!(board.is_in_check(stm), node_checkers != 0);

        for m_entry in moves.iter() {
            let m = m_entry.mv;
            let fast_legal = board.is_legal(m);
            let cached_legal = board.is_legal_with(m, node_checkers, node_pinned);

            board.make_move(m);
            let slow_legal = !board.is_in_check(board.side_to_move.other());
            board.unmake_move(m);

            assert_eq!(
                fast_legal, slow_legal,
                "Legality mismatch for move {:?} in FEN: {}",
                m, fen
            );
            assert_eq!(
                cached_legal, slow_legal,
                "Cached-legality mismatch for move {:?} in FEN: {}",
                m, fen
            );
        }
    }
}

#[test]
fn has_non_pawn_material_detects_major_pieces() {
    use crate::common::helpers::STARTING_FEN;

    let board = BoardState::parse_fen(STARTING_FEN);
    assert!(board.has_non_pawn_material(Side::White));
    assert!(board.has_non_pawn_material(Side::Black));

    let mut bare = BoardState::new();
    bare.add_piece(Square::E1, Side::White, Piece::King, false);
    bare.add_piece(Square::E8, Side::Black, Piece::King, false);
    assert!(!bare.has_non_pawn_material(Side::White));
    assert!(!bare.has_non_pawn_material(Side::Black));

    bare.add_piece(Square::E2, Side::White, Piece::Pawn, false);
    assert!(!bare.has_non_pawn_material(Side::White));

    bare.add_piece(Square::D1, Side::White, Piece::Queen, false);
    assert!(bare.has_non_pawn_material(Side::White));
}

#[test]
fn passed_pawn_detection() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    assert!(board.is_passed_pawn(Square::E4 as usize, Side::White));
    assert_eq!(
        board
            .get_passed_pawns(Side::White)
            .get_bit(Square::E4 as usize),
        1
    );

    board.add_piece(Square::D5, Side::Black, Piece::Pawn, false);
    assert!(!board.is_passed_pawn(Square::E4 as usize, Side::White));
    assert!(board.get_passed_pawns(Side::White).is_empty());
}

#[test]
fn is_aligned_covers_ranks_files_diagonals() {
    assert!(BoardState::is_aligned(Square::A1, Square::H1));
    assert!(BoardState::is_aligned(Square::E1, Square::E8));
    assert!(BoardState::is_aligned(Square::A1, Square::H8));
    assert!(BoardState::is_aligned(Square::E4, Square::E4));
    assert!(!BoardState::is_aligned(Square::A1, Square::H7));
    assert!(!BoardState::is_aligned(Square::E2, Square::F4));
}

#[test]
fn square_attack_queries() {
    use crate::common::helpers::STARTING_FEN;

    let board = BoardState::parse_fen(STARTING_FEN);

    assert!(board.is_square_attacked(Square::E3, Side::White));
    assert!(board.is_square_attacked(Square::C3, Side::White));
    assert!(!board.is_square_attacked(Square::E4, Side::White));
    assert!(!board.is_square_attacked(Square::E4, Side::Black));

    let occ = board.occupancy();
    assert!(board.is_square_attacked_with_occ(Square::E3, Side::White, occ));
    assert!(!board.is_square_attacked_with_occ(Square::E4, Side::White, occ));

    let mut custom = BoardState::new();
    custom.add_piece(Square::E1, Side::White, Piece::King, false);
    custom.add_piece(Square::D3, Side::Black, Piece::Knight, false);
    custom.add_piece(Square::D2, Side::Black, Piece::King, false);
    let custom_occ = custom.occupancy();
    assert!(custom.is_square_attacked_with_occ(Square::E1, Side::Black, custom_occ));

    let mut slider = BoardState::new();
    slider.add_piece(Square::E1, Side::White, Piece::King, false);
    slider.add_piece(Square::H4, Side::Black, Piece::Bishop, false);
    slider.add_piece(Square::E8, Side::Black, Piece::Rook, false);
    let slider_occ = slider.occupancy();
    assert!(slider.is_square_attacked_with_occ(Square::E1, Side::Black, slider_occ));
}

#[test]
fn pinned_pieces_and_checkers() {
    use crate::common::helpers::STARTING_FEN;

    let start = BoardState::parse_fen(STARTING_FEN);
    assert!(start.pinned_pieces(Side::White).is_empty());
    assert!(start.checkers(Side::White).is_empty());
    assert!(BoardState::new().pinned_pieces(Side::White).is_empty());
    assert!(BoardState::new().checkers(Side::White).is_empty());

    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E2, Side::White, Piece::Queen, false);
    board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
    assert_eq!(
        board.pinned_pieces(Side::White).0,
        1u64 << Square::E2 as usize
    );
    assert!(board.pinned_pieces(Side::Black).is_empty());

    let mut check = BoardState::new();
    check.add_piece(Square::E1, Side::White, Piece::King, false);
    check.add_piece(Square::E8, Side::Black, Piece::Rook, false);
    assert!(check.is_in_check(Side::White));
    assert_eq!(check.checkers(Side::White).0, 1u64 << Square::E8 as usize);
}

#[test]
fn check_squares_and_threat_by_lesser() {
    use crate::common::helpers::STARTING_FEN;

    assert_eq!(BoardState::new().check_squares(Side::White), [0; 6]);

    let start = BoardState::parse_fen(STARTING_FEN);
    let squares = start.check_squares(Side::White);
    assert_ne!(squares[Piece::Knight as usize], 0);
    assert_eq!(squares[Piece::King as usize], 0);
    assert_eq!(
        squares[Piece::Queen as usize],
        squares[Piece::Bishop as usize] | squares[Piece::Rook as usize]
    );

    let threats = start.threat_by_lesser(Side::White);
    assert_eq!(threats[Piece::Pawn as usize], 0);
    assert_eq!(threats[Piece::King as usize], 0);
    assert_ne!(threats[Piece::Knight as usize], 0);
    assert_eq!(
        threats[Piece::Queen as usize] & threats[Piece::Rook as usize],
        threats[Piece::Rook as usize]
    );
    assert_eq!(
        threats[Piece::Rook as usize] & threats[Piece::Knight as usize],
        threats[Piece::Knight as usize]
    );
}

#[test]
fn clipped_phase_tracks_material() {
    use crate::common::helpers::STARTING_FEN;

    assert_eq!(BoardState::new().clipped_phase(), 0);
    let start = BoardState::parse_fen(STARTING_FEN);
    assert!(start.clipped_phase() > 0);
}

#[test]
fn pseudo_legal_accepts_generated_moves_and_rejects_obvious_illegal() {
    use crate::common::helpers::STARTING_FEN;
    use crate::common::move_list::MoveList;
    use crate::common::move_type::MoveType;

    let board = BoardState::parse_fen(STARTING_FEN);
    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);
    assert!(!moves.is_empty());
    for entry in moves.iter() {
        assert!(board.is_pseudo_legal(entry.mv), "{:?}", entry.mv);
    }

    assert!(!board.is_pseudo_legal(Move::NO_MOVE));

    assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E2, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::A7, Square::A6, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D2, MoveType::Capture)));

    assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::Castle)));

    assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::B3, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Capture)));

    assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::Quiet)));

    assert!(board.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::DoublePush)));

    assert!(board.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::QueenPromotion)));
}

#[test]
fn pseudo_legal_rejects_bad_squares() {
    use crate::common::helpers::STARTING_FEN;

    let board = BoardState::parse_fen(STARTING_FEN);
    assert!(!board.is_pseudo_legal(Move::NO_MOVE));
    assert!(!board.is_pseudo_legal(Move::new(Square::NoSquare, Square::E4, MoveType::Quiet)));
    assert!(!board.is_pseudo_legal(Move::new(Square::E2, Square::NoSquare, MoveType::Quiet)));
}

#[test]
fn pawn_pseudo_legal_push_and_double_push_branches() {
    use crate::common::helpers::STARTING_FEN;

    let start = BoardState::parse_fen(STARTING_FEN);
    assert!(start.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

    assert!(!start.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Capture)));

    let mut blocked = BoardState::new();
    blocked.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    blocked.add_piece(Square::E1, Side::White, Piece::King, false);
    blocked.add_piece(Square::E8, Side::Black, Piece::King, false);
    blocked.add_piece(Square::E3, Side::White, Piece::Pawn, false);
    blocked.add_piece(Square::E2, Side::White, Piece::Pawn, false);
    assert!(!blocked.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));
    assert!(!blocked.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::DoublePush)));

    let mut occupied = BoardState::new();
    occupied.add_piece(Square::E2, Side::White, Piece::Pawn, false);
    occupied.add_piece(Square::E3, Side::Black, Piece::Pawn, false);
    assert!(!occupied.is_pseudo_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

    let mut target_busy = BoardState::new();
    target_busy.add_piece(Square::E2, Side::White, Piece::Pawn, false);
    target_busy.add_piece(Square::E4, Side::Black, Piece::Knight, false);
    assert!(!target_busy.is_pseudo_legal(Move::new(Square::E2, Square::E4, MoveType::DoublePush)));

    let mut mistimed = BoardState::new();
    mistimed.add_piece(Square::E3, Side::White, Piece::Pawn, false);
    assert!(!mistimed.is_pseudo_legal(Move::new(Square::E3, Square::E5, MoveType::DoublePush)));

    let mut black = BoardState::new();
    black.add_piece(Square::E7, Side::Black, Piece::Pawn, false);
    black.side_to_move = Side::Black;
    assert!(black.is_pseudo_legal(Move::new(Square::E7, Square::E6, MoveType::Quiet)));
    assert!(black.is_pseudo_legal(Move::new(Square::E7, Square::E5, MoveType::DoublePush)));

    let mut broken = BoardState::new();
    broken.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    broken.piece_mapping[Square::E4 as usize] = Piece::None;
    assert!(!broken.is_pseudo_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));
}

#[test]
fn pawn_pseudo_legal_captures_promotions_and_en_passant() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    board.add_piece(Square::D5, Side::Black, Piece::Pawn, false);
    assert!(board.is_pseudo_legal(Move::new(Square::E4, Square::D5, MoveType::Capture)));
    assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::D5, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::E6, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::E4, Square::F5, MoveType::Capture)));

    let mut promo = BoardState::new();
    promo.add_piece(Square::E7, Side::White, Piece::Pawn, false);
    assert!(promo.is_pseudo_legal(Move::new(Square::E7, Square::E8, MoveType::QueenPromotion)));
    assert!(!promo.is_pseudo_legal(Move::new(Square::E7, Square::E8, MoveType::Quiet)));
    assert!(!promo.is_pseudo_legal(Move::new(
        Square::E7,
        Square::E8,
        MoveType::QueenPromotionCapture
    )));
    let mut black_promo = BoardState::new();
    black_promo.add_piece(Square::E2, Side::Black, Piece::Pawn, false);
    black_promo.side_to_move = Side::Black;
    assert!(black_promo.is_pseudo_legal(Move::new(
        Square::E2,
        Square::E1,
        MoveType::QueenPromotion
    )));
    assert!(!black_promo.is_pseudo_legal(Move::new(Square::E2, Square::E1, MoveType::Quiet)));

    let mut promo_cap = BoardState::new();
    promo_cap.add_piece(Square::F7, Side::White, Piece::Pawn, false);
    promo_cap.add_piece(Square::G8, Side::Black, Piece::Rook, false);
    assert!(promo_cap.is_pseudo_legal(Move::new(
        Square::F7,
        Square::G8,
        MoveType::QueenPromotionCapture
    )));

    let ep = BoardState::parse_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    assert!(ep.is_pseudo_legal(Move::new(Square::E5, Square::D6, MoveType::EnPassant)));
    let mut no_ep = BoardState::new();
    no_ep.add_piece(Square::E5, Side::White, Piece::Pawn, false);
    assert!(!no_ep.is_pseudo_legal(Move::new(Square::E5, Square::D6, MoveType::EnPassant)));
}

#[test]
fn knight_pseudo_legal_branches() {
    let mut board = BoardState::new();
    board.add_piece(Square::B1, Side::White, Piece::Knight, false);
    board.add_piece(Square::C3, Side::Black, Piece::Pawn, false);
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E8, Side::Black, Piece::King, false);
    assert!(board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Capture)));

    assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::B3, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::C3, MoveType::Castle)));
    assert!(!board.is_pseudo_legal(Move::new(Square::B1, Square::A3, MoveType::KnightPromotion)));
}

#[test]
fn slider_pseudo_legal_branches() {
    let mut board = BoardState::new();
    board.add_piece(Square::C1, Side::White, Piece::Bishop, false);
    board.add_piece(Square::D2, Side::White, Piece::Pawn, false);
    board.add_piece(Square::A1, Side::White, Piece::Rook, false);
    board.add_piece(Square::A2, Side::White, Piece::Pawn, false);
    board.add_piece(Square::D1, Side::White, Piece::Queen, false);
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E8, Side::Black, Piece::King, false);

    assert!(!board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Quiet)));
    assert!(!board.is_pseudo_legal(Move::new(Square::A1, Square::A4, MoveType::Quiet)));
    assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D3, MoveType::Quiet)));

    assert!(!board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Castle)));
    assert!(!board.is_pseudo_legal(Move::new(Square::A1, Square::A3, MoveType::RookPromotion)));
    assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D2, MoveType::QueenPromotion)));

    board.remove_piece(Square::D2, false);
    board.add_piece(Square::E3, Side::Black, Piece::Pawn, false);
    assert!(board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Capture)));
    assert!(!board.is_pseudo_legal(Move::new(Square::C1, Square::E3, MoveType::Quiet)));

    board.remove_piece(Square::A2, false);
    board.add_piece(Square::A4, Side::Black, Piece::Rook, false);
    assert!(board.is_pseudo_legal(Move::new(Square::A1, Square::A4, MoveType::Capture)));
    assert!(board.is_pseudo_legal(Move::new(Square::A1, Square::A2, MoveType::Quiet)));

    board.remove_piece(Square::D2, false);
    board.add_piece(Square::D3, Side::Black, Piece::Knight, false);
    assert!(board.is_pseudo_legal(Move::new(Square::D1, Square::D3, MoveType::Capture)));
    assert!(!board.is_pseudo_legal(Move::new(Square::D1, Square::D4, MoveType::Quiet)));
}

#[test]
fn king_pseudo_legal_castle_branches() {
    use crate::common::helpers::STARTING_FEN;

    let mut open = BoardState::new();
    open.add_piece(Square::E1, Side::White, Piece::King, false);
    open.add_piece(Square::E8, Side::Black, Piece::King, false);
    assert!(!open.is_pseudo_legal(Move::new(Square::E1, Square::E2, MoveType::QueenPromotion)));

    assert!(!open.is_pseudo_legal(Move::new(Square::E1, Square::E3, MoveType::Quiet)));

    assert!(!open.is_pseudo_legal(Move::new(Square::E1, Square::E2, MoveType::Capture)));
    assert!(open.is_pseudo_legal(Move::new(Square::E1, Square::E2, MoveType::Quiet)));

    let mut off = BoardState::new();
    off.add_piece(Square::E2, Side::White, Piece::King, false);
    off.add_piece(Square::E8, Side::Black, Piece::King, false);
    off.castle = Castle::WHITE_SHORT;
    assert!(!off.is_pseudo_legal(Move::new(Square::E2, Square::G2, MoveType::Castle)));
    let home = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    assert!(!home.is_pseudo_legal(Move::new(Square::E1, Square::F1, MoveType::Castle)));

    let no_rights = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1");
    assert!(!no_rights.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    let start = BoardState::parse_fen(STARTING_FEN);
    assert!(!start.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(!start.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));

    let attacked = BoardState::parse_fen("r3k2r/pppppppp/8/8/2b5/8/PPPP1PPP/R3K2R w KQkq - 0 1");
    assert!(!attacked.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(attacked.is_pseudo_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));

    assert!(home.is_pseudo_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(home.is_pseudo_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));
    let black = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
    assert!(black.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
    assert!(black.is_pseudo_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));
}

#[test]
fn is_legal_rejects_null_moves_and_kingless_boards() {
    use crate::common::move_list::MoveList;

    let board = BoardState::starting_position();
    assert!(!board.is_legal(Move::NO_MOVE));
    assert!(!board.is_legal(Move::new(Square::NoSquare, Square::E4, MoveType::Quiet)));
    assert!(!board.is_legal(Move::new(Square::E2, Square::NoSquare, MoveType::Quiet)));

    let mut kingless = BoardState::new();
    kingless.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    assert!(!kingless.is_legal(Move::new(Square::E4, Square::E5, MoveType::Quiet)));

    let mut moves = MoveList::new();
    kingless.generate_moves(&mut moves);
    for entry in moves.iter() {
        assert!(!kingless.is_legal(entry.mv));
    }
}

#[test]
fn is_legal_castling_needs_rook_rights_and_quiet_path() {
    let white = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    assert!(white.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(white.is_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));
    let black = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
    assert!(black.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
    assert!(black.is_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));

    let mut rookless = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w K - 0 1");
    rookless.remove_piece(Square::H1, false);
    let castle = Move::new(Square::E1, Square::G1, MoveType::Castle);
    assert!(rookless.is_pseudo_legal(castle));
    assert!(!rookless.is_legal(castle));

    let mut off_home = BoardState::new();
    off_home.add_piece(Square::E2, Side::White, Piece::King, false);
    off_home.add_piece(Square::E8, Side::Black, Piece::King, false);
    off_home.castle = Castle::WHITE_SHORT;
    assert!(!off_home.is_legal(Move::new(Square::E2, Square::G2, MoveType::Castle)));
    assert!(!white.is_legal(Move::new(Square::E1, Square::F1, MoveType::Castle)));
    let no_rights = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1");
    assert!(!no_rights.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(!BoardState::starting_position().is_legal(Move::new(
        Square::E1,
        Square::G1,
        MoveType::Castle
    )));
}

#[test]
fn is_legal_castling_respects_checks() {
    let in_check = BoardState::parse_fen("4r1k1/8/8/8/8/8/8/R3K2R w K - 0 1");
    assert!(in_check.is_in_check(Side::White));
    assert!(!in_check.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));

    let transit = BoardState::parse_fen("r3k2r/pppppppp/8/8/2b5/8/PPPP1PPP/R3K2R w KQkq - 0 1");
    assert!(!transit.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(transit.is_legal(Move::new(Square::E1, Square::C1, MoveType::Castle)));

    let dest = BoardState::parse_fen("4k3/8/8/8/8/8/7b/R3K2R w K - 0 1");
    assert!(!dest.is_legal(Move::new(Square::E1, Square::G1, MoveType::Castle)));
    assert!(dest.is_legal(Move::new(Square::E1, Square::F1, MoveType::Quiet)));
}

#[test]
fn is_legal_king_steps_out_of_check() {
    let board = BoardState::parse_fen("4r1k1/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    assert!(board.is_in_check(Side::White));

    assert!(board.is_legal(Move::new(Square::E1, Square::F1, MoveType::Quiet)));

    assert!(!board.is_legal(Move::new(Square::E1, Square::E2, MoveType::Quiet)));
}

#[test]
fn is_legal_en_passant_pin_and_precheck_branches() {
    let free = BoardState::parse_fen("4k3/8/8/3Pp3/4K3/8/8/8 w - e6 0 1");
    let mv = Move::new(Square::D5, Square::E6, MoveType::EnPassant);
    assert!(free.is_pseudo_legal(mv));
    assert!(free.is_legal(mv));

    let rook = BoardState::parse_fen("7k/8/8/r2Pp2K/8/8/8/8 w - e6 0 1");
    let rm = Move::new(Square::D5, Square::E6, MoveType::EnPassant);
    assert!(rook.is_pseudo_legal(rm));
    assert!(!rook.is_legal(rm));

    let bishop = BoardState::parse_fen("4k3/1b6/8/3Pp3/4K3/8/8/8 w - e6 0 1");
    assert!(!bishop.is_legal(Move::new(Square::D5, Square::E6, MoveType::EnPassant)));

    let knight = BoardState::parse_fen("4k3/8/3n4/3Pp3/4K3/8/8/8 w - e6 0 1");
    assert!(knight.is_in_check(Side::White));
    assert!(!knight.is_legal(Move::new(Square::D5, Square::E6, MoveType::EnPassant)));

    let pawn = BoardState::parse_fen("4k3/8/8/3Ppp2/4K3/8/8/8 w - e6 0 1");
    assert!(!pawn.is_legal(Move::new(Square::D5, Square::E6, MoveType::EnPassant)));
}

#[test]
fn is_legal_single_and_double_check() {
    let dbl = BoardState::parse_fen("4rk2/8/8/8/1b6/8/8/4K1N1 w - - 0 1");
    assert!(dbl.is_in_check(Side::White));
    assert!(!dbl.is_legal(Move::new(Square::G1, Square::F3, MoveType::Quiet)));
    assert!(dbl.is_legal(Move::new(Square::E1, Square::F1, MoveType::Quiet)));

    let pin = BoardState::parse_fen("4r1k1/8/8/8/8/8/4Q3/4K3 w - - 0 1");
    assert!(!pin.is_in_check(Side::White));
    assert!(pin.is_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));
    assert!(!pin.is_legal(Move::new(Square::E2, Square::D3, MoveType::Quiet)));
    assert!(pin.is_legal(Move::new(Square::E2, Square::E8, MoveType::Capture)));

    let block = BoardState::parse_fen("4k3/8/8/8/1b6/8/P1P5/4K3 w - - 0 1");
    assert!(block.is_legal(Move::new(Square::C2, Square::C3, MoveType::Quiet)));
    assert!(!block.is_legal(Move::new(Square::A2, Square::A3, MoveType::Quiet)));
}

#[test]
fn square_attacked_covers_every_piece_type() {
    let start = BoardState::parse_fen(crate::common::helpers::STARTING_FEN);
    assert!(start.is_square_attacked(Square::E3, Side::White));

    let knight = BoardState::parse_fen("4k3/8/8/8/8/3n4/8/4K3 w - - 0 1");
    assert!(knight.is_square_attacked(Square::E1, Side::Black));
    let king = BoardState::parse_fen("4k3/8/8/8/8/8/3k4/4K3 w - - 0 1");
    assert!(king.is_square_attacked(Square::E1, Side::Black));
    let queen = BoardState::parse_fen("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1");
    assert!(queen.is_square_attacked(Square::E1, Side::Black));

    let bare = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert!(!bare.is_square_attacked(Square::E4, Side::Black));
    assert!(!bare.is_square_attacked(Square::E4, Side::White));
}

#[test]
fn checkers_cover_all_piece_types() {
    let pawn = BoardState::parse_fen("4k3/8/8/3p4/4K3/8/8/8 w - - 0 1");
    assert_eq!(pawn.checkers(Side::White).0, 1u64 << Square::D5 as usize);

    let knight = BoardState::parse_fen("4k3/8/3n4/8/4K3/8/8/8 w - - 0 1");
    assert_eq!(knight.checkers(Side::White).0, 1u64 << Square::D6 as usize);

    let bishop = BoardState::parse_fen("4k3/1b6/8/8/4K3/8/8/8 w - - 0 1");
    assert_eq!(bishop.checkers(Side::White).0, 1u64 << Square::B7 as usize);

    let queen = BoardState::parse_fen("4q3/8/8/8/4K3/8/8/8 w - - 0 1");
    assert_eq!(queen.checkers(Side::White).0, 1u64 << Square::E8 as usize);
}

#[test]
fn pinned_pieces_with_two_blockers_is_empty() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E2, Side::White, Piece::Pawn, false);
    board.add_piece(Square::E3, Side::White, Piece::Pawn, false);
    board.add_piece(Square::E8, Side::Black, Piece::Rook, false);
    assert!(board.pinned_pieces(Side::White).is_empty());

    let mut diag = BoardState::new();
    diag.add_piece(Square::E1, Side::White, Piece::King, false);
    diag.add_piece(Square::D2, Side::White, Piece::Bishop, false);
    diag.add_piece(Square::B4, Side::Black, Piece::Bishop, false);
    assert_eq!(
        diag.pinned_pieces(Side::White).0,
        1u64 << Square::D2 as usize
    );
}

#[test]
fn pseudo_legal_black_castle_edges() {
    let mut off = BoardState::new();
    off.add_piece(Square::E7, Side::Black, Piece::King, false);
    off.add_piece(Square::E1, Side::White, Piece::King, false);
    off.side_to_move = Side::Black;
    off.castle = Castle::BLACK_SHORT;
    assert!(!off.is_pseudo_legal(Move::new(Square::E7, Square::G8, MoveType::Castle)));

    let black = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
    assert!(!black.is_pseudo_legal(Move::new(Square::E8, Square::F8, MoveType::Castle)));

    let transit = BoardState::parse_fen("r3k2r/8/8/2B5/8/8/8/4K3 b kq - 0 1");
    assert!(!transit.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
    assert!(transit.is_pseudo_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));

    let dest = BoardState::parse_fen("4k3/8/8/8/8/8/B7/4K3 b k - 0 1");
    assert!(!dest.is_pseudo_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
}

#[test]
fn is_legal_black_castling_attack_and_check_branches() {
    let transit = BoardState::parse_fen("r3k2r/8/8/2B5/8/8/8/4K3 b kq - 0 1");
    assert!(!transit.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));
    assert!(transit.is_legal(Move::new(Square::E8, Square::C8, MoveType::Castle)));

    let dest = BoardState::parse_fen("4k3/8/8/8/8/8/B7/4K3 b k - 0 1");
    assert!(!dest.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));

    let in_check = BoardState::parse_fen("4k3/4Q3/8/8/8/8/8/4K3 b k - 0 1");
    assert!(in_check.is_in_check(Side::Black));
    assert!(!in_check.is_legal(Move::new(Square::E8, Square::G8, MoveType::Castle)));

    let mut off_home = BoardState::new();
    off_home.add_piece(Square::E7, Side::Black, Piece::King, false);
    off_home.add_piece(Square::E1, Side::White, Piece::King, false);
    off_home.side_to_move = Side::Black;
    off_home.castle = Castle::BLACK_SHORT;
    assert!(!off_home.is_legal(Move::new(Square::E7, Square::G8, MoveType::Castle)));
    let home = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
    assert!(!home.is_legal(Move::new(Square::E8, Square::F8, MoveType::Castle)));

    let mut rookless = BoardState::parse_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b kq - 0 1");
    rookless.remove_piece(Square::A8, false);
    let castle = Move::new(Square::E8, Square::C8, MoveType::Castle);
    assert!(rookless.is_pseudo_legal(castle));
    assert!(!rookless.is_legal(castle));
}

#[test]
fn pin_edge_cases_adjacent_pinner_and_enemy_blocker() {
    let adj = BoardState::parse_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
    assert!(adj.pinned_pieces(Side::White).is_empty());
    assert_eq!(adj.checkers(Side::White).0, 1u64 << Square::E2 as usize);

    let enemy = BoardState::parse_fen("k3r3/8/8/8/8/8/4p3/4K3 w - - 0 1");
    assert!(!enemy.is_in_check(Side::White));
    assert!(enemy.pinned_pieces(Side::White).is_empty());
}

#[test]
fn queen_attacks_and_checkers_on_files_and_diagonals() {
    let file = BoardState::parse_fen("k3q3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert!(file.is_square_attacked(Square::E1, Side::Black));

    let diag = BoardState::parse_fen("4k3/1q6/8/8/4K3/8/8/8 w - - 0 1");
    assert_eq!(diag.checkers(Side::White).0, 1u64 << Square::B7 as usize);
    assert!(diag.is_in_check(Side::White));
}

#[test]
fn passed_pawns_with_multiple_pawns_per_side() {
    let mut board = BoardState::new();
    board.add_piece(Square::E1, Side::White, Piece::King, false);
    board.add_piece(Square::E8, Side::Black, Piece::King, false);
    board.add_piece(Square::A4, Side::White, Piece::Pawn, false);
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    board.add_piece(Square::B5, Side::Black, Piece::Pawn, false);
    board.add_piece(Square::H5, Side::Black, Piece::Pawn, false);

    assert!(!board.is_passed_pawn(Square::A4 as usize, Side::White));
    assert!(board.is_passed_pawn(Square::E4 as usize, Side::White));
    assert_eq!(
        board.get_passed_pawns(Side::White).0,
        1u64 << Square::E4 as usize
    );

    assert!(!board.is_passed_pawn(Square::B5 as usize, Side::Black));
    assert!(board.is_passed_pawn(Square::H5 as usize, Side::Black));
    assert_eq!(
        board.get_passed_pawns(Side::Black).0,
        1u64 << Square::H5 as usize
    );
}

#[test]
fn is_legal_pinned_piece_in_single_knight_check() {
    let board = BoardState::parse_fen("k3r3/8/8/8/8/3n4/2P1Q3/4K3 w - - 0 1");
    assert!(board.is_in_check(Side::White));

    assert!(!board.is_legal(Move::new(Square::E2, Square::D3, MoveType::Capture)));

    assert!(!board.is_legal(Move::new(Square::E2, Square::E3, MoveType::Quiet)));

    assert!(board.is_legal(Move::new(Square::C2, Square::D3, MoveType::Capture)));

    assert!(!board.is_legal(Move::new(Square::C2, Square::C3, MoveType::Quiet)));
}

#[test]
fn remove_piece_on_empty_square_returns_none() {
    let mut board = BoardState::new();
    assert_eq!(board.remove_piece(Square::E4, false), Piece::None);
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    assert_eq!(board.remove_piece(Square::E4, false), Piece::Pawn);
    assert_eq!(board.remove_piece(Square::E4, false), Piece::None);
}

#[test]
fn get_piece_on_covers_pawn_values_and_minor_material() {
    let mut board = BoardState::new();
    board.add_piece(Square::E4, Side::White, Piece::Pawn, false);
    board.add_piece(Square::E5, Side::Black, Piece::Pawn, false);
    board.add_piece(Square::C3, Side::White, Piece::Knight, false);
    assert_eq!(board.get_piece_on(Square::E4), Piece::Pawn as i32);
    assert_eq!(board.get_piece_on(Square::E5), 6 + Piece::Pawn as i32);
    assert!(board.has_non_pawn_material(Side::White));
    assert!(!board.has_non_pawn_material(Side::Black));
}

#[test]
fn pawn_pseudo_legal_black_en_passant_and_blocked_double_push() {
    let ep = BoardState::parse_fen("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
    assert!(ep.is_pseudo_legal(Move::new(Square::D4, Square::E3, MoveType::EnPassant)));
    assert!(!ep.is_pseudo_legal(Move::new(Square::D4, Square::C3, MoveType::EnPassant)));

    let mut blocked = BoardState::new();
    blocked.add_piece(Square::E7, Side::Black, Piece::Pawn, false);
    blocked.add_piece(Square::E6, Side::White, Piece::Pawn, false);
    blocked.side_to_move = Side::Black;
    assert!(!blocked.is_pseudo_legal(Move::new(Square::E7, Square::E5, MoveType::DoublePush)));
    assert!(!blocked.is_pseudo_legal(Move::new(Square::E7, Square::E6, MoveType::Quiet)));
}

#[test]
fn threat_by_lesser_on_bare_and_queen_boards() {
    let bare = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert_eq!(bare.threat_by_lesser(Side::White), [0; 6]);

    let queen = BoardState::parse_fen("4k3/8/8/3q4/8/8/8/4K3 w - - 0 1");
    let threats = queen.threat_by_lesser(Side::White);
    assert_eq!(threats[Piece::Pawn as usize], 0);
    assert_eq!(threats[Piece::King as usize], 0);
    assert_ne!(threats[Piece::Rook as usize], 0);
    assert_ne!(threats[Piece::Queen as usize], 0);
    assert_eq!(
        threats[Piece::Queen as usize] & threats[Piece::Rook as usize],
        threats[Piece::Rook as usize]
    );

    assert_ne!(
        threats[Piece::Queen as usize] & (1u64 << Square::D1 as usize),
        0
    );
    assert_ne!(
        threats[Piece::Rook as usize] & (1u64 << Square::A2 as usize),
        0
    );
}
