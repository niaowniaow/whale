use crate::bitboard::lookups::{
    get_bishop_attacks_from_table, get_rook_attacks_from_table, king_attacks, knight_attacks,
    pawn_attacks,
};
use crate::board::node_threats::NodeThreats;
use crate::board::state::BoardState;
use crate::common::move_list::MoveList;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

pub fn available_pawn_breaks(board: &BoardState) -> u32 {
    count_piece_moves(board, Piece::Pawn, true)
}

pub fn knight_routes(board: &BoardState) -> u32 {
    count_piece_moves(board, Piece::Knight, false)
}

fn count_piece_moves(board: &BoardState, piece: Piece, non_captures_only: bool) -> u32 {
    let nt = NodeThreats::compute(board);
    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);
    let mut n = 0u32;
    for i in 0..moves.len() {
        let m = moves[i].mv;
        if non_captures_only && m.is_capture() {
            continue;
        }
        if board.piece_mapping[m.source as usize] != piece {
            continue;
        }
        if board.is_legal_with(m, nt.checkers, nt.pinned) {
            n += 1;
        }
    }
    n
}

pub fn defenders_of(board: &BoardState, sq: Square, side: Side) -> u32 {
    let s = sq as usize;
    let occ = board.occupancy();
    let mut n = 0u32;
    n += (board.get_pieces(side, Piece::Pawn).0 & pawn_attacks()[side.other() as usize][s])
        .count_ones();
    n += (board.get_pieces(side, Piece::Knight).0 & knight_attacks()[s]).count_ones();
    n += (board.get_pieces(side, Piece::King).0 & king_attacks()[s]).count_ones();
    let bq = board.get_pieces(side, Piece::Bishop).0 | board.get_pieces(side, Piece::Queen).0;
    n += (get_bishop_attacks_from_table(sq, occ).0 & bq).count_ones();
    let rq = board.get_pieces(side, Piece::Rook).0 | board.get_pieces(side, Piece::Queen).0;
    n += (get_rook_attacks_from_table(sq, occ).0 & rq).count_ones();
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;
    use crate::common::square::Square;

    #[test]
    fn starting_plan_counts() {
        let board = BoardState::parse_fen(STARTING_FEN);

        assert_eq!(available_pawn_breaks(&board), 16);

        assert_eq!(knight_routes(&board), 4);
    }

    #[test]
    fn defender_counts_sane_at_start() {
        let board = BoardState::parse_fen(STARTING_FEN);

        assert_eq!(defenders_of(&board, Square::E2, Side::White), 4);

        assert_eq!(defenders_of(&board, Square::E4, Side::White), 0);
    }

    #[test]
    fn breaks_drop_after_pawn_push() {
        let mut board = BoardState::parse_fen(STARTING_FEN);
        let before = available_pawn_breaks(&board);

        let mut list = MoveList::new();
        board.generate_moves(&mut list);
        let e4 = list
            .iter()
            .map(|e| e.mv)
            .find(|m| m.source == Square::E2 && m.target == Square::E4)
            .expect("e2e4 must exist");
        board.make_move(e4);
        assert!(available_pawn_breaks(&board) <= before);

        assert_eq!(knight_routes(&board), 4);
    }
}
