use crate::board::state::BoardState;
use crate::common::castle::Castle;
use crate::common::helpers::STARTING_FEN;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;
use crate::common::zobrist;

impl BoardState {
    pub fn parse_fen(fen: &str) -> Self {
        let mut board = BoardState::new();
        let sections: Vec<&str> = fen.split(' ').collect();
        if sections.len() < 4 {
            board.board_hash = zobrist::get_board_hash(&board);
            return board;
        }

        parse_pieces(&mut board, sections[0]);
        parse_side_to_move(&mut board, sections[1]);
        parse_castling(&mut board, sections[2]);
        parse_en_passant(&mut board, sections[3]);
        if sections.len() > 4 {
            parse_ply(&mut board, sections[4]);
        }
        if sections.len() > 5 {
            parse_move_count(&mut board, sections[5]);
        }
        board.board_hash = zobrist::get_board_hash(&board);

        board
    }

    pub fn starting_position() -> Self {
        Self::parse_fen(STARTING_FEN)
    }

    pub fn to_fen(&self) -> String {
        let mut board = String::new();
        for rank in 0..8 {
            if rank > 0 {
                board.push('/');
            }
            let mut empty = 0;
            for file in 0..8 {
                let square = rank * 8 + file;
                let piece = self.piece_mapping[square];
                if piece == Piece::None {
                    empty += 1;
                    continue;
                }
                if empty > 0 {
                    board.push(char::from_digit(empty, 10).unwrap());
                    empty = 0;
                }
                let mut symbol = match piece {
                    Piece::Pawn => 'p',
                    Piece::Knight => 'n',
                    Piece::Bishop => 'b',
                    Piece::Rook => 'r',
                    Piece::Queen => 'q',
                    Piece::King => 'k',
                    Piece::None => unreachable!(),
                };
                if self.occupancies[Side::White].get_bit(square) == 1 {
                    symbol = symbol.to_ascii_uppercase();
                }
                board.push(symbol);
            }
            if empty > 0 {
                board.push(char::from_digit(empty, 10).unwrap());
            }
        }

        let castling = if self.castle == Castle::NONE {
            "-".to_string()
        } else {
            let mut rights = String::new();
            if self.castle.contains(Castle::WHITE_SHORT) {
                rights.push('K');
            }
            if self.castle.contains(Castle::WHITE_LONG) {
                rights.push('Q');
            }
            if self.castle.contains(Castle::BLACK_SHORT) {
                rights.push('k');
            }
            if self.castle.contains(Castle::BLACK_LONG) {
                rights.push('q');
            }
            rights
        };

        format!(
            "{} {} {} {} {} {}",
            board,
            if self.side_to_move == Side::White {
                "w"
            } else {
                "b"
            },
            castling,
            self.en_passant_square,
            self.half_move_clock,
            self.move_count / 2 + 1,
        )
    }
}

fn parse_pieces(board: &mut BoardState, fen: &str) {
    for (rank, rank_str) in fen.split('/').enumerate() {
        if rank >= 8 {
            break;
        }
        let mut index = rank * 8;
        for symbol in rank_str.chars() {
            if index >= 64 {
                break;
            }
            if symbol.is_ascii_alphabetic() {
                board.add_piece(
                    Square::from(index),
                    symbol_to_side(symbol),
                    symbol_to_piece(symbol),
                    true,
                );
                board.flush_pending_updates(board.history.index);
                index += 1;
            } else if let Some(skip) = symbol.to_digit(10) {
                index += skip as usize;
            }
        }
    }
}

fn parse_side_to_move(board: &mut BoardState, fen: &str) {
    board.side_to_move = if fen == "w" { Side::White } else { Side::Black };
}

fn parse_castling(board: &mut BoardState, fen: &str) {
    for ch in fen.chars() {
        match ch {
            'K' => board.castle |= Castle::WHITE_SHORT,
            'Q' => board.castle |= Castle::WHITE_LONG,
            'k' => board.castle |= Castle::BLACK_SHORT,
            'q' => board.castle |= Castle::BLACK_LONG,
            _ => {}
        }
    }
}

fn parse_en_passant(board: &mut BoardState, fen: &str) {
    if fen == "-" {
        return;
    }
    let bytes = fen.as_bytes();
    if bytes.len() < 2 || bytes[0] < b'a' || bytes[0] > b'h' || bytes[1] < b'1' || bytes[1] > b'8' {
        return;
    }
    let file = (bytes[0] - b'a') as usize;
    let rank = (bytes[1] - b'1') as usize;
    let sq_index = (7 - rank) * 8 + file;
    if sq_index >= 64 {
        return;
    }
    let sq_rank = sq_index / 8;
    let valid_rank = (board.side_to_move == Side::Black && sq_rank == 5)
        || (board.side_to_move == Side::White && sq_rank == 2);
    if !valid_rank {
        return;
    }
    let pushed = if board.side_to_move == Side::Black {
        sq_index.saturating_sub(8)
    } else {
        sq_index.saturating_add(8).min(63)
    };
    let left = if pushed % 8 == 0 {
        0
    } else {
        1u64 << (pushed - 1)
    };
    let right = if pushed % 8 == 7 {
        0
    } else {
        1u64 << (pushed + 1)
    };
    let own_pawns = board.get_pieces(board.side_to_move, Piece::Pawn).0;
    if own_pawns & (left | right) == 0 {
        return;
    }
    // The enemy pawn that just double-pushed must still stand immediately
    // in front of the EP square (Stockfish position.cpp: target bitboard).
    let enemy = board.side_to_move.other();
    if board.piece_mapping[pushed] != Piece::Pawn || board.occupancies[enemy].get_bit(pushed) == 0 {
        return;
    }
    // Both the EP square and the square behind it must be empty
    // (Stockfish: nothing on epSquare or epSquare + pawn_push(side)).
    let behind = if board.side_to_move == Side::White {
        sq_index.saturating_sub(8)
    } else {
        sq_index.saturating_add(8).min(63)
    };
    let occ = board.occupancy();
    if occ.get_bit(sq_index) == 1 || occ.get_bit(behind) == 1 {
        return;
    }
    board.en_passant_square = Square::from(sq_index);
}

fn parse_ply(board: &mut BoardState, halfmove: &str) {
    if let Ok(clock) = halfmove.parse::<u8>() {
        board.half_move_clock = clock;
    }
}

fn parse_move_count(board: &mut BoardState, fullmove: &str) {
    if let Ok(fm) = fullmove.parse::<i32>() {
        let plies = (fm.max(1) - 1) * 2;
        board.move_count = plies
            + if board.side_to_move == Side::White {
                0
            } else {
                1
            };
    }
}

pub fn symbol_to_piece(symbol: char) -> Piece {
    match symbol.to_ascii_lowercase() {
        'p' => Piece::Pawn,
        'r' => Piece::Rook,
        'n' => Piece::Knight,
        'b' => Piece::Bishop,
        'q' => Piece::Queen,
        'k' => Piece::King,
        _ => Piece::None,
    }
}

pub fn symbol_to_side(symbol: char) -> Side {
    if symbol.is_ascii_uppercase() {
        Side::White
    } else {
        Side::Black
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn should_parse_starting_fen_piece_bitboards() {
        let board = BoardState::parse_fen(STARTING_FEN);

        assert_eq!(
            board.get_pieces(Side::White, Piece::Pawn).0,
            71776119061217280
        );
        assert_eq!(
            board.get_pieces(Side::White, Piece::Knight).0,
            4755801206503243776
        );
        assert_eq!(
            board.get_pieces(Side::White, Piece::Bishop).0,
            2594073385365405696
        );
        assert_eq!(
            board.get_pieces(Side::White, Piece::Rook).0,
            9295429630892703744
        );
        assert_eq!(
            board.get_pieces(Side::White, Piece::Queen).0,
            576460752303423488
        );
        assert_eq!(
            board.get_pieces(Side::White, Piece::King).0,
            1152921504606846976
        );

        assert_eq!(board.get_pieces(Side::Black, Piece::Pawn).0, 65280);
        assert_eq!(board.get_pieces(Side::Black, Piece::Knight).0, 66);
        assert_eq!(board.get_pieces(Side::Black, Piece::Bishop).0, 36);
        assert_eq!(board.get_pieces(Side::Black, Piece::Rook).0, 129);
        assert_eq!(board.get_pieces(Side::Black, Piece::Queen).0, 8);
        assert_eq!(board.get_pieces(Side::Black, Piece::King).0, 16);
    }

    #[test]
    fn serializes_starting_fen_without_loss() {
        assert_eq!(BoardState::starting_position().to_fen(), STARTING_FEN);
    }
}
