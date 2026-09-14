use crate::board::state::BoardState;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

const fn next_u64(mut state: u64) -> (u64, u64) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    (state, state)
}

const fn generate_zobrist_table() -> [[u64; 64]; 14] {
    let mut table = [[0u64; 64]; 14];
    let mut state = 1804289383u64;

    let mut i = 0;
    while i < 14 {
        let mut j = 0;
        while j < 64 {
            let (next_state, val) = next_u64(state);
            state = next_state;
            table[i][j] = val;
            j += 1;
        }
        i += 1;
    }

    table
}

// TODO: flatten
pub static ZOBRIST_TABLE: [[u64; 64]; 14] = generate_zobrist_table();

pub fn init() {}

#[inline(always)]
pub fn zobrist_table() -> &'static [[u64; 64]; 14] {
    &ZOBRIST_TABLE
}

pub fn get_board_hash(board_state: &BoardState) -> u64 {
    let mut current_hash = 0;

    for square in 0..64 {
        let piece = board_state.get_piece_on(Square::from(square));
        if piece != -1 {
            current_hash ^= zobrist_table()[piece as usize][square];
        }
    }

    current_hash = hash_side_to_move(board_state, current_hash);
    current_hash = hash_castling_rights(board_state, current_hash);
    current_hash = hash_en_passant(board_state, current_hash);

    current_hash
}

pub fn hash_castling_rights(board_state: &BoardState, current_hash: u64) -> u64 {
    // Offset by 2 to avoid collision with side-to-move keys (which use [13][0] and [13][1])
    current_hash ^ zobrist_table()[13][2 + board_state.castle.bits() as usize]
}

fn hash_side_to_move(board_state: &BoardState, current_hash: u64) -> u64 {
    current_hash
        ^ if board_state.side_to_move == Side::White {
            zobrist_table()[13][0]
        } else {
            zobrist_table()[13][1]
        }
}

pub fn flip_side_to_move_hashes(_board_state: &BoardState, current_hash: u64) -> u64 {
    current_hash ^ zobrist_table()[13][0] ^ zobrist_table()[13][1]
}

pub fn hash_en_passant(board_state: &BoardState, current_hash: u64) -> u64 {
    if board_state.en_passant_square != Square::NoSquare {
        current_hash ^ zobrist_table()[12][board_state.en_passant_square as usize]
    } else {
        current_hash
    }
}

pub fn get_pawn_hash(board_state: &BoardState) -> u64 {
    let mut current_hash = 0;
    let mut white_pawns =
        (board_state.pieces[Piece::Pawn] & board_state.occupancies[Side::White]).0;
    while white_pawns != 0 {
        let sq = white_pawns.trailing_zeros() as usize;
        current_hash ^= zobrist_table()[0][sq];
        white_pawns &= white_pawns - 1;
    }
    let mut black_pawns =
        (board_state.pieces[Piece::Pawn] & board_state.occupancies[Side::Black]).0;
    while black_pawns != 0 {
        let sq = black_pawns.trailing_zeros() as usize;
        current_hash ^= zobrist_table()[6][sq];
        black_pawns &= black_pawns - 1;
    }
    current_hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn test_starting_hash_is_deterministic() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let hash = get_board_hash(&board);
        assert_eq!(hash, 17316932686648747093);
    }

    #[test]
    fn test_pawn_hash_matches_starting_fen() {
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut expected = 0;
        for square in 0..64 {
            let piece = board.get_piece_on(Square::from(square));
            if piece == 0 || piece == 6 {
                expected ^= zobrist_table()[piece as usize][square];
            }
        }
        assert_eq!(get_pawn_hash(&board), expected);
    }
}
