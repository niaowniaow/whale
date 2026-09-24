use crate::board::state::BoardState;

#[inline(always)]
pub fn is_draw(board: &BoardState, ply: u16) -> bool {
    board.is_draw_in_search(ply)
}

#[inline(always)]
pub fn is_game_draw(board: &BoardState) -> bool {
    board.is_draw()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    #[test]
    fn startpos_is_not_draw() {
        let b = BoardState::parse_fen(STARTING_FEN);
        assert!(!is_draw(&b, 0));
        assert!(!is_game_draw(&b));
    }
}
