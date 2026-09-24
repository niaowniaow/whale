use crate::board::state::BoardState;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;
use crate::eval::nnue::accumulator::Accumulator;
use crate::eval::nnue::features::get_feature_index;
use crate::eval::nnue::loader::Network;
use crate::eval::nnue::v16::{self as sfnn16, SfnnPosition};

impl BoardState {
    #[inline(always)]
    pub fn nnue_add_piece(&mut self, square: Square, side: Side, piece: Piece) {
        if let Some(w_idx) = get_feature_index(piece, side, square) {
            let idx = self.pending_adds as usize;
            if idx < 2 {
                let mirrored_sq = square.mirrored();
                if let Some(b_idx) = get_feature_index(piece, side.other(), mirrored_sq) {
                    self.pending_adds_w[idx] = w_idx;
                    self.pending_adds_b[idx] = b_idx;
                    self.pending_adds += 1;
                }
            }
        }
        sfnn16::note_add(&mut self.sfnn16_pending, square, side, piece);
    }

    #[inline(always)]
    pub fn nnue_remove_piece(&mut self, square: Square, side: Side, piece: Piece) {
        if let Some(w_idx) = get_feature_index(piece, side, square) {
            let idx = self.pending_removes as usize;
            if idx < 2 {
                let mirrored_sq = square.mirrored();
                if let Some(b_idx) = get_feature_index(piece, side.other(), mirrored_sq) {
                    self.pending_dels_w[idx] = w_idx;
                    self.pending_dels_b[idx] = b_idx;
                    self.pending_removes += 1;
                }
            }
        }
        sfnn16::note_remove(&mut self.sfnn16_pending, square, side, piece);
    }

    pub fn flush_pending_updates(&mut self, target_idx: usize) {
        let network = Network::get_embedded();
        match (self.pending_adds, self.pending_removes) {
            (0, 0) => {}
            (1, 0) => {
                self.history.accumulators[target_idx]
                    .white
                    .add_feature(self.pending_adds_w[0], network);
                self.history.accumulators[target_idx]
                    .black
                    .add_feature(self.pending_adds_b[0], network);
            }
            (1, 1) => {
                self.history.accumulators[target_idx].white.add_1_sub_1(
                    self.pending_adds_w[0],
                    self.pending_dels_w[0],
                    network,
                );
                self.history.accumulators[target_idx].black.add_1_sub_1(
                    self.pending_adds_b[0],
                    self.pending_dels_b[0],
                    network,
                );
            }
            (1, 2) => {
                self.history.accumulators[target_idx].white.add_1_sub_2(
                    self.pending_adds_w[0],
                    self.pending_dels_w[0],
                    self.pending_dels_w[1],
                    network,
                );
                self.history.accumulators[target_idx].black.add_1_sub_2(
                    self.pending_adds_b[0],
                    self.pending_dels_b[0],
                    self.pending_dels_b[1],
                    network,
                );
            }
            (2, 2) => {
                self.history.accumulators[target_idx].white.add_2_sub_2(
                    self.pending_adds_w[0],
                    self.pending_adds_w[1],
                    self.pending_dels_w[0],
                    self.pending_dels_w[1],
                    network,
                );
                self.history.accumulators[target_idx].black.add_2_sub_2(
                    self.pending_adds_b[0],
                    self.pending_adds_b[1],
                    self.pending_dels_b[0],
                    self.pending_dels_b[1],
                    network,
                );
            }
            _ => {
                self.refresh_accumulator(Side::White, network);
                self.refresh_accumulator(Side::Black, network);
            }
        }
        self.history.computed[target_idx] = true;
        self.pending_adds = 0;
        self.pending_removes = 0;

        if sfnn16::maintenance_active() {
            let pos = SfnnPosition::from_board(self);
            let accs = &mut self.history.sfnn16[target_idx];
            sfnn16::flush_pending(&pos, accs, &mut self.sfnn16_pending);
            self.history.sfnn16_computed[target_idx] = [true, true];
        }
    }

    pub fn record_pending_updates(&mut self, target_idx: usize) {
        self.history.dirty_updates[target_idx] = crate::board::history::DirtyUpdate {
            adds_w: self.pending_adds_w,
            dels_w: self.pending_dels_w,
            adds_b: self.pending_adds_b,
            dels_b: self.pending_dels_b,
            n_adds: self.pending_adds,
            n_dels: self.pending_removes,
        };
        self.history.computed[target_idx] = false;
        self.pending_adds = 0;
        self.pending_removes = 0;

        if sfnn16::maintenance_active() {
            self.history.sfnn16_pending[target_idx] = self.sfnn16_pending;
            self.history.sfnn16_computed[target_idx] = [false; 2];
            self.sfnn16_pending.clear();
        }
    }

    pub fn ensure_accumulators_fresh(&mut self) {
        let target_idx = self.history.index;
        if self.history.computed[target_idx] {
            return;
        }

        let mut ancestor = target_idx;
        while ancestor > 0 && !self.history.computed[ancestor] {
            ancestor -= 1;
        }

        let network = Network::get_embedded();
        for idx in (ancestor + 1)..=target_idx {
            self.history.accumulators[idx] = self.history.accumulators[idx - 1];
            let dirty = self.history.dirty_updates[idx];
            Self::apply_dirty(&mut self.history.accumulators[idx], &dirty, network);
            self.history.computed[idx] = true;
        }
    }

    fn apply_dirty(
        acc: &mut crate::eval::nnue::accumulator::Accumulators,
        dirty: &crate::board::history::DirtyUpdate,
        network: &Network,
    ) {
        match (dirty.n_adds, dirty.n_dels) {
            (0, 0) => {}
            (1, 0) => {
                acc.white.add_feature(dirty.adds_w[0], network);
                acc.black.add_feature(dirty.adds_b[0], network);
            }
            (1, 1) => {
                acc.white
                    .add_1_sub_1(dirty.adds_w[0], dirty.dels_w[0], network);
                acc.black
                    .add_1_sub_1(dirty.adds_b[0], dirty.dels_b[0], network);
            }
            (1, 2) => {
                acc.white
                    .add_1_sub_2(dirty.adds_w[0], dirty.dels_w[0], dirty.dels_w[1], network);
                acc.black
                    .add_1_sub_2(dirty.adds_b[0], dirty.dels_b[0], dirty.dels_b[1], network);
            }
            (2, 2) => {
                acc.white.add_2_sub_2(
                    dirty.adds_w[0],
                    dirty.adds_w[1],
                    dirty.dels_w[0],
                    dirty.dels_w[1],
                    network,
                );
                acc.black.add_2_sub_2(
                    dirty.adds_b[0],
                    dirty.adds_b[1],
                    dirty.dels_b[0],
                    dirty.dels_b[1],
                    network,
                );
            }
            _ => {}
        }
    }

    pub fn refresh_accumulator(&mut self, side: Side, network: &Network) {
        let mut acc = Accumulator::new();
        acc.init_with_biases(network);

        for &p_side in &[Side::White, Side::Black] {
            let relative_side = if p_side == side {
                Side::White
            } else {
                Side::Black
            };

            for &piece in &Piece::ALL {
                if piece == Piece::None {
                    continue;
                }
                let mut pieces_bb = self.get_pieces(p_side, piece);
                while pieces_bb.is_not_empty() {
                    let sq_raw = pieces_bb.get_lsb() as usize;
                    pieces_bb.clear_lsb();

                    let sq = Square::from(sq_raw);
                    let sq = if side == Side::White {
                        sq
                    } else {
                        sq.mirrored()
                    };

                    if let Some(feature_idx) = get_feature_index(piece, relative_side, sq) {
                        acc.add_feature(feature_idx, network);
                    }
                }
            }
        }

        if side == Side::White {
            self.history.accumulators[self.history.index].white = acc;
        } else {
            self.history.accumulators[self.history.index].black = acc;
        }
        self.history.computed[self.history.index] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;
    use crate::common::moves::Move;

    fn startpos() -> BoardState {
        BoardState::parse_fen(STARTING_FEN)
    }

    fn flush_shape(adds: &[Square], removes: &[Square]) -> BoardState {
        let mut board = startpos();
        for &sq in adds {
            board.add_piece(sq, Side::White, Piece::Pawn, true);
        }
        for &sq in removes {
            board.remove_piece(sq, true);
        }
        board.flush_pending_updates(0);
        board
    }

    #[test]
    fn flush_each_pending_shape_marks_computed() {
        use Square::*;

        let board = flush_shape(&[], &[]);
        assert!(board.history.computed[0]);

        let board = flush_shape(&[E4], &[]);
        assert!(board.history.computed[0]);
        assert_eq!(board.pending_adds, 0);

        let board = flush_shape(&[E4], &[D2]);
        assert!(board.history.computed[0]);
        assert_eq!((board.pending_adds, board.pending_removes), (0, 0));

        let board = flush_shape(&[E4], &[D2, E2]);
        assert!(board.history.computed[0]);

        let board = flush_shape(&[E4, E5], &[D2, E2]);
        assert!(board.history.computed[0]);

        let board = flush_shape(&[], &[D2]);
        assert!(board.history.computed[0]);
        let board = flush_shape(&[E4, E5], &[D2]);
        assert!(board.history.computed[0]);
    }

    #[test]
    fn record_then_ensure_replays_dirty_updates() {
        let mut board = startpos();
        let m = Move::parse_long_algebraic("e2e4").unwrap();
        board.make_move(m);
        assert_eq!(board.history.index, 1);
        assert!(!board.history.computed[1]);
        board.ensure_accumulators_fresh();
        assert!(board.history.computed[1]);

        let mut clean = startpos();
        assert!(clean.history.computed[clean.history.index]);
        clean.ensure_accumulators_fresh();
        assert!(clean.history.computed[clean.history.index]);
    }

    #[test]
    fn ensure_replays_each_dirty_shape() {
        use Square::*;

        let shapes: &[(&[Square], &[Square])] = &[
            (&[], &[]),
            (&[E4], &[]),
            (&[E4], &[D2]),
            (&[E4], &[D2, E2]),
            (&[E4, E5], &[D2, E2]),
            (&[], &[D2]),
        ];
        for (adds, removes) in shapes {
            let mut board = startpos();
            for &sq in *adds {
                board.add_piece(sq, Side::White, Piece::Pawn, true);
            }
            for &sq in *removes {
                board.remove_piece(sq, true);
            }
            board.record_pending_updates(1);
            assert!(!board.history.computed[1]);
            board.history.index = 1;
            board.ensure_accumulators_fresh();
            assert!(
                board.history.computed[1],
                "shape ({}, {}) did not replay",
                adds.len(),
                removes.len()
            );
        }
    }

    #[test]
    fn refresh_rebuilds_both_perspectives() {
        let network = Network::get_embedded();
        let mut board = startpos();
        board.refresh_accumulator(Side::White, network);
        board.refresh_accumulator(Side::Black, network);
        assert!(board.history.computed[board.history.index]);
    }
}
