use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PieceMap<T>(pub [T; 6]);

impl<T> PieceMap<T> {
    #[inline(always)]
    pub const fn new(init: T) -> Self
    where
        T: Copy,
    {
        Self([init; 6])
    }
}

impl<T> Index<Piece> for PieceMap<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: Piece) -> &Self::Output {
        match index {
            Piece::None => panic!("Cannot index PieceMap with Piece::None"),
            _ => &self.0[index as usize],
        }
    }
}

impl<T> IndexMut<Piece> for PieceMap<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: Piece) -> &mut Self::Output {
        match index {
            Piece::None => panic!("Cannot index PieceMap with Piece::None"),
            _ => &mut self.0[index as usize],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Piece {
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    None,
}

impl Piece {
    pub const ALL_PIECES: usize = 6;

    pub const ALL: [Piece; 6] = [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ];
}

impl From<usize> for Piece {
    fn from(value: usize) -> Self {
        match value {
            0 => Piece::Pawn,
            1 => Piece::Knight,
            2 => Piece::Bishop,
            3 => Piece::Rook,
            4 => Piece::Queen,
            5 => Piece::King,
            6 => Piece::None,
            _ => panic!("Invalid piece index: {}", value),
        }
    }
}

impl From<Piece> for usize {
    fn from(p: Piece) -> Self {
        p as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piece_from_usize() {
        assert_eq!(Piece::from(0), Piece::Pawn);
        assert_eq!(Piece::from(5), Piece::King);
        assert_eq!(Piece::from(6), Piece::None);
    }

    #[test]
    #[should_panic(expected = "Invalid piece index: 7")]
    fn test_piece_from_usize_panics_on_invalid() {
        let _ = Piece::from(7);
    }

    #[test]
    fn test_piece_into_usize() {
        assert_eq!(usize::from(Piece::Pawn), 0);
        assert_eq!(usize::from(Piece::King), 5);
        assert_eq!(usize::from(Piece::None), 6);
    }

    #[test]
    fn test_piece_all_constants() {
        assert_eq!(Piece::ALL_PIECES, 6);
        assert_eq!(Piece::ALL.len(), 6);
        assert_eq!(
            Piece::ALL,
            [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King
            ]
        );
        for (i, p) in Piece::ALL.iter().enumerate() {
            assert_eq!(Piece::from(i), *p);
            assert_eq!(usize::from(*p), i);
        }
    }

    #[test]
    fn test_piece_from_usize_all_valid() {
        assert_eq!(Piece::from(1), Piece::Knight);
        assert_eq!(Piece::from(2), Piece::Bishop);
        assert_eq!(Piece::from(3), Piece::Rook);
        assert_eq!(Piece::from(4), Piece::Queen);
    }

    #[test]
    fn test_piece_map_new_and_index() {
        let map = PieceMap::new(0_i32);
        for p in Piece::ALL {
            assert_eq!(map[p], 0);
        }
        let map = PieceMap::new(7_i32);
        assert_eq!(map[Piece::Pawn], 7);
        assert_eq!(map[Piece::King], 7);
    }

    #[test]
    fn test_piece_map_index_mut() {
        let mut map = PieceMap::new(0_i32);
        map[Piece::Pawn] = 1;
        map[Piece::Queen] = 9;
        assert_eq!(map[Piece::Pawn], 1);
        assert_eq!(map[Piece::Queen], 9);
        assert_eq!(map[Piece::Knight], 0);
    }

    #[test]
    #[should_panic(expected = "Cannot index PieceMap with Piece::None")]
    fn test_piece_map_index_panics_on_none() {
        let map = PieceMap::new(0_i32);
        let _ = map[Piece::None];
    }

    #[test]
    #[should_panic(expected = "Cannot index PieceMap with Piece::None")]
    fn test_piece_map_index_mut_panics_on_none() {
        let mut map = PieceMap::new(0_i32);
        map[Piece::None] = 1;
    }
}
