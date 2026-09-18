use crate::common::move_type::MoveType;
use crate::common::piece::Piece;
use crate::common::square::Square;
use std::hash::{Hash, Hasher};

// TODO: optimize memory here, can save a lot of TT space
// ref for ideas https://github.com/codedeliveryservice/Reckless/blob/main/src/types/moves.rs
#[derive(Debug, Clone, Copy)]
pub struct Move {
    pub source: Square,
    pub target: Square,
    pub move_type: MoveType,
}

impl Move {
    pub const NO_MOVE: Move = Move {
        source: Square::NoSquare,
        target: Square::NoSquare,
        move_type: MoveType::Quiet,
    };

    pub fn new(source: Square, target: Square, move_type: MoveType) -> Self {
        Self {
            source,
            target,
            move_type,
        }
    }

    pub fn is_capture(&self) -> bool {
        self.move_type.is_capture()
    }

    pub fn promotion_char(&self) -> Option<char> {
        self.move_type.promotion_char()
    }

    pub fn is_promotion(&self) -> bool {
        self.move_type.promotion_piece() != Piece::None
    }

    pub fn is_castle(&self) -> bool {
        self.move_type == MoveType::Castle
    }

    pub fn parse_long_algebraic(move_string: &str) -> Option<Self> {
        if move_string.len() < 4 || move_string.len() > 5 {
            return None;
        }

        let parse_square = |s: &str| -> Option<Square> {
            let mut chars = s.chars();
            let f = chars.next()?.to_ascii_lowercase();
            let r = chars.next()?;

            if !('a'..='h').contains(&f) || !('1'..='8').contains(&r) {
                return None;
            }

            let file = (f as u8) - b'a';
            let rank = (r as u8) - b'1';

            let sq_idx = (7 - rank) * 8 + file;
            Some(Square::from(sq_idx as usize))
        };

        let source = parse_square(&move_string[0..2])?;
        let target = parse_square(&move_string[2..4])?;

        let move_type = if move_string.len() == 5 {
            match move_string.chars().nth(4)?.to_ascii_lowercase() {
                'q' => MoveType::QueenPromotion,
                'r' => MoveType::RookPromotion,
                'b' => MoveType::BishopPromotion,
                'n' => MoveType::KnightPromotion,
                _ => return None,
            }
        } else {
            MoveType::Quiet
        };

        Some(Self::new(source, target, move_type))
    }
}

impl PartialEq for Move {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
            && self.target == other.target
            && self.move_type == other.move_type
    }
}

impl Eq for Move {}

impl Hash for Move {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.target.hash(state);
        self.move_type.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_equality() {
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::E2, Square::E4, MoveType::Quiet);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_parse_long_algebraic() {
        let m = Move::parse_long_algebraic("e2e4").unwrap();
        assert_eq!(m.source, Square::E2);
        assert_eq!(m.target, Square::E4);
        assert_eq!(m.move_type, MoveType::Quiet);

        let m_prom = Move::parse_long_algebraic("e7e8q").unwrap();
        assert_eq!(m_prom.source, Square::E7);
        assert_eq!(m_prom.target, Square::E8);
        assert_eq!(m_prom.move_type, MoveType::QueenPromotion);
    }

    #[test]
    fn test_properties() {
        let m_capture = Move::new(Square::E4, Square::D5, MoveType::Capture);
        assert!(m_capture.is_capture());
        assert!(!m_capture.is_promotion());
        assert!(!m_capture.is_castle());
        assert_eq!(m_capture.promotion_char(), None);

        let m_castle = Move::new(Square::E1, Square::G1, MoveType::Castle);
        assert!(!m_castle.is_capture());
        assert!(!m_castle.is_promotion());
        assert!(m_castle.is_castle());
        assert_eq!(m_castle.promotion_char(), None);

        let m_prom_cap = Move::new(Square::E7, Square::D8, MoveType::KnightPromotionCapture);
        assert!(m_prom_cap.is_capture());
        assert!(m_prom_cap.is_promotion());
        assert!(!m_prom_cap.is_castle());
        assert_eq!(m_prom_cap.promotion_char(), Some('n'));
    }

    #[test]
    fn test_equal_moves_should_be_equal() {
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_moves_with_different_sources_should_not_be_equal() {
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::D2, Square::E4, MoveType::Quiet);
        assert_ne!(m1, m2);
    }

    #[test]
    fn test_moves_with_different_targets_should_not_be_equal() {
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::E2, Square::E3, MoveType::Quiet);
        assert_ne!(m1, m2);
    }

    #[test]
    fn test_moves_with_different_types_should_not_be_equal() {
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::E2, Square::E4, MoveType::Capture);
        assert_ne!(m1, m2);
    }

    #[test]
    fn test_no_move_should_equal_itself() {
        let m1 = Move::NO_MOVE;
        let m2 = Move::NO_MOVE;
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_no_move_fields() {
        assert_eq!(Move::NO_MOVE.source, Square::NoSquare);
        assert_eq!(Move::NO_MOVE.target, Square::NoSquare);
        assert_eq!(Move::NO_MOVE.move_type, MoveType::Quiet);
        assert!(!Move::NO_MOVE.is_capture());
        assert!(!Move::NO_MOVE.is_promotion());
        assert!(!Move::NO_MOVE.is_castle());
    }

    #[test]
    fn test_parse_rejects_bad_lengths() {
        assert_eq!(Move::parse_long_algebraic(""), None);
        assert_eq!(Move::parse_long_algebraic("e2"), None);
        assert_eq!(Move::parse_long_algebraic("e2e"), None);
        assert_eq!(Move::parse_long_algebraic("e2e44q"), None);
        assert_eq!(Move::parse_long_algebraic("e2e4qr"), None);
    }

    #[test]
    fn test_parse_rejects_bad_squares() {
        assert_eq!(Move::parse_long_algebraic("i2e4"), None);
        assert_eq!(Move::parse_long_algebraic("e9e4"), None);
        assert_eq!(Move::parse_long_algebraic("e2i4"), None);
        assert_eq!(Move::parse_long_algebraic("e2e9"), None);
        assert_eq!(Move::parse_long_algebraic("11e4"), None);
    }

    #[test]
    fn test_parse_all_promotion_chars() {
        for (suffix, expected) in [
            ('q', MoveType::QueenPromotion),
            ('r', MoveType::RookPromotion),
            ('b', MoveType::BishopPromotion),
            ('n', MoveType::KnightPromotion),
        ] {
            let s = format!("e7e8{suffix}");
            let m = Move::parse_long_algebraic(&s).unwrap();
            assert_eq!(m.move_type, expected, "suffix {suffix}");
            assert!(m.is_promotion());
            assert!(!m.is_capture());
        }
    }

    #[test]
    fn test_parse_promotion_uppercase_suffix() {
        let m = Move::parse_long_algebraic("e7e8Q").unwrap();
        assert_eq!(m.move_type, MoveType::QueenPromotion);
        let m = Move::parse_long_algebraic("e7e8N").unwrap();
        assert_eq!(m.move_type, MoveType::KnightPromotion);
    }

    #[test]
    fn test_parse_rejects_invalid_promotion_char() {
        assert_eq!(Move::parse_long_algebraic("e7e8x"), None);
        assert_eq!(Move::parse_long_algebraic("e7e8k"), None);
    }

    #[test]
    fn test_parse_uppercase_squares() {
        let m = Move::parse_long_algebraic("E2E4").unwrap();
        assert_eq!(m.source, Square::E2);
        assert_eq!(m.target, Square::E4);
        assert_eq!(m.move_type, MoveType::Quiet);
    }

    #[test]
    fn test_is_promotion_covers_all_promotion_types() {
        for mt in [
            MoveType::KnightPromotion,
            MoveType::BishopPromotion,
            MoveType::RookPromotion,
            MoveType::QueenPromotion,
            MoveType::KnightPromotionCapture,
            MoveType::BishopPromotionCapture,
            MoveType::RookPromotionCapture,
            MoveType::QueenPromotionCapture,
        ] {
            let m = Move::new(Square::E7, Square::E8, mt);
            assert!(m.is_promotion(), "{mt:?} should be promotion");
        }
        for mt in [
            MoveType::Quiet,
            MoveType::Capture,
            MoveType::EnPassant,
            MoveType::DoublePush,
            MoveType::Castle,
        ] {
            let m = Move::new(Square::E2, Square::E4, mt);
            assert!(!m.is_promotion(), "{mt:?} should not be promotion");
        }
    }

    #[test]
    fn test_promotion_char_delegates_to_move_type() {
        assert_eq!(
            Move::new(Square::E7, Square::E8, MoveType::QueenPromotion).promotion_char(),
            Some('q')
        );
        assert_eq!(
            Move::new(Square::E7, Square::E8, MoveType::RookPromotion).promotion_char(),
            Some('r')
        );
        assert_eq!(
            Move::new(Square::E7, Square::F8, MoveType::BishopPromotionCapture).promotion_char(),
            Some('b')
        );
        assert_eq!(
            Move::new(Square::E2, Square::E4, MoveType::Quiet).promotion_char(),
            None
        );
    }

    #[test]
    fn test_move_hash_matches_fields() {
        use std::collections::hash_map::DefaultHasher;
        fn hash_of(m: &Move) -> u64 {
            let mut h = DefaultHasher::new();
            m.hash(&mut h);
            h.finish()
        }
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        assert_eq!(hash_of(&m1), hash_of(&m2));
        let m3 = Move::new(Square::E2, Square::E4, MoveType::Capture);
        assert_ne!(hash_of(&m1), hash_of(&m3));
    }

    #[test]
    fn test_move_clone_copy_and_debug() {
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let cloned = m;
        assert_eq!(m, cloned);
        let debug = format!("{m:?}");
        assert!(debug.contains("E2"));
        assert!(debug.contains("E4"));
    }
}
