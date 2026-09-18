use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Castle(pub u8);

impl Castle {
    pub const NONE: Self = Self(0);
    pub const WHITE_SHORT: Self = Self(1);
    pub const WHITE_LONG: Self = Self(2);
    pub const BLACK_SHORT: Self = Self(4);
    pub const BLACK_LONG: Self = Self(8);

    #[inline]
    pub const fn bits(&self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn from_bits_retain(bits: u8) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn contains(&self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl BitOr for Castle {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Castle {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for Castle {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Castle {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_castle_bitflags() {
        let mut castle = Castle::NONE;
        assert_eq!(castle.bits(), 0);

        castle |= Castle::WHITE_SHORT;
        assert!(castle.contains(Castle::WHITE_SHORT));
        assert!(!castle.contains(Castle::WHITE_LONG));

        castle |= Castle::BLACK_LONG;
        assert!(castle.contains(Castle::WHITE_SHORT));
        assert!(castle.contains(Castle::BLACK_LONG));
        assert_eq!(castle.bits(), 1 | 8);

        castle.remove(Castle::WHITE_SHORT);
        assert!(!castle.contains(Castle::WHITE_SHORT));
        assert!(castle.contains(Castle::BLACK_LONG));
    }

    #[test]
    fn test_castle_from_bits_retain_roundtrip() {
        assert_eq!(Castle::from_bits_retain(0), Castle::NONE);
        assert_eq!(Castle::from_bits_retain(1), Castle::WHITE_SHORT);
        assert_eq!(Castle::from_bits_retain(2), Castle::WHITE_LONG);
        assert_eq!(Castle::from_bits_retain(4), Castle::BLACK_SHORT);
        assert_eq!(Castle::from_bits_retain(8), Castle::BLACK_LONG);
        let all = Castle::from_bits_retain(0b1111);
        assert_eq!(all.bits(), 0b1111);
        assert!(all.contains(Castle::WHITE_SHORT));
        assert!(all.contains(Castle::WHITE_LONG));
        assert!(all.contains(Castle::BLACK_SHORT));
        assert!(all.contains(Castle::BLACK_LONG));
        // Retains unknown bits verbatim.
        assert_eq!(Castle::from_bits_retain(0b1111_0000).bits(), 0b1111_0000);
    }

    #[test]
    fn test_castle_bitor_creates_new_value() {
        let combined = Castle::WHITE_SHORT | Castle::WHITE_LONG;
        assert_eq!(combined.bits(), 1 | 2);
        assert!(combined.contains(Castle::WHITE_SHORT));
        assert!(combined.contains(Castle::WHITE_LONG));
        assert!(!combined.contains(Castle::BLACK_SHORT));
        // Originals unchanged (Copy semantics).
        assert_eq!(Castle::WHITE_SHORT.bits(), 1);
    }

    #[test]
    fn test_castle_bitand_filters_bits() {
        let all = Castle::WHITE_SHORT | Castle::WHITE_LONG | Castle::BLACK_SHORT;
        let masked = all & Castle::WHITE_LONG;
        assert_eq!(masked, Castle::WHITE_LONG);
        assert_eq!((all & Castle::BLACK_LONG).bits(), 0);
        assert_eq!((Castle::NONE & Castle::WHITE_SHORT).bits(), 0);
    }

    #[test]
    fn test_castle_bitand_assign_narrows_value() {
        let mut castle =
            Castle::WHITE_SHORT | Castle::WHITE_LONG | Castle::BLACK_SHORT | Castle::BLACK_LONG;
        castle &= Castle::WHITE_SHORT | Castle::BLACK_SHORT;
        assert_eq!(castle.bits(), 1 | 4);
        assert!(castle.contains(Castle::WHITE_SHORT));
        assert!(!castle.contains(Castle::WHITE_LONG));
    }

    #[test]
    fn test_castle_default_is_none_and_contains_none() {
        assert_eq!(Castle::default(), Castle::NONE);
        assert_eq!(Castle::default().bits(), 0);
        // `contains(NONE)` is vacuously true for any value.
        assert!(Castle::NONE.contains(Castle::NONE));
        assert!(Castle::WHITE_SHORT.contains(Castle::NONE));
    }

    #[test]
    fn test_castle_remove_absent_flag_is_noop() {
        let mut castle = Castle::WHITE_SHORT;
        castle.remove(Castle::BLACK_SHORT);
        assert_eq!(castle, Castle::WHITE_SHORT);
        castle.remove(Castle::WHITE_SHORT | Castle::WHITE_LONG);
        assert_eq!(castle, Castle::NONE);
        // Removing from empty stays empty.
        castle.remove(Castle::BLACK_LONG);
        assert_eq!(castle, Castle::NONE);
    }
}
