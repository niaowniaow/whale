use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::square::Square;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum TranspositionEntryType {
    #[default]
    None = 0,
    Exact = 1,
    Alpha = 2,
    Beta = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranspositionTableEntry {
    pub hash: u64,
    pub score: i16,
    pub best_move: Move,
    pub depth: u8,
    pub entry_type: TranspositionEntryType,
    pub generation: u8,
}

impl Default for TranspositionTableEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            score: 0,
            best_move: Move::NO_MOVE,
            depth: 0,
            entry_type: TranspositionEntryType::None,
            generation: 0,
        }
    }
}

pub const CLUSTER_SIZE: usize = 4;

#[repr(C)]
pub struct AtomicEntry {
    pub key: AtomicU64,
    pub data: AtomicU64,
}

impl Default for AtomicEntry {
    fn default() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

#[inline(always)]
fn pack_entry(
    score: i16,
    depth: u8,
    entry_type: TranspositionEntryType,
    generation: u8,
    m: Move,
) -> u64 {
    let s = (score as u16) as u64;
    let d = (depth as u64) << 16;
    let et = (entry_type as u8 as u64) << 24;
    let generation_bits = (generation as u64) << 32;
    let src = (m.source as u8 as u64) << 40;
    let tgt = (m.target as u8 as u64) << 48;
    let mt = (m.move_type.value() as u64) << 56;
    s | d | et | generation_bits | src | tgt | mt
}

#[inline(always)]
fn unpack_entry(hash: u64, data: u64) -> TranspositionTableEntry {
    let score = (data & 0xFFFF) as u16 as i16;
    let depth = ((data >> 16) & 0xFF) as u8;
    let entry_type = match ((data >> 24) & 0xFF) as u8 {
        1 => TranspositionEntryType::Exact,
        2 => TranspositionEntryType::Alpha,
        3 => TranspositionEntryType::Beta,
        _ => TranspositionEntryType::None,
    };
    let generation = ((data >> 32) & 0xFF) as u8;
    let src_byte = ((data >> 40) & 0xFF) as u8;
    let tgt_byte = ((data >> 48) & 0xFF) as u8;
    let mt_byte = ((data >> 56) & 0xFF) as u8;
    let source = if (src_byte as usize) <= 64 {
        Square::from(src_byte as usize)
    } else {
        Square::NoSquare
    };
    let target = if (tgt_byte as usize) <= 64 {
        Square::from(tgt_byte as usize)
    } else {
        Square::NoSquare
    };
    let move_type = match mt_byte {
        0 => MoveType::Quiet,
        1 => MoveType::Capture,
        2 => MoveType::EnPassant,
        3 => MoveType::DoublePush,
        4 => MoveType::KnightPromotion,
        5 => MoveType::BishopPromotion,
        6 => MoveType::RookPromotion,
        7 => MoveType::QueenPromotion,
        12 => MoveType::KnightPromotionCapture,
        13 => MoveType::BishopPromotionCapture,
        14 => MoveType::RookPromotionCapture,
        15 => MoveType::QueenPromotionCapture,
        16 => MoveType::Castle,
        _ => MoveType::Quiet,
    };
    let best_move = if source == Square::NoSquare && target == Square::NoSquare {
        Move::NO_MOVE
    } else {
        Move::new(source, target, move_type)
    };
    TranspositionTableEntry {
        hash,
        score,
        best_move,
        depth,
        entry_type,
        generation,
    }
}

#[repr(C, align(64))]
pub struct Cluster {
    pub entries: [AtomicEntry; CLUSTER_SIZE],
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            entries: [
                AtomicEntry::default(),
                AtomicEntry::default(),
                AtomicEntry::default(),
                AtomicEntry::default(),
            ],
        }
    }
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    cluster_count: usize,
    pub capacity: usize,
    pub generation: AtomicU8,
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub const DEFAULT_CAPACITY: usize = 65536 * 32;

    pub fn new(capacity: usize) -> Self {
        let mut cluster_count = (capacity / 2).max(256);
        if !cluster_count.is_power_of_two() {
            cluster_count = 1 << cluster_count.ilog2();
        }
        let clusters = (0..cluster_count).map(|_| Cluster::default()).collect();
        Self {
            clusters,
            cluster_count,
            capacity,
            generation: AtomicU8::new(0),
        }
    }

    pub fn new_mb(mb_size: usize) -> Self {
        let max_bytes = mb_size * 1024 * 1024;
        let cluster_size = size_of::<Cluster>();
        let max_clusters = max_bytes / cluster_size;
        let cluster_count = if max_clusters < 256 {
            256
        } else {
            1 << max_clusters.ilog2()
        };
        let clusters = (0..cluster_count).map(|_| Cluster::default()).collect();
        Self {
            clusters,
            cluster_count,
            capacity: cluster_count * 2,
            generation: AtomicU8::new(0),
        }
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn resize(&mut self, mb_size: usize) {
        let max_bytes = mb_size * 1024 * 1024;
        let cluster_size = size_of::<Cluster>();
        let max_clusters = max_bytes / cluster_size;

        let cluster_count = if max_clusters < 256 {
            256
        } else {
            1 << max_clusters.ilog2()
        };

        self.cluster_count = cluster_count;
        self.capacity = cluster_count * 2;
        self.clusters = (0..cluster_count).map(|_| Cluster::default()).collect();
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&self) {
        for cluster in &self.clusters {
            for entry in &cluster.entries {
                entry.key.store(0, Ordering::Relaxed);
                entry.data.store(0, Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let index = (hash as usize) & (self.cluster_count - 1);
            let ptr = self.clusters.as_ptr().add(index) as *const i8;
            _mm_prefetch(ptr, _MM_HINT_T0);
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64) -> Option<TranspositionTableEntry> {
        let index = (hash as usize) & (self.cluster_count - 1);
        let cluster = &self.clusters[index];
        for entry in &cluster.entries {
            let k = entry.key.load(Ordering::Relaxed);
            if k == hash {
                let d = entry.data.load(Ordering::Relaxed);
                let unpacked = unpack_entry(k, d);
                if unpacked.entry_type != TranspositionEntryType::None {
                    return Some(unpacked);
                }
            }
        }
        None
    }

    pub fn get_entry(
        &self,
        hash: u64,
        alpha: i16,
        beta: i16,
        depth: u8,
        ply: u8,
    ) -> (bool, i16, Option<Move>) {
        let entry = match self.probe(hash) {
            Some(e) => e,
            None => return (false, 0, None),
        };

        if entry.depth < depth {
            return (false, 0, Some(entry.best_move));
        }

        let tt_score = Self::retrieve_score(entry.score, ply as i32);

        match entry.entry_type {
            TranspositionEntryType::Exact => (true, tt_score, Some(entry.best_move)),
            TranspositionEntryType::Alpha => {
                if tt_score <= alpha {
                    (true, tt_score, Some(entry.best_move))
                } else {
                    (false, 0, Some(entry.best_move))
                }
            }
            TranspositionEntryType::Beta => {
                if tt_score >= beta {
                    (true, tt_score, Some(entry.best_move))
                } else {
                    (false, 0, Some(entry.best_move))
                }
            }
            TranspositionEntryType::None => (false, 0, None),
        }
    }

    pub fn submit_entry(
        &self,
        hash: u64,
        score: i16,
        depth: u8,
        mut best_move: Move,
        entry_type: TranspositionEntryType,
    ) {
        let index = (hash as usize) & (self.cluster_count - 1);
        let cluster = &self.clusters[index];
        let cur_gen = self.generation.load(Ordering::Relaxed);

        for entry in &cluster.entries {
            let k = entry.key.load(Ordering::Relaxed);
            if k == hash {
                let d = entry.data.load(Ordering::Relaxed);
                let existing = unpack_entry(k, d);
                if existing.entry_type != TranspositionEntryType::None {
                    if best_move == Move::NO_MOVE {
                        best_move = existing.best_move;
                    }
                    let is_old = existing.generation != cur_gen;
                    let should_replace = is_old
                        || depth >= existing.depth
                        || (entry_type == TranspositionEntryType::Exact
                            && existing.entry_type != TranspositionEntryType::Exact);

                    if should_replace {
                        let packed = pack_entry(score, depth, entry_type, cur_gen, best_move);
                        entry.data.store(packed, Ordering::Relaxed);
                        entry.key.store(hash, Ordering::Relaxed);
                    } else if existing.best_move == Move::NO_MOVE && best_move != Move::NO_MOVE {
                        let packed = pack_entry(
                            existing.score,
                            existing.depth,
                            existing.entry_type,
                            existing.generation,
                            best_move,
                        );
                        entry.data.store(packed, Ordering::Relaxed);
                        entry.key.store(hash, Ordering::Relaxed);
                    }
                    return;
                }
            }
        }

        let mut replace_idx = 0;
        let mut lowest_priority = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            let k = entry.key.load(Ordering::Relaxed);
            let d = entry.data.load(Ordering::Relaxed);
            let existing = unpack_entry(k, d);
            if existing.entry_type == TranspositionEntryType::None {
                replace_idx = i;
                break;
            }
            let age = cur_gen.wrapping_sub(existing.generation) as i32;
            let priority = existing.depth as i32 - 8 * age;
            if priority < lowest_priority {
                lowest_priority = priority;
                replace_idx = i;
            }
        }

        let packed = pack_entry(score, depth, entry_type, cur_gen, best_move);
        let target = &cluster.entries[replace_idx];
        target.data.store(packed, Ordering::Relaxed);
        target.key.store(hash, Ordering::Relaxed);
    }

    pub fn adjust_score(score: i16, ply: i32) -> i16 {
        if !Self::is_close_to_checkmate(score) {
            return score;
        }
        score + if score > 0 { ply as i16 } else { -ply as i16 }
    }

    pub fn retrieve_score(score: i16, ply: i32) -> i16 {
        if !Self::is_close_to_checkmate(score) {
            return score;
        }
        score + if score > 0 { -ply as i16 } else { ply as i16 }
    }

    fn is_close_to_checkmate(score: i16) -> bool {
        (MAX_CENTIPAWN_EVAL as i32 - (score as i32).abs()) <= MAX_PLY as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::move_type::MoveType;
    use crate::common::square::Square;

    #[test]
    fn test_cluster_size_and_alignment() {
        assert_eq!(size_of::<TranspositionTableEntry>(), 16);
        assert_eq!(size_of::<Cluster>(), 64);
        assert_eq!(align_of::<Cluster>(), 64);
    }

    #[test]
    fn test_tt_store_retrieve() {
        let tt = TranspositionTable::new(1024);
        let hash = 123456789;
        let best_move = Move::new(Square::E2, Square::E4, MoveType::Quiet);

        tt.submit_entry(hash, 100, 5, best_move, TranspositionEntryType::Exact);

        let (found, score, m) = tt.get_entry(hash, -1000, 1000, 5, 0);
        assert!(found);
        assert_eq!(score, 100);
        assert_eq!(m, Some(best_move));
    }

    #[test]
    fn test_tt_depth_priority() {
        let tt = TranspositionTable::new(1024);
        let hash = 123456789;
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::D2, Square::D4, MoveType::Quiet);

        tt.submit_entry(hash, 100, 5, m1, TranspositionEntryType::Exact);
        tt.submit_entry(hash, 200, 3, m2, TranspositionEntryType::Exact);

        let (_, score, m) = tt.get_entry(hash, -1000, 1000, 1, 0);
        assert_eq!(score, 100);
        assert_eq!(m, Some(m1));

        tt.submit_entry(hash, 300, 10, m2, TranspositionEntryType::Exact);
        let (_, score, m) = tt.get_entry(hash, -1000, 1000, 1, 0);
        assert_eq!(score, 300);
        assert_eq!(m, Some(m2));
    }

    #[test]
    fn test_tt_score_adjustment() {
        let mate_score = MAX_CENTIPAWN_EVAL - 5;
        let adjusted = TranspositionTable::adjust_score(mate_score, 10);
        assert_eq!(adjusted, mate_score + 10);

        let retrieved = TranspositionTable::retrieve_score(adjusted, 10);
        assert_eq!(retrieved, mate_score);
    }

    #[test]
    fn test_tt_resize() {
        let mut tt = TranspositionTable::new(1024);
        assert_eq!(tt.capacity, 1024);

        tt.resize(1);
        assert_eq!(tt.capacity, 32768);

        tt.resize(64);
        assert_eq!(tt.capacity, 2097152);

        tt.resize(128);
        assert_eq!(tt.capacity, 4194304);
    }
}
