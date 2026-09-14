use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::moves::Move;

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

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cluster {
    pub entries: [TranspositionTableEntry; CLUSTER_SIZE],
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            entries: [TranspositionTableEntry::default(); CLUSTER_SIZE],
        }
    }
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    cluster_count: usize,
    pub capacity: usize,
    pub generation: u8,
}

impl TranspositionTable {
    pub const DEFAULT_CAPACITY: usize = 65536 * 32;

    pub fn new(capacity: usize) -> Self {
        let mut cluster_count = (capacity / 2).max(256);
        if !cluster_count.is_power_of_two() {
            cluster_count = 1 << cluster_count.ilog2();
        }
        Self {
            clusters: vec![Cluster::default(); cluster_count],
            cluster_count,
            capacity,
            generation: 0,
        }
    }

    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
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
        self.clusters = vec![Cluster::default(); cluster_count];
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        for cluster in self.clusters.iter_mut() {
            *cluster = Cluster::default();
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64) -> Option<TranspositionTableEntry> {
        let index = (hash as usize) & (self.cluster_count - 1);
        let cluster = &self.clusters[index];
        for entry in &cluster.entries {
            if entry.hash == hash && entry.entry_type != TranspositionEntryType::None {
                return Some(*entry);
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
        &mut self,
        hash: u64,
        score: i16,
        depth: u8,
        mut best_move: Move,
        entry_type: TranspositionEntryType,
    ) {
        let index = (hash as usize) & (self.cluster_count - 1);
        let cluster = &mut self.clusters[index];

        for entry in cluster.entries.iter_mut() {
            if entry.hash == hash && entry.entry_type != TranspositionEntryType::None {
                if best_move == Move::NO_MOVE {
                    best_move = entry.best_move;
                }
                let is_old = entry.generation != self.generation;
                let should_replace = is_old
                    || depth >= entry.depth
                    || (entry_type == TranspositionEntryType::Exact
                        && entry.entry_type != TranspositionEntryType::Exact);

                if should_replace {
                    entry.hash = hash;
                    entry.score = score;
                    entry.depth = depth;
                    entry.best_move = best_move;
                    entry.entry_type = entry_type;
                    entry.generation = self.generation;
                } else if entry.best_move == Move::NO_MOVE && best_move != Move::NO_MOVE {
                    entry.best_move = best_move;
                }
                return;
            }
        }

        let mut replace_idx = 0;
        let mut lowest_priority = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.entry_type == TranspositionEntryType::None {
                replace_idx = i;
                break;
            }
            let age = self.generation.wrapping_sub(entry.generation) as i32;
            let priority = entry.depth as i32 - 8 * age;
            if priority < lowest_priority {
                lowest_priority = priority;
                replace_idx = i;
            }
        }

        cluster.entries[replace_idx] = TranspositionTableEntry {
            hash,
            score,
            best_move,
            depth,
            entry_type,
            generation: self.generation,
        };
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
        let mut tt = TranspositionTable::new(1024);
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
        let mut tt = TranspositionTable::new(1024);
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
