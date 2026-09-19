use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::square::Square;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

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
#[derive(Default)]
pub struct Cluster {
    pub entries: [AtomicEntry; CLUSTER_SIZE],
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    cluster_count: usize,
    pub capacity: usize,
    pub generation: AtomicU8,
}

// SAFETY: `TranspositionTable` is shared across Lazy SMP search threads via
// `Arc<TranspositionTable>`. Following Stockfish (tt.cpp) and Reckless
// (transposition.rs), all entry access is lock-free with benign races:
// key/data are separate atomics, so a torn read can at worst return a stale
// but well-formed entry (same pack/unpack as a locked read). `clusters` is
// never resized while shared — `resize` takes `&mut self`. `prefetch` is a
// side-effect-free CPU hint.
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

    /// Permille (0..1000) of occupied slots, Stockfish `hashfull` concept.
    /// Lock-free best-effort read; torn entries simply count or not.
    pub fn hashfull(&self) -> u32 {
        let mut used = 0u64;
        // Sample up to 1024 clusters to bound cost on huge tables.
        let step = (self.cluster_count / 1024).max(1);
        let mut sampled = 0u64;
        let mut i = 0;
        while i < self.cluster_count {
            for entry in &self.clusters[i].entries {
                let k = entry.key.load(Ordering::Relaxed);
                let d = entry.data.load(Ordering::Relaxed);
                if k != 0 && unpack_entry(k, d).entry_type != TranspositionEntryType::None {
                    used += 1;
                }
            }
            sampled += CLUSTER_SIZE as u64;
            i += step;
        }
        if sampled == 0 {
            return 0;
        }
        ((used * 1000) / sampled) as u32
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
        {
            // SAFETY: `index` is masked to `cluster_count - 1` (a power-of-two),
            // so it is always in bounds of `self.clusters`. The resulting pointer
            // targets a valid, allocated `Cluster`. `_mm_prefetch` with `_MM_HINT_T0`
            // is a no-op hint to the CPU cache hierarchy and never traps, even if
            // the address were invalid (it would simply be ignored).
            unsafe {
                use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                let index = (hash as usize) & (self.cluster_count - 1);
                let ptr = self.clusters.as_ptr().add(index) as *const i8;
                _mm_prefetch(ptr, _MM_HINT_T0);
            }
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
        halfmove: u8,
    ) -> (bool, i16, Option<Move>) {
        let entry = match self.probe(hash) {
            Some(e) => e,
            None => return (false, 0, None),
        };

        if entry.depth < depth {
            return (false, 0, Some(entry.best_move));
        }

        let tt_score = Self::retrieve_score(entry.score, ply as i32, halfmove);

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
            let k = entry.key.load(Ordering::Acquire);
            if k == hash {
                let d = entry.data.load(Ordering::Acquire);
                if entry.key.load(Ordering::Acquire) != hash {
                    continue;
                }
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
                        entry.key.store(0, Ordering::Relaxed);
                        entry.data.store(packed, Ordering::Release);
                        entry.key.store(hash, Ordering::Release);
                    }
                    return;
                }
            }
        }

        let mut replace_idx = 0;
        let mut lowest_priority = i32::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            let k = entry.key.load(Ordering::Acquire);
            let d = entry.data.load(Ordering::Acquire);
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
        target.key.store(0, Ordering::Relaxed);
        target.data.store(packed, Ordering::Release);
        target.key.store(hash, Ordering::Release);
    }

    // FIX MATE-DOWNGRADE-50: mirror Reckless transposition.rs:315-351.
    // Write path only folds ply into decisive scores; read path additionally
    // downgrades mate/TB scores that the 50-move rule may invalidate.
    // Thresholds are derived from our own MAX_CENTIPAWN_EVAL/MAX_PLY so they
    // stay consistent with syzygy::TB_WIN.
    const fn mate_score() -> i32 {
        MAX_CENTIPAWN_EVAL as i32
    }
    const fn mate_in_max() -> i32 {
        MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32
    }
    const fn tb_win() -> i32 {
        MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32 - 10
    }
    const fn tb_win_in_max() -> i32 {
        MAX_CENTIPAWN_EVAL as i32 - MAX_PLY as i32 - 10 - MAX_PLY as i32
    }
    #[inline(always)]
    fn is_win_score(score: i32) -> bool {
        score >= Self::tb_win_in_max()
    }
    #[inline(always)]
    fn is_loss_score(score: i32) -> bool {
        score <= -Self::tb_win_in_max()
    }

    pub fn adjust_score(score: i16, ply: i32, _halfmove: u8) -> i16 {
        let s = score as i32;
        if Self::is_win_score(s) {
            return (s + ply) as i16;
        }
        if Self::is_loss_score(s) {
            return (s - ply) as i16;
        }
        score
    }

    pub fn retrieve_score(score: i16, ply: i32, halfmove: u8) -> i16 {
        let s = score as i32;
        let hm = halfmove as i32;
        if s == 0 {
            return score;
        }
        if Self::is_win_score(s) {
            // Downgrade a potentially false mate score.
            if s >= Self::mate_in_max() && Self::mate_score() - s > 100 - hm {
                return (Self::tb_win_in_max() - 1) as i16;
            }
            // Downgrade a potentially false TB score.
            if Self::tb_win() - s > 100 - hm {
                return (Self::tb_win_in_max() - 1) as i16;
            }
            return (s - ply) as i16;
        }
        if Self::is_loss_score(s) {
            if s <= -Self::mate_in_max() && Self::mate_score() + s > 100 - hm {
                return (-Self::tb_win_in_max() + 1) as i16;
            }
            if Self::tb_win() + s > 100 - hm {
                return (-Self::tb_win_in_max() + 1) as i16;
            }
            return (s + ply) as i16;
        }
        score
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
    fn test_tt_concurrent_colliding_entries() {
        // Lock-free semantics (Stockfish/Reckless): under contention an entry
        // may belong to a colliding hash, but its packed data must always be
        // internally consistent (score/depth/type come from one atomic pack).
        let tt = TranspositionTable::new(1024);
        std::thread::scope(|scope| {
            for worker in 0..8u64 {
                let tt = &tt;
                scope.spawn(move || {
                    for iteration in 0..20_000u64 {
                        let id = (iteration + worker) % 8;
                        let hash = id * tt.cluster_count as u64;
                        tt.submit_entry(
                            hash,
                            id as i16 * 101,
                            id as u8 + 1,
                            Move::NO_MOVE,
                            TranspositionEntryType::Exact,
                        );
                        for probe_id in 0..8u64 {
                            if let Some(entry) = tt.probe(probe_id * tt.cluster_count as u64) {
                                assert_eq!(entry.entry_type, TranspositionEntryType::Exact);
                                let got = entry.score / 101;
                                assert!((0..8).contains(&got));
                                assert_eq!(entry.depth, got as u8 + 1);
                            }
                        }
                    }
                });
            }
        });
    }

    #[test]
    fn test_tt_store_retrieve() {
        let tt = TranspositionTable::new(1024);
        let hash = 123456789;
        let best_move = Move::new(Square::E2, Square::E4, MoveType::Quiet);

        tt.submit_entry(hash, 100, 5, best_move, TranspositionEntryType::Exact);

        let (found, score, m) = tt.get_entry(hash, -1000, 1000, 5, 0, 0);
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

        let (_, score, m) = tt.get_entry(hash, -1000, 1000, 1, 0, 0);
        assert_eq!(score, 100);
        assert_eq!(m, Some(m1));

        tt.submit_entry(hash, 300, 10, m2, TranspositionEntryType::Exact);
        let (_, score, m) = tt.get_entry(hash, -1000, 1000, 1, 0, 0);
        assert_eq!(score, 300);
        assert_eq!(m, Some(m2));
    }

    #[test]
    fn test_tt_score_adjustment() {
        let mate_score = MAX_CENTIPAWN_EVAL - 5;
        let adjusted = TranspositionTable::adjust_score(mate_score, 10, 0);
        assert_eq!(adjusted, mate_score + 10);

        let retrieved = TranspositionTable::retrieve_score(adjusted, 10, 0);
        assert_eq!(retrieved, mate_score);
    }

    #[test]
    fn test_tt_mate_downgrade_near_fifty() {
        // Mate score that the 50-move rule may invalidate must downgrade.
        let mate_score = MAX_CENTIPAWN_EVAL - 5;
        let stored = TranspositionTable::adjust_score(mate_score, 0, 0);
        // halfmove=95: MATE - score(5) <= 100-95(5)? 5 > 5 false, no downgrade at boundary;
        // halfmove=96: 5 > 4 true -> downgrade.
        let ok = TranspositionTable::retrieve_score(stored, 0, 0);
        assert_eq!(ok, mate_score);
        let downgraded = TranspositionTable::retrieve_score(stored, 0, 96);
        assert!(downgraded < mate_score);
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

    #[test]
    fn test_entry_type_default_is_none() {
        assert_eq!(
            TranspositionEntryType::default(),
            TranspositionEntryType::None
        );
    }

    #[test]
    fn test_entry_default_fields() {
        let e = TranspositionTableEntry::default();
        assert_eq!(e.hash, 0);
        assert_eq!(e.score, 0);
        assert_eq!(e.best_move, Move::NO_MOVE);
        assert_eq!(e.depth, 0);
        assert_eq!(e.entry_type, TranspositionEntryType::None);
        assert_eq!(e.generation, 0);
    }

    #[test]
    fn test_atomic_entry_default_is_zero() {
        let e = AtomicEntry::default();
        assert_eq!(e.key.load(std::sync::atomic::Ordering::Relaxed), 0);
        assert_eq!(e.data.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn test_tt_new_small_capacity_still_usable() {
        let tt = TranspositionTable::new(10);
        assert_eq!(tt.capacity(), 10);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(999, 42, 3, m, TranspositionEntryType::Exact);
        let (found, score, got) = tt.get_entry(999, -1000, 1000, 3, 0, 0);
        assert!(found);
        assert_eq!(score, 42);
        assert_eq!(got, Some(m));
    }

    #[test]
    fn test_tt_new_non_pow2_capacity_usable() {
        let tt = TranspositionTable::new(1000);
        assert_eq!(tt.capacity(), 1000);
        let m = Move::new(Square::D2, Square::D4, MoveType::DoublePush);
        tt.submit_entry(555, 77, 2, m, TranspositionEntryType::Exact);
        assert_eq!(tt.probe(555).map(|e| e.score), Some(77));
    }

    #[test]
    fn test_tt_new_mb_tiny_clamps_to_minimum() {
        let tt = TranspositionTable::new_mb(0);
        assert_eq!(tt.capacity(), 512);
        assert_eq!(tt.capacity, 512);
    }

    #[test]
    fn test_tt_default_capacity_constant() {
        assert_eq!(TranspositionTable::DEFAULT_CAPACITY, 65536 * 32);
    }

    #[test]
    fn test_tt_new_search_bumps_generation() {
        let tt = TranspositionTable::new(1024);
        assert_eq!(tt.generation.load(std::sync::atomic::Ordering::Relaxed), 0);
        tt.new_search();
        assert_eq!(tt.generation.load(std::sync::atomic::Ordering::Relaxed), 1);
        tt.new_search();
        assert_eq!(tt.generation.load(std::sync::atomic::Ordering::Relaxed), 2);
    }

    #[test]
    fn test_tt_probe_miss_returns_none() {
        let tt = TranspositionTable::new(1024);
        assert_eq!(tt.probe(0xDEAD_BEEF), None);
        assert_eq!(tt.probe(1), None);
    }

    #[test]
    fn test_tt_prefetch_does_not_panic() {
        let tt = TranspositionTable::new(1024);
        tt.prefetch(123456789);
        tt.prefetch(0);
        tt.prefetch(u64::MAX);
    }

    #[test]
    fn test_tt_clear_removes_entries() {
        let tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(424242, 100, 5, m, TranspositionEntryType::Exact);
        assert!(tt.probe(424242).is_some());
        tt.clear();
        assert_eq!(tt.probe(424242), None);
        let (found, _, _) = tt.get_entry(424242, -1000, 1000, 1, 0, 0);
        assert!(!found);
    }

    #[test]
    fn test_tt_pack_unpack_roundtrip_all_move_types() {
        let tt = TranspositionTable::new(1024);
        let cases = [
            Move::new(Square::E2, Square::E4, MoveType::Quiet),
            Move::new(Square::E4, Square::D5, MoveType::Capture),
            Move::new(Square::E5, Square::D6, MoveType::EnPassant),
            Move::new(Square::D2, Square::D4, MoveType::DoublePush),
            Move::new(Square::E7, Square::E8, MoveType::KnightPromotion),
            Move::new(Square::E7, Square::E8, MoveType::BishopPromotion),
            Move::new(Square::E7, Square::E8, MoveType::RookPromotion),
            Move::new(Square::E7, Square::E8, MoveType::QueenPromotion),
            Move::new(Square::E7, Square::D8, MoveType::KnightPromotionCapture),
            Move::new(Square::E7, Square::D8, MoveType::BishopPromotionCapture),
            Move::new(Square::E7, Square::D8, MoveType::RookPromotionCapture),
            Move::new(Square::E7, Square::D8, MoveType::QueenPromotionCapture),
            Move::new(Square::E1, Square::G1, MoveType::Castle),
        ];
        for (i, m) in cases.iter().enumerate() {
            let hash = 1_000_000 + i as u64 * 1_000_003;
            tt.submit_entry(hash, 123, 4, *m, TranspositionEntryType::Exact);
            let entry = tt.probe(hash).expect("entry must be present");
            assert_eq!(entry.hash, hash);
            assert_eq!(entry.score, 123);
            assert_eq!(entry.depth, 4);
            assert_eq!(entry.entry_type, TranspositionEntryType::Exact);
            assert_eq!(entry.best_move, *m);
        }
    }

    #[test]
    fn test_tt_pack_unpack_no_move_and_negative_score() {
        let tt = TranspositionTable::new(1024);
        tt.submit_entry(
            777001,
            -321,
            0,
            Move::NO_MOVE,
            TranspositionEntryType::Exact,
        );
        let entry = tt.probe(777001).unwrap();
        assert_eq!(entry.best_move, Move::NO_MOVE);
        assert_eq!(entry.score, -321);
        assert_eq!(entry.depth, 0);

        tt.submit_entry(
            777002,
            i16::MAX,
            u8::MAX,
            Move::new(Square::A8, Square::H1, MoveType::Capture),
            TranspositionEntryType::Beta,
        );
        let entry = tt.probe(777002).unwrap();
        assert_eq!(entry.score, i16::MAX);
        assert_eq!(entry.depth, u8::MAX);
        assert_eq!(entry.entry_type, TranspositionEntryType::Beta);

        tt.submit_entry(
            777003,
            i16::MIN,
            1,
            Move::new(Square::A1, Square::A2, MoveType::Quiet),
            TranspositionEntryType::Alpha,
        );
        assert_eq!(tt.probe(777003).unwrap().score, i16::MIN);
    }

    #[test]
    fn test_tt_submit_records_generation() {
        let tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(31337, 10, 1, m, TranspositionEntryType::Exact);
        assert_eq!(tt.probe(31337).unwrap().generation, 0);
        tt.new_search();
        tt.submit_entry(31338, 10, 1, m, TranspositionEntryType::Exact);
        assert_eq!(tt.probe(31338).unwrap().generation, 1);
    }

    #[test]
    fn test_tt_none_entries_are_ignored() {
        let tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(600600, 50, 3, m, TranspositionEntryType::None);
        assert_eq!(tt.probe(600600), None);
        let (found, _, _) = tt.get_entry(600600, -1000, 1000, 1, 0, 0);
        assert!(!found);
    }

    #[test]
    fn test_tt_get_entry_miss() {
        let tt = TranspositionTable::new(1024);
        let (found, score, m) = tt.get_entry(0xABCD, -1000, 1000, 1, 0, 0);
        assert!(!found);
        assert_eq!(score, 0);
        assert_eq!(m, None);
    }

    #[test]
    fn test_tt_get_entry_shallow_depth_returns_move_without_cutoff() {
        let tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(700700, 100, 5, m, TranspositionEntryType::Exact);
        let (found, score, got) = tt.get_entry(700700, -1000, 1000, 6, 0, 0);
        assert!(!found);
        assert_eq!(score, 0);
        assert_eq!(got, Some(m));
    }

    #[test]
    fn test_tt_get_entry_alpha_cutoff_taken_and_not_taken() {
        let tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(800801, 50, 5, m, TranspositionEntryType::Alpha);
        let (found, score, got) = tt.get_entry(800801, 100, 200, 5, 0, 0);
        assert!(found);
        assert_eq!(score, 50);
        assert_eq!(got, Some(m));

        let tt2 = TranspositionTable::new(1024);
        tt2.submit_entry(800802, 150, 5, m, TranspositionEntryType::Alpha);
        let (found, score, got) = tt2.get_entry(800802, 100, 200, 5, 0, 0);
        assert!(!found);
        assert_eq!(score, 0);
        assert_eq!(got, Some(m));
    }

    #[test]
    fn test_tt_get_entry_beta_cutoff_taken_and_not_taken() {
        let tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(900901, 250, 5, m, TranspositionEntryType::Beta);
        let (found, score, got) = tt.get_entry(900901, 100, 200, 5, 0, 0);
        assert!(found);
        assert_eq!(score, 250);
        assert_eq!(got, Some(m));

        let tt2 = TranspositionTable::new(1024);
        tt2.submit_entry(900902, 150, 5, m, TranspositionEntryType::Beta);
        let (found, score, got) = tt2.get_entry(900902, 100, 200, 5, 0, 0);
        assert!(!found);
        assert_eq!(score, 0);
        assert_eq!(got, Some(m));
    }

    #[test]
    fn test_tt_adjust_score_loss_and_normal() {
        // Loss path subtracts ply.
        let loss = -(MAX_CENTIPAWN_EVAL - 5);
        assert_eq!(TranspositionTable::adjust_score(loss, 10, 0), loss - 10);
        // Normal scores pass through untouched.
        assert_eq!(TranspositionTable::adjust_score(100, 10, 0), 100);
        assert_eq!(TranspositionTable::adjust_score(-100, 10, 0), -100);
        assert_eq!(TranspositionTable::adjust_score(0, 10, 0), 0);
        // Just below the win threshold stays normal.
        assert_eq!(TranspositionTable::adjust_score(30861, 10, 0), 30861);
        assert_eq!(TranspositionTable::adjust_score(-30861, 10, 0), -30861);
    }

    #[test]
    fn test_tt_retrieve_score_zero_and_normal() {
        assert_eq!(TranspositionTable::retrieve_score(0, 5, 0), 0);
        assert_eq!(TranspositionTable::retrieve_score(100, 7, 0), 100);
        assert_eq!(TranspositionTable::retrieve_score(-100, 7, 50), -100);
    }

    #[test]
    fn test_tt_retrieve_score_loss_roundtrip() {
        let mate_against = -(MAX_CENTIPAWN_EVAL - 5);
        let stored = TranspositionTable::adjust_score(mate_against, 10, 0);
        assert_eq!(stored, mate_against - 10);
        let retrieved = TranspositionTable::retrieve_score(stored, 10, 0);
        assert_eq!(retrieved, mate_against);
    }

    #[test]
    fn test_tt_retrieve_score_tb_win_downgrade() {
        // 30870 is a TB win (>= 30862) but below mate-in-max (30936),
        // so only the TB-downgrade branch can fire at high halfmove.
        let tb = 30870_i16;
        assert_eq!(TranspositionTable::retrieve_score(tb, 0, 0), tb);
        let downgraded = TranspositionTable::retrieve_score(tb, 0, 96);
        assert_eq!(downgraded, 30861);
        let downgraded_loss = TranspositionTable::retrieve_score(-tb, 0, 96);
        assert_eq!(downgraded_loss, -30861);
        assert_eq!(TranspositionTable::retrieve_score(-tb, 0, 0), -tb);
    }

    #[test]
    fn test_tt_retrieve_score_mate_loss_downgrade() {
        let mate_against = -(MAX_CENTIPAWN_EVAL - 5);
        let stored = TranspositionTable::adjust_score(mate_against, 0, 0);
        assert_eq!(
            TranspositionTable::retrieve_score(stored, 0, 0),
            mate_against
        );
        let downgraded = TranspositionTable::retrieve_score(stored, 0, 96);
        assert!(downgraded > mate_against);
    }

    #[test]
    fn test_tt_submit_preserves_best_move_when_no_move_given() {
        let tt = TranspositionTable::new(1024);
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let hash = 111222333;
        tt.submit_entry(hash, 100, 5, m1, TranspositionEntryType::Exact);
        tt.submit_entry(hash, 200, 6, Move::NO_MOVE, TranspositionEntryType::Exact);
        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.score, 200);
        assert_eq!(entry.best_move, m1);
    }

    #[test]
    fn test_tt_submit_exact_upgrades_shallower_non_exact() {
        let tt = TranspositionTable::new(1024);
        let m1 = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        let m2 = Move::new(Square::D2, Square::D4, MoveType::Quiet);
        let hash = 444555666;
        tt.submit_entry(hash, 100, 5, m1, TranspositionEntryType::Alpha);
        tt.submit_entry(hash, 200, 3, m2, TranspositionEntryType::Exact);
        let entry = tt.probe(hash).unwrap();
        assert_eq!(entry.score, 200);
        assert_eq!(entry.best_move, m2);
        assert_eq!(entry.entry_type, TranspositionEntryType::Exact);
    }

    #[test]
    fn test_tt_submit_evicts_lowest_depth_in_full_cluster() {
        let tt = TranspositionTable::new(1024);
        // Stride is a multiple of any small power-of-two cluster count,
        // so all five hashes land in the same cluster.
        const STRIDE: u64 = 4096;
        let moves = [
            Move::new(Square::A2, Square::A3, MoveType::Quiet),
            Move::new(Square::B2, Square::B3, MoveType::Quiet),
            Move::new(Square::C2, Square::C3, MoveType::Quiet),
            Move::new(Square::D2, Square::D3, MoveType::Quiet),
        ];
        for (i, m) in moves.iter().enumerate() {
            tt.submit_entry(
                (i as u64 + 1) * STRIDE,
                (i as i16 + 1) * 10,
                i as u8 + 1,
                *m,
                TranspositionEntryType::Exact,
            );
        }
        for (i, m) in moves.iter().enumerate() {
            let entry = tt.probe((i as u64 + 1) * STRIDE).unwrap();
            assert_eq!(entry.best_move, *m);
        }
        // Fifth colliding entry with highest depth evicts the depth-1 entry.
        let newcomer = Move::new(Square::E2, Square::E3, MoveType::Quiet);
        tt.submit_entry(5 * STRIDE, 999, 10, newcomer, TranspositionEntryType::Exact);
        assert_eq!(tt.probe(5 * STRIDE).unwrap().best_move, newcomer);
        assert_eq!(tt.probe(STRIDE), None);
        assert!(tt.probe(2 * STRIDE).is_some());
    }

    #[test]
    fn test_tt_hashfull_tracks_usage() {
        let tt = TranspositionTable::new(1024);
        assert_eq!(tt.hashfull(), 0);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        // One entry in 2048 slots is 0 permille; fill enough to register.
        for i in 0..600u64 {
            tt.submit_entry(424243 + i * 7919, 100, 5, m, TranspositionEntryType::Exact);
        }
        assert!(tt.hashfull() > 0);
        assert!(tt.hashfull() <= 1000);
    }

    #[test]
    fn test_tt_resize_drops_old_entries() {
        let mut tt = TranspositionTable::new(1024);
        let m = Move::new(Square::E2, Square::E4, MoveType::Quiet);
        tt.submit_entry(999888, 100, 5, m, TranspositionEntryType::Exact);
        assert!(tt.probe(999888).is_some());
        tt.resize(1);
        assert_eq!(tt.probe(999888), None);
    }
}
