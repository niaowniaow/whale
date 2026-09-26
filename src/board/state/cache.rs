use super::BoardState;
use crate::board::history::StateCache;

impl BoardState {
    pub fn update_node_cache(&mut self) {
        let stm = self.side_to_move;
        let them = stm.other();
        let checkers = self.checkers(stm).0;
        let pinned = self.pinned_pieces(stm).0;
        let pinned_them = self.pinned_pieces(them).0;
        let pinners = self.compute_pinners(stm).0;
        let all_threats = self.compute_all_threats(them).0;
        let check_squares = self.check_squares(them);
        let threats_us = self.threat_by_lesser(stm);
        let threats_them = self.threat_by_lesser(them);
        self.history.store_cache(StateCache {
            checkers,
            pinned,
            pinners,
            pinned_them,
            all_threats,
            check_squares,
            threats_us,
            threats_them,
        });
    }

    pub fn refresh_cache(&mut self) {
        self.update_node_cache();
    }

    pub fn ensure_cache_fresh(&mut self) {
        if !self.history.is_cache_valid() {
            self.update_node_cache();
        }
    }

    pub fn cached_checkers(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().checkers;
        }
        self.checkers(self.side_to_move).0
    }

    pub fn cached_pinned(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().pinned;
        }
        self.pinned_pieces(self.side_to_move).0
    }

    pub fn cached_pinners(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().pinners;
        }
        self.compute_pinners(self.side_to_move).0
    }

    pub fn cached_all_threats(&self) -> u64 {
        if self.history.is_cache_valid() {
            return self.history.current_cache().all_threats;
        }
        self.compute_all_threats(self.side_to_move.other()).0
    }

    pub fn cached_check_squares(&self) -> [u64; 6] {
        if self.history.is_cache_valid() {
            return self.history.current_cache().check_squares;
        }
        self.check_squares(self.side_to_move.other())
    }

    pub fn cached_threats_us(&self) -> [u64; 6] {
        if self.history.is_cache_valid() {
            return self.history.current_cache().threats_us;
        }
        self.threat_by_lesser(self.side_to_move)
    }

    pub fn cached_threats_them(&self) -> [u64; 6] {
        if self.history.is_cache_valid() {
            return self.history.current_cache().threats_them;
        }
        self.threat_by_lesser(self.side_to_move.other())
    }
}
