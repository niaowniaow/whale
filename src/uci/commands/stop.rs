use crate::uci::UciClient;
use std::sync::atomic::Ordering;

impl UciClient {
    pub(crate) fn run_stop(&mut self, _parameters: &[&str]) {
        self.is_pondering.store(false, Ordering::Relaxed);
        self.precompute_cancel.store(true, Ordering::Relaxed);
        if let Some(cancel) = &self.current_search {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}
