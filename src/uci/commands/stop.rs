use crate::uci::UciClient;
use std::sync::atomic::Ordering;

impl UciClient {
    pub(crate) fn run_stop(&mut self, _parameters: &[&str]) {
        if let Some(cancel) = &self.current_search {
            cancel.store(true, Ordering::Relaxed);
        }
    }
}
