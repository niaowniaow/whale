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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn stop_clears_ponder_and_cancels_search() {
        let mut client = UciClient::new();
        let cancel = Arc::new(AtomicBool::new(false));
        client.current_search = Some(Arc::clone(&cancel));
        client.is_pondering.store(true, Ordering::Relaxed);
        client.run_stop(&[]);
        assert!(!client.is_pondering.load(Ordering::Relaxed));
        assert!(client.precompute_cancel.load(Ordering::Relaxed));
        assert!(cancel.load(Ordering::Relaxed));
    }

    #[test]
    fn stop_without_running_search_is_noop() {
        let mut client = UciClient::new();
        client.run_stop(&[]);
        assert!(!client.is_pondering.load(Ordering::Relaxed));
        assert!(client.precompute_cancel.load(Ordering::Relaxed));
        assert!(client.current_search.is_none());
    }
}
