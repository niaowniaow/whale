use crate::uci::UciClient;

impl UciClient {
    pub(crate) fn run_setoption(&mut self, parameters: &[&str]) {
        let mut name = String::new();
        let mut value = String::new();

        let mut i = 0;
        while i < parameters.len() {
            if parameters[i].eq_ignore_ascii_case("name") && i + 1 < parameters.len() {
                let mut name_parts = Vec::new();
                i += 1;
                while i < parameters.len() && !parameters[i].eq_ignore_ascii_case("value") {
                    name_parts.push(parameters[i]);
                    i += 1;
                }
                name = name_parts.join(" ");
                continue;
            }
            if parameters[i].eq_ignore_ascii_case("value") && i + 1 < parameters.len() {
                value = parameters[i + 1..].join(" ");
                break;
            }
            i += 1;
        }

        if let Ok(mut state) = self.search_state.try_lock() {
            if name.eq_ignore_ascii_case("Clear Hash") {
                state.tt.clear();
                crate::uci::cli::write_line("info string Hash cleared");
            }
            if name.eq_ignore_ascii_case("Hash")
                && let Ok(mb_size) = value.parse::<usize>()
            {
                let mb_size = mb_size.clamp(1, 2048);
                state.tt =
                    std::sync::Arc::new(crate::common::tt::TranspositionTable::new_mb(mb_size));
            }
            if name.eq_ignore_ascii_case("Threads")
                && let Ok(v) = value.parse::<usize>()
            {
                self.num_threads = v.clamp(1, 256);
            }
            if (name.eq_ignore_ascii_case("Move Overhead")
                || name.eq_ignore_ascii_case("Move_Overhead")
                || name.eq_ignore_ascii_case("MoveOverhead"))
                && let Ok(v) = value.parse::<i32>()
            {
                self.move_overhead = v.clamp(0, 5000);
            }
            if name.eq_ignore_ascii_case("Ponder") {
                self.ponder_enabled = value.eq_ignore_ascii_case("true") || value == "1";
            }
            if name.eq_ignore_ascii_case("SyzygyProbeLimit")
                && let Ok(v) = value.parse::<u8>()
            {
                crate::syzygy::set_probe_limit(v.min(7));
            }
            if name.eq_ignore_ascii_case("SyzygyProbeDepth")
                && let Ok(v) = value.parse::<u8>()
            {
                crate::syzygy::set_probe_depth(v.max(1));
            }
            if name.eq_ignore_ascii_case("Syzygy50MoveRule") {
                crate::syzygy::set_50mr_rule(
                    !(value.eq_ignore_ascii_case("false") || value == "0"),
                );
            }
            if name.eq_ignore_ascii_case("RFP_Margin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.rfp_margin_mult = v.clamp(50, 300);
            }
            if name.eq_ignore_ascii_case("Futility_Margin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.futility_margin_mult = v.clamp(50, 300);
            }
            if name.eq_ignore_ascii_case("Singular_Margin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.singular_margin_mult = v.clamp(1, 5);
            }
            if name.eq_ignore_ascii_case("ProbCut_Margin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.probcut_margin = v.clamp(50, 400);
            }
            if name.eq_ignore_ascii_case("NMP_Base")
                && let Ok(v) = value.parse::<u8>()
            {
                state.params.nmp_base = v.clamp(1, 6);
            }
            if name.eq_ignore_ascii_case("NMP_Depth_Div")
                && let Ok(v) = value.parse::<u8>()
            {
                state.params.nmp_depth_div = v.clamp(2, 8);
            }
            if name.eq_ignore_ascii_case("LMR_Base")
                && let Ok(v) = value.parse::<f64>()
            {
                state.params.lmr_base = (v / 100.0).clamp(0.10, 1.50);
                state.lmr_table =
                    crate::search::lmr::LmrTable::new(state.params.lmr_base, state.params.lmr_div);
            }
            if name.eq_ignore_ascii_case("LMR_Div")
                && let Ok(v) = value.parse::<f64>()
            {
                state.params.lmr_div = (v / 100.0).clamp(1.0, 3.50);
                state.lmr_table =
                    crate::search::lmr::LmrTable::new(state.params.lmr_base, state.params.lmr_div);
            }
            if name.eq_ignore_ascii_case("History_Weight")
                && let Ok(v) = value.parse::<i32>()
            {
                state.params.history_weight_mult = v.clamp(1, 4);
            }
            if name.eq_ignore_ascii_case("ALP_Enabled") {
                state.params.alp_enabled = value.eq_ignore_ascii_case("true");
            }
            if name.eq_ignore_ascii_case("ALP_Threshold")
                && let Ok(v) = value.parse::<u8>()
            {
                state.params.alp_threshold = v.clamp(50, 95);
            }
            if name.eq_ignore_ascii_case("PSM_Enabled") {
                state.params.psm_enabled = value.eq_ignore_ascii_case("true");
            }
            if name.eq_ignore_ascii_case("GTP_Enabled") {
                state.params.gtp_enabled = value.eq_ignore_ascii_case("true");
            }
            if name.eq_ignore_ascii_case("GTP_Threshold")
                && let Ok(v) = value.parse::<u8>()
            {
                state.params.gtp_threshold = v.clamp(5, 50);
            }
        } else {
            // The search thread holds the search_state lock for its whole run,
            // so a failed try_lock means a search is in flight. Stop it and
            // say so instead of silently dropping the option (Stockfish
            // uci.cpp:483-486 stops the search before applying options); the
            // GUI can resend the option once the search ends.
            crate::uci::cli::write_line(
                "info string setoption ignored while searching (stop the search and resend)",
            );
            if let Some(cancel) = &self.current_search {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            self.precompute_cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        if name.eq_ignore_ascii_case("EvalFile") || name.eq_ignore_ascii_case("EvalFileSmall") {
            match crate::eval::nnue::v16::set_eval_file(&name, &value) {
                Ok(msg) => crate::uci::cli::write_line(&format!("info string {msg}")),
                Err(e) => crate::uci::cli::write_line(&format!("info string EvalFile error: {e}")),
            }
        }

        if name.eq_ignore_ascii_case("SyzygyPath") {
            match crate::syzygy::set_path(&value) {
                Ok(n) => {
                    crate::uci::cli::write_line(&format!("info string Syzygy tables loaded: {n}"))
                }
                Err(e) => crate::uci::cli::write_line(&format!("info string Syzygy error: {e}")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_resize_transposition_table() {
        let mut uci_client = UciClient::new();

        {
            let state = uci_client.search_state.lock().unwrap();
            assert_eq!(state.tt.capacity(), 524288);
        }

        uci_client.run_setoption(&["name", "Hash", "value", "128"]);

        {
            let state = uci_client.search_state.lock().unwrap();
            assert_eq!(state.tt.capacity(), 4194304); // 128MB
        }

        uci_client.run_setoption(&["name", "hash", "value", "1"]);

        {
            let state = uci_client.search_state.lock().unwrap();
            assert_eq!(state.tt.capacity(), 32768);
        }
    }

    #[test]
    fn should_handle_eval_file_option() {
        let mut uci_client = UciClient::new();
        uci_client.run_setoption(&["name", "EvalFile", "value", "non_existent.nnue"]);
    }

    #[test]
    fn should_update_search_parameters() {
        let mut uci_client = UciClient::new();
        uci_client.run_setoption(&["name", "RFP_Margin", "value", "180"]);
        uci_client.run_setoption(&["name", "LMR_Base", "value", "60"]);
        uci_client.run_setoption(&["name", "LMR_Div", "value", "220"]);
        let state = uci_client.search_state.lock().unwrap();
        assert_eq!(state.params.rfp_margin_mult, 180);
        assert!((state.params.lmr_base - 0.60).abs() < 1e-5);
        assert!((state.params.lmr_div - 2.20).abs() < 1e-5);
    }

    #[test]
    fn should_apply_threads_overhead_and_ponder() {
        let mut client = UciClient::new();
        client.run_setoption(&["name", "Threads", "value", "8"]);
        assert_eq!(client.num_threads, 8);
        client.run_setoption(&["name", "Threads", "value", "9999"]);
        assert_eq!(client.num_threads, 256);
        client.run_setoption(&["name", "Threads", "value", "bogus"]);
        assert_eq!(client.num_threads, 256);
        client.run_setoption(&["name", "Move Overhead", "value", "100"]);
        assert_eq!(client.move_overhead, 100);
        client.run_setoption(&["name", "MoveOverhead", "value", "99999"]);
        assert_eq!(client.move_overhead, 5000);
        client.run_setoption(&["name", "Ponder", "value", "true"]);
        assert!(client.ponder_enabled);
        client.run_setoption(&["name", "Ponder", "value", "0"]);
        assert!(!client.ponder_enabled);
    }

    #[test]
    fn should_clear_hash_and_clamp_margins() {
        let mut client = UciClient::new();
        client.run_setoption(&["name", "Clear", "Hash"]);
        client.run_setoption(&["name", "Futility_Margin", "value", "1"]);
        client.run_setoption(&["name", "Singular_Margin", "value", "99"]);
        client.run_setoption(&["name", "ProbCut_Margin", "value", "9"]);
        client.run_setoption(&["name", "NMP_Base", "value", "0"]);
        client.run_setoption(&["name", "NMP_Depth_Div", "value", "99"]);
        client.run_setoption(&["name", "History_Weight", "value", "99"]);
        let state = client.search_state.lock().unwrap();
        assert_eq!(state.params.futility_margin_mult, 50);
        assert_eq!(state.params.singular_margin_mult, 5);
        assert_eq!(state.params.probcut_margin, 50);
        assert_eq!(state.params.nmp_base, 1);
        assert_eq!(state.params.nmp_depth_div, 8);
        assert_eq!(state.params.history_weight_mult, 4);
    }

    #[test]
    fn should_toggle_feature_flags_and_thresholds() {
        let mut client = UciClient::new();
        for name in ["ALP_Enabled", "PSM_Enabled", "GTP_Enabled"] {
            client.run_setoption(&["name", name, "value", "true"]);
        }
        client.run_setoption(&["name", "ALP_Threshold", "value", "1"]);
        client.run_setoption(&["name", "GTP_Threshold", "value", "99"]);
        {
            let state = client.search_state.lock().unwrap();
            assert!(state.params.alp_enabled);
            assert!(state.params.psm_enabled);
            assert!(state.params.gtp_enabled);
            assert_eq!(state.params.alp_threshold, 50);
            assert_eq!(state.params.gtp_threshold, 50);
        }
        client.run_setoption(&["name", "ALP_Enabled", "value", "false"]);
        assert!(!client.search_state.lock().unwrap().params.alp_enabled);
    }

    #[test]
    fn should_ignore_malformed_options() {
        let mut client = UciClient::new();
        client.run_setoption(&[]);
        client.run_setoption(&["name"]);
        client.run_setoption(&["name", "NoSuchOption", "value", "1"]);
        assert_eq!(client.num_threads, 1);
        assert_eq!(client.move_overhead, 10);
    }

    #[test]
    fn should_forward_syzygy_options_without_loading() {
        let _serial = crate::syzygy::SYZYGY_TEST_LOCK.lock().unwrap();
        let mut client = UciClient::new();
        client.run_setoption(&["name", "SyzygyProbeLimit", "value", "3"]);
        assert_eq!(crate::syzygy::probe_limit(), 3);
        client.run_setoption(&["name", "SyzygyProbeDepth", "value", "2"]);
        assert_eq!(crate::syzygy::probe_depth(), 2);
        client.run_setoption(&["name", "Syzygy50MoveRule", "value", "false"]);
        assert!(!crate::syzygy::use_50mr());
        client.run_setoption(&["name", "SyzygyProbeLimit", "value", "bogus"]);
        assert_eq!(crate::syzygy::probe_limit(), 3);
        client.run_setoption(&["name", "SyzygyProbeLimit", "value", "7"]);
        client.run_setoption(&["name", "SyzygyProbeDepth", "value", "1"]);
        client.run_setoption(&["name", "Syzygy50MoveRule", "value", "true"]);
        client.run_setoption(&["name", "SyzygyPath", "value", "<empty>"]);
        assert_eq!(crate::syzygy::table_count(), 0);
        client.run_setoption(&["name", "SyzygyPath", "value", "definitely/missing"]);
        assert_eq!(crate::syzygy::table_count(), 0);
    }

    #[test]
    fn should_defer_setoption_while_searching() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let mut client = UciClient::new();
        let state = Arc::clone(&client.search_state);
        let _guard = state.lock().unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        client.current_search = Some(Arc::clone(&cancel));
        client.run_setoption(&["name", "Threads", "value", "4"]);
        assert_eq!(client.num_threads, 1);
        assert!(cancel.load(Ordering::Relaxed));
        assert!(client.precompute_cancel.load(Ordering::Relaxed));

        let mut idle = UciClient::new();
        let idle_state = Arc::clone(&idle.search_state);
        let _idle_guard = idle_state.lock().unwrap();
        idle.run_setoption(&["name", "Threads", "value", "4"]);
        assert_eq!(idle.num_threads, 1);
        assert!(idle.precompute_cancel.load(Ordering::Relaxed));
    }
}
