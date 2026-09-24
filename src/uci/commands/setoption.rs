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
            if (name.eq_ignore_ascii_case("MaxMoveTime")
                || name.eq_ignore_ascii_case("Max_Move_Time")
                || name.eq_ignore_ascii_case("MaxMove_Time"))
                && let Ok(v) = value.parse::<i32>()
            {
                self.max_move_time = v.max(0);
            }
            if name.eq_ignore_ascii_case("Ponder") {
                self.ponder_enabled = value.eq_ignore_ascii_case("true") || value == "1";
            }
            if name.eq_ignore_ascii_case("DualNet")
                || name.eq_ignore_ascii_case("Dual_Net")
                || name.eq_ignore_ascii_case("Dual Net")
            {
                let enabled = value.eq_ignore_ascii_case("true") || value == "1";
                crate::eval::set_dual_net(enabled);
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
            if name.eq_ignore_ascii_case("LQT_Threshold")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.lqt_threshold = v.clamp(300, 900);
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

            let flag_on = value.eq_ignore_ascii_case("true") || value == "1";
            if name.eq_ignore_ascii_case("CFSS_Enabled") {
                state.params.cfss_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("RAS_Enabled") {
                state.params.ras_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("BMO_Enabled") {
                state.params.bmo_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("TCE_Enabled") {
                state.params.tce_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("LQT_Enabled") {
                state.params.lqt_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("SPS_Enabled") {
                state.params.sps_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("DAD_Enabled") {
                state.params.dad_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("MultiPV")
                && let Ok(v) = value.parse::<usize>()
            {
                state.multipv = v.clamp(1, 8);
            }
            if name.eq_ignore_ascii_case("CPI_Enabled") {
                state.params.cpi_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("State_Enabled") {
                state.params.state_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Risk_Enabled") {
                state.params.risk_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Pressure_Enabled") {
                state.params.pressure_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Attack_Enabled") {
                state.params.attack_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Conversion_Enabled") {
                state.params.conversion_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("QS_Checks_Enabled")
                || name.eq_ignore_ascii_case("QS_Checks")
            {
                state.params.qs_checks_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("LearnedHeads_Enabled")
                || name.eq_ignore_ascii_case("Learned_Enabled")
            {
                state.params.learned_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Razor_Enabled")
                || name.eq_ignore_ascii_case("Razoring_Enabled")
            {
                state.params.razor_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Razor_Margin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.razor_margin = v.clamp(100, 800);
            }
            if name.eq_ignore_ascii_case("IIR_Enabled") {
                state.params.iir_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("NMP_Verify")
                || name.eq_ignore_ascii_case("NMP_Verify_Enabled")
            {
                state.params.nmp_verify_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("NMP_Verify_Margin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.nmp_verify_margin = v.clamp(0, 500);
            }
            if name.eq_ignore_ascii_case("Conspiracy_Enabled") {
                state.params.conspiracy_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("Conspiracy_Tolerance")
                && let Ok(v) = value.parse::<i16>()
            {
                state.params.conspiracy_tolerance = v.clamp(5, 200);
            }
            if name.eq_ignore_ascii_case("SplitRoot_Enabled")
                || name.eq_ignore_ascii_case("Split_At_Root")
            {
                state.params.split_root_enabled = flag_on;
            }
            if name.eq_ignore_ascii_case("UseBook") || name.eq_ignore_ascii_case("Use_Book") {
                state.use_book = flag_on;
            }
            if name.eq_ignore_ascii_case("BookDepth") || name.eq_ignore_ascii_case("Book_Depth")
            {
                if let Ok(v) = value.parse::<u8>() {
                    state.book_depth = v;
                }
            }
            if name.eq_ignore_ascii_case("BookFile")
                || name.eq_ignore_ascii_case("Book_File")
                || name.eq_ignore_ascii_case("BookPath")
            {
                let v = value.trim().to_string();
                if v.is_empty() || v.eq_ignore_ascii_case("<empty>") {
                    state.book_path.clear();
                } else {
                    state.book_path = v;
                }
                state.book_cache.clear();
                state.book_cache_path.clear();
            }

            if name.eq_ignore_ascii_case("MustTryGain")
                && let Ok(v) = value.parse::<i32>()
            {
                state.risk_envelope.min_gain_cp = v.clamp(10, 100);
            }
            if name.eq_ignore_ascii_case("MustTryRisk")
                && let Ok(v) = value.parse::<i32>()
            {
                state.risk_envelope.must_try_max = v.clamp(40, 250);
            }
            if name.eq_ignore_ascii_case("ConvertScore")
                && let Ok(v) = value.parse::<i16>()
            {
                state.conversion_params.convert_min_cp = v.clamp(100, 600);
            }
            if name.eq_ignore_ascii_case("CrushScore")
                && let Ok(v) = value.parse::<i16>()
            {
                state.conversion_params.crush_min_cp = v.clamp(300, 1200);
            }
            if name.eq_ignore_ascii_case("DefendCPI")
                && let Ok(v) = value.parse::<i32>()
            {
                state.state_thresholds.defend_cpi = v.clamp(40, 250);
            }
            if name.eq_ignore_ascii_case("DefendScore")
                && let Ok(v) = value.parse::<i16>()
            {
                state.state_thresholds.defend_score = v.clamp(-400, 0);
            }
            if name.eq_ignore_ascii_case("AttackScore")
                && let Ok(v) = value.parse::<i16>()
            {
                state.state_thresholds.attack_score = v.clamp(0, 300);
            }
            if name.eq_ignore_ascii_case("AttackCPI")
                && let Ok(v) = value.parse::<i32>()
            {
                state.state_thresholds.attack_cpi = v.clamp(10, 250);
            }
            if name.eq_ignore_ascii_case("ConvertCPI")
                && let Ok(v) = value.parse::<i32>()
            {
                state.state_thresholds.convert_cpi = v.clamp(0, 150);
                state.conversion_params.max_opp_cpi = v.clamp(0, 150);
            }
            if name.eq_ignore_ascii_case("ResetDrop")
                && let Ok(v) = value.parse::<i16>()
            {
                state.state_thresholds.reset_drop = v.clamp(10, 150);
            }
            if name.eq_ignore_ascii_case("ConcessionMin")
                && let Ok(v) = value.parse::<i16>()
            {
                state.state_thresholds.concession_min_cp = v.clamp(5, 50);
            }
            if name.eq_ignore_ascii_case("RiskNormal")
                && let Ok(v) = value.parse::<i32>()
            {
                state.risk_envelope.normal_max = v.clamp(0, 150);
            }
            if name.eq_ignore_ascii_case("RiskElevated")
                && let Ok(v) = value.parse::<i32>()
            {
                state.risk_envelope.elevated_max = v.clamp(10, 250);
            }
            if name.eq_ignore_ascii_case("PressureMomDiv")
                && let Ok(v) = value.parse::<i32>()
            {
                state.pressure_weights.momentum_div = v.clamp(1, 8);
            }
            if name.eq_ignore_ascii_case("PressureCpiW")
                && let Ok(v) = value.parse::<i32>()
            {
                state.pressure_weights.cpi_w = v.clamp(0, 4);
            }
            if name.eq_ignore_ascii_case("PressureFreeW")
                && let Ok(v) = value.parse::<i32>()
            {
                state.pressure_weights.freedom_w = v.clamp(0, 4);
            }
            if name.eq_ignore_ascii_case("PressurePlanW")
                && let Ok(v) = value.parse::<i32>()
            {
                state.pressure_weights.plans_w = v.clamp(0, 6);
            }
            if name.eq_ignore_ascii_case("UrgencyMinGain")
                && let Ok(v) = value.parse::<i32>()
            {
                state.urgency_thresholds.min_gain = v.clamp(5, 100);
            }
            if name.eq_ignore_ascii_case("UrgencyMustTryGain")
                && let Ok(v) = value.parse::<i32>()
            {
                state.urgency_thresholds.musttry_gain = v.clamp(10, 200);
            }
            if name.eq_ignore_ascii_case("VerifyTolerance")
                && let Ok(v) = value.parse::<i16>()
            {
                state.verify_tolerance_cp = v.clamp(5, 100);
            }
            if name.eq_ignore_ascii_case("Extension_Cap_Enabled")
                || name.eq_ignore_ascii_case("Extension_Cap")
                || name.eq_ignore_ascii_case("ExtensionCap_Enabled")
                || name.eq_ignore_ascii_case("ExtensionCap")
                || name.eq_ignore_ascii_case("Cap_Enabled")
                || name.eq_ignore_ascii_case("Cap")
            {
                state.params.extension_cap_enabled = flag_on;
            }

            if name.eq_ignore_ascii_case("Contempt")
                && let Ok(v) = value.parse::<i16>()
            {
                state.contempt_cp = v.clamp(-200, 200);
            }
            if name.eq_ignore_ascii_case("DrawScore")
                && let Ok(v) = value.parse::<i16>()
            {
                state.draw_score_cp = v.clamp(-200, 200);
            }
            if name.eq_ignore_ascii_case("ShowWDL") {
                state.show_wdl = !(value.eq_ignore_ascii_case("false") || value == "0");
            }
        } else {
            crate::uci::cli::write_line(
                "info string setoption ignored while searching (stop the search and resend)",
            );
            if let Some(cancel) = &self.current_search {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            self.precompute_cancel
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        if name.eq_ignore_ascii_case("EvalFile")
            || name.eq_ignore_ascii_case("EvalFileSmall")
            || name.eq_ignore_ascii_case("Model")
        {
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
            assert_eq!(state.tt.capacity(), 4194304);
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
    fn should_toggle_experimental_heuristics() {
        let mut client = UciClient::new();
        for name in [
            "CFSS_Enabled",
            "RAS_Enabled",
            "BMO_Enabled",
            "TCE_Enabled",
            "LQT_Enabled",
            "SPS_Enabled",
            "DAD_Enabled",
            "Extension_Cap_Enabled",
            "CPI_Enabled",
            "State_Enabled",
            "Risk_Enabled",
            "Pressure_Enabled",
            "Attack_Enabled",
            "Conversion_Enabled",
            "LearnedHeads_Enabled",
        ] {
            client.run_setoption(&["name", name, "value", "false"]);
        }
        {
            let state = client.search_state.lock().unwrap();
            assert!(!state.params.cfss_enabled);
            assert!(!state.params.ras_enabled);
            assert!(!state.params.bmo_enabled);
            assert!(!state.params.tce_enabled);
            assert!(!state.params.lqt_enabled);
            assert!(!state.params.sps_enabled);
            assert!(!state.params.dad_enabled);
            assert!(!state.params.extension_cap_enabled);
            assert!(!state.params.cpi_enabled);
            assert!(!state.params.state_enabled);
            assert!(!state.params.risk_enabled);
            assert!(!state.params.pressure_enabled);
            assert!(!state.params.attack_enabled);
            assert!(!state.params.conversion_enabled);
            assert!(!state.params.learned_enabled);
        }
        client.run_setoption(&["name", "CFSS_Enabled", "value", "1"]);
        assert!(client.search_state.lock().unwrap().params.cfss_enabled);
        client.run_setoption(&["name", "Extension_Cap", "value", "1"]);
        assert!(
            client
                .search_state
                .lock()
                .unwrap()
                .params
                .extension_cap_enabled
        );
        client.run_setoption(&["name", "Cap", "value", "0"]);
        assert!(
            !client
                .search_state
                .lock()
                .unwrap()
                .params
                .extension_cap_enabled
        );
    }

    #[test]
    fn should_toggle_feature_flags_and_thresholds() {
        let mut client = UciClient::new();
        for name in ["ALP_Enabled", "PSM_Enabled", "GTP_Enabled"] {
            client.run_setoption(&["name", name, "value", "true"]);
        }
        client.run_setoption(&["name", "ALP_Threshold", "value", "1"]);
        client.run_setoption(&["name", "LQT_Threshold", "value", "500"]);
        client.run_setoption(&["name", "GTP_Threshold", "value", "99"]);
        {
            let state = client.search_state.lock().unwrap();
            assert!(state.params.alp_enabled);
            assert!(state.params.psm_enabled);
            assert!(state.params.gtp_enabled);
            assert_eq!(state.params.alp_threshold, 50);
            assert_eq!(state.params.lqt_threshold, 500);
            assert_eq!(state.params.gtp_threshold, 50);
        }
        client.run_setoption(&["name", "ALP_Enabled", "value", "false"]);
        assert!(!client.search_state.lock().unwrap().params.alp_enabled);
    }

    #[test]
    fn should_apply_aprm_thresholds_with_clamps() {
        let mut client = UciClient::new();
        client.run_setoption(&["name", "MultiPV", "value", "3"]);
        client.run_setoption(&["name", "MustTryGain", "value", "50"]);
        client.run_setoption(&["name", "MustTryRisk", "value", "90"]);
        client.run_setoption(&["name", "ConvertScore", "value", "300"]);
        client.run_setoption(&["name", "CrushScore", "value", "800"]);
        client.run_setoption(&["name", "DefendCPI", "value", "100"]);
        client.run_setoption(&["name", "DefendScore", "value", "-100"]);
        client.run_setoption(&["name", "AttackScore", "value", "90"]);
        client.run_setoption(&["name", "AttackCPI", "value", "70"]);
        client.run_setoption(&["name", "ConvertCPI", "value", "35"]);
        client.run_setoption(&["name", "ResetDrop", "value", "60"]);
        client.run_setoption(&["name", "ConcessionMin", "value", "20"]);
        client.run_setoption(&["name", "RiskNormal", "value", "25"]);
        client.run_setoption(&["name", "RiskElevated", "value", "80"]);
        client.run_setoption(&["name", "PressureMomDiv", "value", "3"]);
        client.run_setoption(&["name", "PressureCpiW", "value", "2"]);
        client.run_setoption(&["name", "PressureFreeW", "value", "1"]);
        client.run_setoption(&["name", "PressurePlanW", "value", "4"]);
        client.run_setoption(&["name", "UrgencyMinGain", "value", "40"]);
        client.run_setoption(&["name", "UrgencyMustTryGain", "value", "70"]);
        client.run_setoption(&["name", "VerifyTolerance", "value", "25"]);
        {
            let state = client.search_state.lock().unwrap();
            assert_eq!(state.multipv, 3);
            assert_eq!(state.risk_envelope.min_gain_cp, 50);
            assert_eq!(state.risk_envelope.must_try_max, 90);
            assert_eq!(state.conversion_params.convert_min_cp, 300);
            assert_eq!(state.conversion_params.crush_min_cp, 800);
            assert_eq!(state.state_thresholds.defend_cpi, 100);
            assert_eq!(state.state_thresholds.defend_score, -100);
            assert_eq!(state.state_thresholds.attack_score, 90);
            assert_eq!(state.state_thresholds.attack_cpi, 70);
            assert_eq!(state.state_thresholds.convert_cpi, 35);
            assert_eq!(state.conversion_params.max_opp_cpi, 35);
            assert_eq!(state.state_thresholds.reset_drop, 60);
            assert_eq!(state.state_thresholds.concession_min_cp, 20);
            assert_eq!(state.risk_envelope.normal_max, 25);
            assert_eq!(state.risk_envelope.elevated_max, 80);
            assert_eq!(state.pressure_weights.momentum_div, 3);
            assert_eq!(state.pressure_weights.cpi_w, 2);
            assert_eq!(state.pressure_weights.freedom_w, 1);
            assert_eq!(state.pressure_weights.plans_w, 4);
            assert_eq!(state.urgency_thresholds.min_gain, 40);
            assert_eq!(state.urgency_thresholds.musttry_gain, 70);
            assert_eq!(state.verify_tolerance_cp, 25);
        }

        client.run_setoption(&["name", "MultiPV", "value", "99"]);
        client.run_setoption(&["name", "MustTryGain", "value", "9999"]);
        client.run_setoption(&["name", "DefendScore", "value", "-9999"]);
        client.run_setoption(&["name", "PressureMomDiv", "value", "0"]);
        client.run_setoption(&["name", "VerifyTolerance", "value", "9999"]);
        {
            let state = client.search_state.lock().unwrap();
            assert_eq!(state.multipv, 8);
            assert_eq!(state.risk_envelope.min_gain_cp, 100);
            assert_eq!(state.state_thresholds.defend_score, -400);
            assert_eq!(state.pressure_weights.momentum_div, 1);
            assert_eq!(state.verify_tolerance_cp, 100);
        }
    }

    #[test]
    fn should_apply_contempt_drawscore_showwdl() {
        let mut client = UciClient::new();
        client.run_setoption(&["name", "Contempt", "value", "30"]);
        client.run_setoption(&["name", "DrawScore", "value", "-10"]);
        client.run_setoption(&["name", "ShowWDL", "value", "false"]);
        {
            let state = client.search_state.lock().unwrap();
            assert_eq!(state.contempt_cp, 30);
            assert_eq!(state.draw_score_cp, -10);
            assert!(!state.show_wdl);
        }
        client.run_setoption(&["name", "Contempt", "value", "9999"]);
        client.run_setoption(&["name", "ShowWDL", "value", "true"]);
        {
            let state = client.search_state.lock().unwrap();
            assert_eq!(state.contempt_cp, 200);
            assert!(state.show_wdl);
        }
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
