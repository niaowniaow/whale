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
            if name.eq_ignore_ascii_case("Hash")
                && let Ok(mb_size) = value.parse::<usize>()
            {
                let mb_size = mb_size.clamp(1, 2048);
                state.tt = std::sync::Arc::new(crate::common::tt::TranspositionTable::new_mb(mb_size));
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
            assert_eq!(state.tt.capacity(), 2097152); // 64MB
        }

        uci_client.run_setoption(&["name", "Hash", "value", "128"]);

        {
            let state = uci_client.search_state.lock().unwrap();
            assert_eq!(state.tt.capacity(), 4194304); // 128MB
        }

        uci_client.run_setoption(&["name", "hash", "value", "1"]);

        {
            let state = uci_client.search_state.lock().unwrap();
            assert_eq!(state.tt.capacity(), 32768); // 1MB
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
}
