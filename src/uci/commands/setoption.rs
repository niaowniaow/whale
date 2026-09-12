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

        if name.eq_ignore_ascii_case("Hash")
            && let Ok(mb_size) = value.parse::<usize>()
        {
            let mb_size = mb_size.clamp(1, 2048);
            if let Ok(mut state) = self.search_state.try_lock() {
                state.tt.resize(mb_size);
            }
            // else, as per UCI specs, we should just ignore it (e.g. attempt to resize mid-search)
        }

        if name.eq_ignore_ascii_case("EvalFile") || name.eq_ignore_ascii_case("EvalFileSmall") {
            match crate::eval::nnue::v16::set_eval_file(&name, &value) {
                Ok(msg) => crate::uci::cli::write_line(&format!("info string {msg}")),
                Err(e) => crate::uci::cli::write_line(&format!("info string EvalFile error: {e}")),
            }
        }

        if name.eq_ignore_ascii_case("RFPMarginMult") {
            if let Ok(val) = value.parse::<i16>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.rfp_margin_mult = val;
                }
            }
        } else if name.eq_ignore_ascii_case("FutilityMarginMult") {
            if let Ok(val) = value.parse::<i16>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.futility_margin_mult = val;
                }
            }
        } else if name.eq_ignore_ascii_case("SingularMarginMult") {
            if let Ok(val) = value.parse::<i16>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.singular_margin_mult = val;
                }
            }
        } else if name.eq_ignore_ascii_case("ProbCutMargin") {
            if let Ok(val) = value.parse::<i16>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.probcut_margin = val;
                }
            }
        } else if name.eq_ignore_ascii_case("NmpBase") {
            if let Ok(val) = value.parse::<u8>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.nmp_base = val;
                }
            }
        } else if name.eq_ignore_ascii_case("NmpDepthDiv") {
            if let Ok(val) = value.parse::<u8>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.nmp_depth_div = val;
                }
            }
        } else if name.eq_ignore_ascii_case("LmrBase") {
            if let Ok(val) = value.parse::<f64>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.lmr_base = val / 100.0;
                }
            }
        } else if name.eq_ignore_ascii_case("LmrDiv") {
            if let Ok(val) = value.parse::<f64>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.lmr_div = val / 100.0;
                }
            }
        } else if name.eq_ignore_ascii_case("HistoryWeightMult") {
            if let Ok(val) = value.parse::<i32>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.history_weight_mult = val;
                }
            }
        } else if name.eq_ignore_ascii_case("HistoryWeightMax") {
            if let Ok(val) = value.parse::<i32>() {
                if let Ok(mut state) = self.search_state.try_lock() {
                    state.params.history_weight_max = val;
                }
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
}
