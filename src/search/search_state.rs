use crate::common::constants::MAX_PLY;
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::side::Side;
use crate::common::tt::TranspositionTable;
use crate::eval::move_ordering::MoveOrdering;

#[derive(Clone, Debug)]
pub struct SearchParameters {
    pub rfp_margin_mult: i16,
    pub futility_margin_mult: i16,
    pub singular_margin_mult: i16,
    pub probcut_margin: i16,
    pub nmp_base: u8,
    pub nmp_depth_div: u8,
    pub lmr_base: f64,
    pub lmr_div: f64,
    pub lmr_divisor: [i32; 16],
    pub history_weight_mult: i32,
    pub history_weight_max: i32,
    pub alp_enabled: bool,
    pub alp_threshold: u8,
    pub psm_enabled: bool,
    pub gtp_enabled: bool,
    pub gtp_threshold: u8,

    pub cfss_enabled: bool,
    pub ras_enabled: bool,
    pub bmo_enabled: bool,
    pub tce_enabled: bool,
    pub lqt_enabled: bool,
    pub sps_enabled: bool,
    pub dad_enabled: bool,
    pub extension_cap_enabled: bool,

    pub cpi_enabled: bool,

    pub state_enabled: bool,

    pub risk_enabled: bool,

    pub pressure_enabled: bool,

    pub attack_enabled: bool,

    pub conversion_enabled: bool,

    pub qs_checks_enabled: bool,
}

impl Default for SearchParameters {
    fn default() -> Self {
        Self {
            rfp_margin_mult: 110,
            futility_margin_mult: 120,
            singular_margin_mult: 2,
            probcut_margin: 170,
            nmp_base: 3,
            nmp_depth_div: 3,
            lmr_base: 0.65,
            lmr_div: 2.15,
            lmr_divisor: [
                3637, 2787, 2761, 2939, 3171, 3347, 3147, 2762, 2772, 3106, 3107, 3060, 3112, 2991,
                3090, 3542,
            ],
            history_weight_mult: 2,
            history_weight_max: 16,
            alp_enabled: true,
            alp_threshold: 75,
            psm_enabled: true,
            gtp_enabled: true,
            gtp_threshold: 15,
            cfss_enabled: true,
            ras_enabled: true,
            bmo_enabled: true,
            tce_enabled: true,
            lqt_enabled: true,
            sps_enabled: true,
            dad_enabled: true,
            extension_cap_enabled: true,
            cpi_enabled: true,
            state_enabled: true,
            risk_enabled: true,
            pressure_enabled: true,
            attack_enabled: true,
            conversion_enabled: true,
            qs_checks_enabled: true,
        }
    }
}

use std::sync::Arc;

pub struct SearchState {
    pub params: SearchParameters,
    pub opt_time: i32,
    pub max_time: i32,
    pub best_move: Move,
    pub ponder_move: Move,
    pub score: i16,
    pub nodes: u64,
    pub seldepth: u32,
    pub tbhits: u64,
    pub root_best_move_nodes: u64,

    pub max_nodes: u64,
    pub mate_in: u8,
    pub searchmoves: Vec<Move>,
    pub best_previous_score: Option<i16>,
    pub previous_time_reduction: f64,
    pub best_move_changes: i32,
    pub optimism: [i32; 2],

    pub engine_side: Side,

    pub contempt_cp: i16,

    pub draw_score_cp: i16,

    pub show_wdl: bool,
    pub move_ordering: MoveOrdering,
    pub tt: Arc<TranspositionTable>,

    pub captures_stack: Box<[MoveList; MAX_PLY]>,
    pub quiets_stack: Box<[MoveList; MAX_PLY]>,
    pub eval_stack: Box<[i16; MAX_PLY]>,

    pub extension_streak: Box<[u8; MAX_PLY]>,

    pub lmr_table: crate::search::lmr::LmrTable,
    pub correction_history: crate::search::correction_history::CorrectionHistory,
    pub bmo: crate::search::bmo::BanditMoveOrdering,
    pub ras: crate::search::ras::RuntimeAnnealer,
    pub persona: crate::search::sps::SearchPersona,
    pub psm_stack: Box<crate::search::psm::PsmStack>,

    pub multipv: usize,

    pub multipv_lines: Vec<crate::search::multipv::MultipvLine>,

    pub pressure_state: crate::search::pressure::PressureState,
    pub state_thresholds: crate::search::position_state::StateThresholds,
    pub risk_envelope: crate::search::risk::RiskEnvelope,
    pub conversion_params: crate::search::conversion::ConversionParams,

    pub prev_root_score: Option<i16>,
    pub prev_opp_cpi: Option<i32>,
    pub prev_opp_freedom: Option<u32>,

    pub prev_opp_breaks: Option<u32>,

    pub verification_budget: u8,

    pub reset_mode: bool,

    pub simplify_bias: bool,

    pub last_musttry: bool,

    pub last_verified: Option<i16>,

    pub last_state: crate::search::position_state::PositionState,

    pub last_concession: Option<crate::search::concession::Concession>,
}

impl SearchState {
    pub fn new() -> Self {
        let params = SearchParameters::default();
        let lmr_table = crate::search::lmr::LmrTable::new(params.lmr_base, params.lmr_div);
        Self {
            params,
            opt_time: -1,
            max_time: -1,
            best_move: Move::NO_MOVE,
            ponder_move: Move::NO_MOVE,
            score: 0,
            nodes: 0,
            seldepth: 0,
            tbhits: 0,
            root_best_move_nodes: 0,
            max_nodes: 0,
            mate_in: 0,
            searchmoves: Vec::new(),
            best_previous_score: None,
            previous_time_reduction: 0.85,
            best_move_changes: 0,
            optimism: [0; 2],
            engine_side: Side::White,
            contempt_cp: 0,
            draw_score_cp: 0,
            show_wdl: true,
            move_ordering: MoveOrdering::new(),
            tt: Arc::new(TranspositionTable::new(
                TranspositionTable::DEFAULT_CAPACITY,
            )),
            captures_stack: Box::new([MoveList::new(); MAX_PLY]),
            quiets_stack: Box::new([MoveList::new(); MAX_PLY]),
            eval_stack: Box::new([i16::MIN; MAX_PLY]),
            extension_streak: Box::new([0u8; MAX_PLY]),
            lmr_table,
            correction_history: crate::search::correction_history::CorrectionHistory::new(),
            bmo: crate::search::bmo::BanditMoveOrdering::new(),
            ras: crate::search::ras::RuntimeAnnealer::new(),
            persona: crate::search::sps::SearchPersona::Standard,
            psm_stack: Box::new(crate::search::psm::PsmStack::new()),
            multipv: 1,
            multipv_lines: Vec::new(),
            pressure_state: crate::search::pressure::PressureState::default(),
            state_thresholds: crate::search::position_state::StateThresholds::default(),
            risk_envelope: crate::search::risk::RiskEnvelope::default(),
            conversion_params: crate::search::conversion::ConversionParams::default(),
            prev_root_score: None,
            prev_opp_cpi: None,
            prev_opp_freedom: None,
            prev_opp_breaks: None,
            verification_budget: 0,
            reset_mode: false,
            simplify_bias: false,
            last_musttry: false,
            last_verified: None,
            last_state: crate::search::position_state::PositionState::default(),
            last_concession: None,
        }
    }

    pub fn clone_for_worker(&self, thread_id: usize) -> Self {
        let persona = if self.params.sps_enabled {
            crate::search::sps::persona_for_thread(thread_id)
        } else {
            crate::search::sps::SearchPersona::Standard
        };
        let mut params = self.params.clone();
        let mut lmr_table = self.lmr_table.clone();
        crate::search::sps::apply_persona(persona, &mut params, &mut lmr_table);
        Self {
            params,
            opt_time: self.opt_time,
            max_time: self.max_time,
            best_move: Move::NO_MOVE,
            ponder_move: Move::NO_MOVE,
            score: 0,
            nodes: 0,
            seldepth: 0,
            tbhits: 0,
            root_best_move_nodes: 0,
            max_nodes: self.max_nodes,
            mate_in: self.mate_in,
            searchmoves: self.searchmoves.clone(),
            best_previous_score: None,
            previous_time_reduction: self.previous_time_reduction,
            best_move_changes: 0,
            optimism: self.optimism,
            engine_side: self.engine_side,
            contempt_cp: self.contempt_cp,
            draw_score_cp: self.draw_score_cp,
            show_wdl: self.show_wdl,
            move_ordering: MoveOrdering::new(),
            tt: Arc::clone(&self.tt),
            captures_stack: Box::new([MoveList::new(); MAX_PLY]),
            quiets_stack: Box::new([MoveList::new(); MAX_PLY]),
            eval_stack: Box::new([i16::MIN; MAX_PLY]),
            extension_streak: Box::new([0u8; MAX_PLY]),
            lmr_table,
            correction_history: crate::search::correction_history::CorrectionHistory::new(),
            bmo: self.bmo.clone(),
            ras: crate::search::ras::RuntimeAnnealer::new(),
            persona,
            psm_stack: Box::new(crate::search::psm::PsmStack::new()),

            multipv: 1,
            multipv_lines: Vec::new(),
            pressure_state: crate::search::pressure::PressureState::default(),
            state_thresholds: self.state_thresholds,
            risk_envelope: self.risk_envelope,
            conversion_params: self.conversion_params,
            prev_root_score: None,
            prev_opp_cpi: None,
            prev_opp_freedom: None,
            prev_opp_breaks: None,

            verification_budget: self.verification_budget,
            reset_mode: self.reset_mode,
            simplify_bias: self.simplify_bias,
            last_musttry: false,
            last_verified: None,
            last_state: crate::search::position_state::PositionState::default(),
            last_concession: None,
        }
    }

    pub fn reset_search(&mut self) {
        self.best_move = Move::NO_MOVE;
        self.ponder_move = Move::NO_MOVE;
        self.score = 0;
        self.nodes = 0;
        self.seldepth = 0;
        self.tbhits = 0;
        self.root_best_move_nodes = 0;
        self.best_move_changes = 0;
        self.optimism = [0; 2];

        self.multipv_lines.clear();
        self.pressure_state = crate::search::pressure::PressureState::default();
        self.prev_root_score = None;
        self.prev_opp_cpi = None;
        self.prev_opp_freedom = None;
        self.prev_opp_breaks = None;
        self.verification_budget = 0;
        self.reset_mode = false;
        self.simplify_bias = false;
        self.last_musttry = false;
        self.last_verified = None;
        self.last_state = crate::search::position_state::PositionState::default();
        self.last_concession = None;
        *self.eval_stack = [i16::MIN; MAX_PLY];
        *self.extension_streak = [0u8; MAX_PLY];
        self.bmo.reset();
        self.psm_stack.reset();
    }

    pub fn reset_heuristics(&mut self) {
        self.previous_time_reduction = 0.85;
        self.best_previous_score = None;
        self.move_ordering.reset();
        self.correction_history.clear();
    }
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::sps::SearchPersona;

    #[test]
    fn clone_for_worker_assigns_personas_per_thread() {
        let primary = SearchState::new();

        let w0 = primary.clone_for_worker(0);
        assert_eq!(w0.persona, SearchPersona::Standard);
        assert_eq!(
            w0.params.futility_margin_mult,
            primary.params.futility_margin_mult
        );

        let w1 = primary.clone_for_worker(1);
        assert_eq!(w1.persona, SearchPersona::Tactical);
        assert_eq!(w1.params.futility_margin_mult, 160);
        assert_eq!(w1.params.probcut_margin, 140);

        let w2 = primary.clone_for_worker(2);
        assert_eq!(w2.persona, SearchPersona::Solid);
        assert_eq!(w2.params.nmp_depth_div, 2);

        let w3 = primary.clone_for_worker(3);
        assert_eq!(w3.persona, SearchPersona::Aggressive);
        assert_eq!(w3.params.futility_margin_mult, 140);

        assert_eq!(primary.clone_for_worker(4).persona, SearchPersona::Standard);
    }

    #[test]
    fn clone_for_worker_carries_draw_options() {
        let mut primary = SearchState::new();
        primary.contempt_cp = 25;
        primary.draw_score_cp = -5;
        primary.show_wdl = false;
        primary.engine_side = Side::Black;
        let worker = primary.clone_for_worker(1);
        assert_eq!(worker.contempt_cp, 25);
        assert_eq!(worker.draw_score_cp, -5);
        assert!(!worker.show_wdl);

        assert_eq!(worker.engine_side, Side::Black);
    }

    #[test]
    fn clone_for_worker_keeps_tt_shared_and_optimism_uniform() {
        use std::sync::Arc;

        let mut primary = SearchState::new();
        primary.optimism = [12, -12];

        primary.params.futility_margin_mult = 111;

        for thread_id in 0..8 {
            let worker = primary.clone_for_worker(thread_id);

            assert!(Arc::ptr_eq(&worker.tt, &primary.tt));

            assert_eq!(worker.optimism, [12, -12]);
            if worker.persona == SearchPersona::Standard {
                assert_eq!(worker.params.futility_margin_mult, 111);
            }
            assert_eq!(
                worker.params.extension_cap_enabled,
                primary.params.extension_cap_enabled
            );
        }
    }
}
