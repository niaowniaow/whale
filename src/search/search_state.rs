use crate::common::constants::MAX_PLY;
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
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
    pub nodes: i32,
    pub root_best_move_nodes: i64,
    pub best_previous_score: Option<i16>,
    pub previous_time_reduction: f64,
    pub best_move_changes: i32,
    pub optimism: [i32; 2],
    pub move_ordering: MoveOrdering,
    pub tt: Arc<TranspositionTable>,

    pub captures_stack: Box<[MoveList; MAX_PLY]>,
    pub quiets_stack: Box<[MoveList; MAX_PLY]>,
    pub eval_stack: Box<[i16; MAX_PLY]>,

    pub lmr_table: crate::search::lmr::LmrTable,
    pub correction_history: crate::search::correction_history::CorrectionHistory,
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
            root_best_move_nodes: 0,
            best_previous_score: None,
            previous_time_reduction: 0.85,
            best_move_changes: 0,
            optimism: [0; 2],
            move_ordering: MoveOrdering::new(),
            tt: Arc::new(TranspositionTable::new(TranspositionTable::DEFAULT_CAPACITY)),
            captures_stack: Box::new([MoveList::new(); MAX_PLY]),
            quiets_stack: Box::new([MoveList::new(); MAX_PLY]),
            eval_stack: Box::new([i16::MIN; MAX_PLY]),
            lmr_table,
            correction_history: crate::search::correction_history::CorrectionHistory::new(),
        }
    }

    pub fn clone_for_worker(&self) -> Self {
        Self {
            params: self.params.clone(),
            opt_time: self.opt_time,
            max_time: self.max_time,
            best_move: Move::NO_MOVE,
            ponder_move: Move::NO_MOVE,
            score: 0,
            nodes: 0,
            root_best_move_nodes: 0,
            best_previous_score: None,
            previous_time_reduction: self.previous_time_reduction,
            best_move_changes: 0,
            optimism: self.optimism,
            move_ordering: MoveOrdering::new(),
            tt: Arc::clone(&self.tt),
            captures_stack: Box::new([MoveList::new(); MAX_PLY]),
            quiets_stack: Box::new([MoveList::new(); MAX_PLY]),
            eval_stack: Box::new([i16::MIN; MAX_PLY]),
            lmr_table: self.lmr_table.clone(),
            correction_history: crate::search::correction_history::CorrectionHistory::new(),
        }
    }

    pub fn reset_search(&mut self) {
        self.best_move = Move::NO_MOVE;
        self.ponder_move = Move::NO_MOVE;
        self.score = 0;
        self.nodes = 0;
        self.root_best_move_nodes = 0;
        self.best_move_changes = 0;
        self.optimism = [0; 2];
        *self.eval_stack = [i16::MIN; MAX_PLY];
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
