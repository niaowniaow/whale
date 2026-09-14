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
            rfp_margin_mult: 150,
            futility_margin_mult: 150,
            singular_margin_mult: 1,
            probcut_margin: 200,
            nmp_base: 3,
            nmp_depth_div: 4,
            lmr_base: 0.5,
            lmr_div: 1.95,
            lmr_divisor: [
                3637, 2787, 2761, 2939, 3171, 3347, 3147, 2762, 2772, 3106, 3107, 3060, 3112, 2991,
                3090, 3542,
            ],
            history_weight_mult: 1,
            history_weight_max: 16,
        }
    }
}

pub struct SearchState {
    pub params: SearchParameters,
    pub opt_time: i32,
    pub max_time: i32,
    pub best_move: Move,
    pub score: i16,
    pub nodes: i32,
    pub root_best_move_nodes: i64,
    pub best_previous_score: Option<i16>,
    pub move_ordering: MoveOrdering,
    pub tt: TranspositionTable,

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
            score: 0,
            nodes: 0,
            root_best_move_nodes: 0,
            best_previous_score: None,
            move_ordering: MoveOrdering::new(),
            tt: TranspositionTable::new(TranspositionTable::DEFAULT_CAPACITY),
            captures_stack: Box::new([MoveList::new(); MAX_PLY]),
            quiets_stack: Box::new([MoveList::new(); MAX_PLY]),
            eval_stack: Box::new([i16::MIN; MAX_PLY]),
            lmr_table,
            correction_history: crate::search::correction_history::CorrectionHistory::new(),
        }
    }

    pub fn reset_search(&mut self) {
        self.best_move = Move::NO_MOVE;
        self.score = 0;
        self.nodes = 0;
        self.root_best_move_nodes = 0;
        *self.eval_stack = [i16::MIN; MAX_PLY];
    }

    pub fn reset_heuristics(&mut self) {
        self.move_ordering.reset();
        self.correction_history.clear();
    }
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new()
    }
}
