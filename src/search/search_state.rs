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
            history_weight_mult: 1,
            history_weight_max: 16,
        }
    }
}

pub struct SearchState {
    pub params: SearchParameters,
    pub opt_time: i32,
    pub best_move: Move,
    pub score: i16,
    pub nodes: i32,
    pub move_ordering: MoveOrdering,
    pub tt: TranspositionTable,

    pub captures_stack: [MoveList; MAX_PLY],
    pub quiets_stack: [MoveList; MAX_PLY],

    pub pawn_correction_history: Box<[[i32; 16384]; 2]>,
    pub non_pawn_correction_history: Box<[[i32; 16384]; 2]>,
}

impl SearchState {
    pub fn new() -> Self {
        Self {
            params: SearchParameters::default(),
            opt_time: -1,
            best_move: Move::NO_MOVE,
            score: 0,
            nodes: 0,
            move_ordering: MoveOrdering::new(),
            tt: TranspositionTable::new(TranspositionTable::DEFAULT_CAPACITY),
            captures_stack: [MoveList::new(); MAX_PLY],
            quiets_stack: [MoveList::new(); MAX_PLY],
            pawn_correction_history: Box::new([[0; 16384]; 2]),
            non_pawn_correction_history: Box::new([[0; 16384]; 2]),
        }
    }

    pub fn reset_search(&mut self) {
        self.best_move = Move::NO_MOVE;
        self.score = 0;
        self.nodes = 0;
    }

    pub fn reset_heuristics(&mut self) {
        self.move_ordering.reset();
        *self.pawn_correction_history = [[0; 16384]; 2];
        *self.non_pawn_correction_history = [[0; 16384]; 2];
    }
}

impl Default for SearchState {
    fn default() -> Self {
        Self::new()
    }
}
