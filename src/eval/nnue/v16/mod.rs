use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use crate::board::state::BoardState;
use crate::common::piece::Piece;
use crate::common::side::Side;
use crate::common::square::Square;

pub mod arch;
pub mod eval;
pub mod inference;
pub mod loader;
pub mod pairs;
pub mod position;
#[cfg(test)]
mod tests;
pub mod threats;

pub use arch::{
    BIG_INPUT_DIMS, BISHOP_VALUE, FC0_ACT, FC0_OUT, FC1_IN, FC1_OUT, KING_BUCKET_COUNT,
    KNIGHT_VALUE, L1, MAX_ACTIVE, MAX_PAIR_ACTIVE, MAX_THREAT_ACTIVE, N_BUCKETS, OUTPUT_SCALE,
    PAIR_DIMS, PAIR_HASH, PAWN_VALUE, PS_PLANES, PSQ_DIMS, PSQ_HASH, QUEEN_VALUE, ROOK_VALUE,
    SF_FILE_VERSION, SF17_FILE_VERSION, SMALL_FALLBACK, SMALL_GATE, THREAT_DIMS, THREAT_HASH,
    VERSION, WEIGHT_SCALE_BITS, affine_hash, arch_hash, material_bucket, network_hash, simple_eval,
    transformer_hash,
};
#[cfg(test)]
pub(crate) use eval::EVAL_TEST_LOCK;
pub use eval::{
    FinnyEntry, LoadedNets, Sfnn16Accs, SfnnEval, SfnnEvent, SfnnPending, active_loaded_net,
    active_model_name, apply_queued, collect_pairs_lazy, collect_threats_lazy, ensure_fresh,
    ensure_sfnn16_fresh, evaluate_board, evaluate_board_detailed, evaluate_nets, flush_pending,
    has_avx2, load_net, maintenance_active, note_add, note_remove, pairs_cached_len, refresh_all,
    refresh_perspective, resolve_model_path, set_eval_file, threats_cached_len, trainer_features,
    try_load_default, try_load_default_path, unload_nets, update_perspective_finny_or_refresh,
};
pub use loader::{Sfnn16Net, SfnnArch, SfnnTransformer};
pub use pairs::{append_pairs, collect_pairs, for_each_pair, pair_index_for, pair_make_index};
pub use position::{SfnnPosition, append_halfka, halfka_index};
pub use threats::{append_threats, collect_threats, for_each_threat, threat_index_for};
