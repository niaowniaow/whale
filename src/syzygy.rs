//! Syzygy endgame tablebase support (perfect play with few pieces left).
//!
//! Uses the `shakmaty-syzygy` crate for probing. Table files (`.rtbw` /
//! `.rtbz`) are NOT shipped with this repo; point the UCI option
//! `SyzygyPath` at a directory containing them (3-4-5 piece tables are
//! enough to start, e.g. from https://tablebase.lichess.ovh/tables/standard/).
//! When no tables are configured, every function below is a cheap no-op.
//!
//! Design (deliberately simple and safe):
//! - root: if the position is an unconditional table win, play the
//!   DTZ-optimal table move directly;
//! - search: unconditional WDL Win/Loss results become hard TB bounds so the
//!   search never walks out of a won ending (or into a lost one).
//!   Ambiguous results (cursed wins, blessed losses) are left to normal search
//!   so the 50-move rule stays correct.

use std::sync::{Arc, RwLock};

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase};

use crate::board::state::BoardState;
use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::tt::TranspositionEntryType;

/// Highest score a tablebase win can produce. Sits just below the weakest
/// possible mate score so mates are always preferred over TB wins.
pub const TB_WIN: i16 = MAX_CENTIPAWN_EVAL - MAX_PLY as i16 - 10;

/// Only probe positions with at most this many pieces on the board.
/// Raised 5 -> 7: shakmaty-syzygy 0.28 supports `Chess::MAX_PIECES == 7`
/// (see registry src types.rs), so 6-7 piece tables work when present.
pub const TB_MAX_PIECES: usize = 7;

// FIX M12: configurable Syzygy guards with Stockfish defaults (cardinality 7,
// probe depth 1, useRule50 true). Team UCI calls these from setoption.
static PROBE_LIMIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(7);
static PROBE_DEPTH: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);
static USE_50MR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Maximum pieces to probe (Stockfish `SyzygyProbeLimit`, default 7).
pub fn set_probe_limit(n: u8) {
    PROBE_LIMIT.store(n.min(7), std::sync::atomic::Ordering::Relaxed);
}

/// Minimum depth to probe when piece count equals the limit
/// (Stockfish `SyzygyProbeDepth`, default 1).
pub fn set_probe_depth(n: u8) {
    PROBE_DEPTH.store(n, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the 50-move rule can convert TB wins (Stockfish `Syzygy50MoveRule`,
/// default true).
pub fn set_50mr_rule(b: bool) {
    USE_50MR.store(b, std::sync::atomic::Ordering::Relaxed);
}

#[inline(always)]
pub fn probe_limit() -> u8 {
    PROBE_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn probe_depth() -> u8 {
    PROBE_DEPTH.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline(always)]
pub fn use_50mr() -> bool {
    USE_50MR.load(std::sync::atomic::Ordering::Relaxed)
}

struct LoadedTables {
    tables: Tablebase<Chess>,
    count: usize,
}

static TABLES: RwLock<Option<Arc<LoadedTables>>> = RwLock::new(None);

/// Point the engine at a directory of `.rtbw`/`.rtbz` files. Returns the
/// number of tables loaded. An empty path clears the configuration.
pub fn set_path(path: &str) -> Result<usize, String> {
    if path.is_empty() || path == "<empty>" {
        if let Ok(mut guard) = TABLES.write() {
            *guard = None;
        }
        return Ok(0);
    }
    let mut tables = Tablebase::<Chess>::new();
    let loaded = tables
        .add_directory(path)
        .map_err(|e| format!("syzygy load error: {e}"))?;
    if let Ok(mut guard) = TABLES.write() {
        *guard = Some(Arc::new(LoadedTables {
            tables,
            count: loaded,
        }));
    }
    Ok(loaded)
}

/// Number of configured tables (0 when Syzygy is off).
pub fn table_count() -> usize {
    TABLES
        .read()
        .ok()
        .and_then(|guard| guard.clone())
        .map(|loaded| loaded.count)
        .unwrap_or(0)
}

fn to_shakmaty(board: &BoardState) -> Option<Chess> {
    let fen = board.to_fen();
    Fen::from_ascii(fen.as_bytes())
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()
}

fn move_to_uci(m: Move) -> String {
    let mut s = format!("{}{}", m.source, m.target);
    if let Some(p) = m.promotion_char() {
        s.push(p);
    }
    s
}

/// WDL result from the side-to-move's point of view, mapped onto our score
/// scale with a Stockfish-style bound (Win->LOWER/Beta, Loss->UPPER/Alpha,
/// Draw->EXACT). Only unconditional results produce a bound; cursed wins and
/// blessed losses are left to normal search so the 50-move rule stays correct
/// (unless `set_50mr_rule(false)`, which treats them as decisive).
/// Callers must apply the SYZYGY-BOUND rule: only cut on EXACT or
/// (LOWER && value>=beta) / (UPPER && value<=alpha), otherwise keep searching.
pub fn probe_bound(board: &BoardState, ply: u8) -> Option<(i16, TranspositionEntryType)> {
    if board.occupancy().count_ones() as usize > TB_MAX_PIECES {
        return None;
    }
    if board.occupancy().count_ones() as usize > probe_limit() as usize {
        return None;
    }
    let pos = to_shakmaty(board)?;
    let loaded = TABLES.read().ok().and_then(|guard| guard.clone())?;
    if loaded.count == 0 {
        return None;
    }
    let use_rule50 = use_50mr();
    let bound_score = (TB_WIN - ply as i16).max(0);
    match loaded.tables.probe_wdl(&pos).ok()? {
        AmbiguousWdl::Win => Some((bound_score, TranspositionEntryType::Beta)),
        AmbiguousWdl::Loss => Some((-bound_score, TranspositionEntryType::Alpha)),
        AmbiguousWdl::Draw => Some((0, TranspositionEntryType::Exact)),
        // With Syzygy50MoveRule=false, cursed/blessed are treated as decisive.
        AmbiguousWdl::CursedWin if !use_rule50 => Some((bound_score, TranspositionEntryType::Beta)),
        AmbiguousWdl::BlessedLoss if !use_rule50 => {
            Some((-bound_score, TranspositionEntryType::Alpha))
        }
        _ => None,
    }
}

/// If the root position is an unconditional table win, return the
/// DTZ-optimal table move directly. Returns `None` otherwise.
pub fn root_move(board: &mut BoardState) -> Option<Move> {
    if board.occupancy().count_ones() as usize > TB_MAX_PIECES {
        return None;
    }
    let pos = to_shakmaty(board)?;
    let loaded = TABLES.read().ok().and_then(|guard| guard.clone())?;
    if loaded.count == 0 {
        return None;
    }
    if !matches!(loaded.tables.probe_wdl(&pos).ok()?, AmbiguousWdl::Win) {
        return None;
    }
    let (tb_move, _) = loaded.tables.best_move(&pos).ok()??;
    let want = tb_move.to_uci(CastlingMode::Standard).to_string();

    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);
    for i in 0..moves.len() {
        let move_obj = moves[i].mv;
        if move_to_uci(move_obj) != want {
            continue;
        }
        // Verify legality the same way search does.
        board.make_move(move_obj);
        let legal = !board.is_in_check(board.side_to_move.other());
        board.unmake_move(move_obj);
        if legal {
            return Some(move_obj);
        }
    }
    None
}
