use std::sync::{Arc, RwLock};

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use shakmaty_syzygy::{AmbiguousWdl, Dtz, MaybeRounded, Tablebase};

use crate::board::state::BoardState;
use crate::common::constants::{MAX_CENTIPAWN_EVAL, MAX_PLY};
use crate::common::move_list::MoveList;
use crate::common::moves::Move;
use crate::common::tt::TranspositionEntryType;

pub const TB_WIN: i16 = MAX_CENTIPAWN_EVAL - MAX_PLY as i16 - 10;

pub const TB_MAX_PIECES: usize = 7;

static PROBE_LIMIT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(7);
static PROBE_DEPTH: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(1);
static USE_50MR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_probe_limit(n: u8) {
    PROBE_LIMIT.store(n.min(7), std::sync::atomic::Ordering::Relaxed);
}

pub fn set_probe_depth(n: u8) {
    PROBE_DEPTH.store(n, std::sync::atomic::Ordering::Relaxed);
}

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

#[inline(always)]
pub fn probe_config() -> (u8, u8) {
    (probe_limit(), probe_depth())
}

struct LoadedTables {
    tables: Tablebase<Chess>,
    count: usize,
}

static TABLES: RwLock<Option<Arc<LoadedTables>>> = RwLock::new(None);

pub fn set_path(path: &str) -> Result<usize, String> {
    if path.is_empty() || path == "<empty>" {
        if let Ok(mut guard) = TABLES.write() {
            *guard = None;
        }
        return Ok(0);
    }
    let mut tables = Tablebase::<Chess>::new();
    match tables.add_directory(path) {
        Ok(loaded) => {
            if let Ok(mut guard) = TABLES.write() {
                *guard = Some(Arc::new(LoadedTables {
                    tables,
                    count: loaded,
                }));
            }
            Ok(loaded)
        }
        Err(e) => {
            if let Ok(mut guard) = TABLES.write() {
                *guard = None;
            }
            Err(format!("syzygy load error: {e}"))
        }
    }
}

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

#[inline(always)]
pub fn classify_wdl(w: AmbiguousWdl, use_rule50: bool) -> i8 {
    match w {
        AmbiguousWdl::Win => 1,
        AmbiguousWdl::Loss => -1,
        AmbiguousWdl::Draw => 0,
        AmbiguousWdl::CursedWin => {
            if use_rule50 {
                0
            } else {
                1
            }
        }
        AmbiguousWdl::BlessedLoss => {
            if use_rule50 {
                0
            } else {
                -1
            }
        }
        AmbiguousWdl::MaybeWin => {
            if use_rule50 {
                0
            } else {
                1
            }
        }
        AmbiguousWdl::MaybeLoss => {
            if use_rule50 {
                0
            } else {
                -1
            }
        }
    }
}

#[inline(always)]
fn dtz_to_plies(v: MaybeRounded<Dtz>) -> i32 {
    match v {
        MaybeRounded::Rounded(d) => d.0,
        MaybeRounded::Precise(d) => d.0,
    }
}

pub fn probe_wdl(board: &BoardState) -> Option<i8> {
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
    let w = loaded.tables.probe_wdl(&pos).ok()?;
    Some(classify_wdl(w, use_50mr()))
}

pub fn probe_dtz_plies(board: &BoardState) -> Option<i32> {
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
    let v = loaded.tables.probe_dtz(&pos).ok()?;
    Some(dtz_to_plies(v))
}

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

        AmbiguousWdl::CursedWin if !use_rule50 => Some((bound_score, TranspositionEntryType::Beta)),
        AmbiguousWdl::BlessedLoss if !use_rule50 => {
            Some((-bound_score, TranspositionEntryType::Alpha))
        }
        _ => None,
    }
}

fn fallback_best_move(
    board: &mut BoardState,
    pos: &Chess,
    loaded: &Arc<LoadedTables>,
) -> Option<Move> {
    let (tb_move, _) = loaded.tables.best_move(pos).ok()??;
    let want = tb_move.to_uci(CastlingMode::Standard).to_string();
    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);
    for i in 0..moves.len() {
        let move_obj = moves[i].mv;
        if move_to_uci(move_obj) != want {
            continue;
        }
        board.make_move(move_obj);
        let legal = !board.is_in_check(board.side_to_move.other());
        board.unmake_move(move_obj);
        if legal {
            return Some(move_obj);
        }
    }
    None
}

pub fn root_move(board: &mut BoardState) -> Option<Move> {
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
    let cur_wdl = loaded.tables.probe_wdl(&pos).ok()?;
    let cur = classify_wdl(cur_wdl, use_rule50);
    let mut moves = MoveList::new();
    board.generate_moves(&mut moves);
    let mut legal: Vec<Move> = Vec::new();
    for i in 0..moves.len() {
        let m = moves[i].mv;
        if board.is_legal(m) {
            legal.push(m);
        }
    }
    if legal.is_empty() {
        return None;
    }
    let mut best_win: Option<(Move, i32)> = None;
    let mut best_draw: Option<(Move, i32)> = None;
    let mut best_loss: Option<(Move, i32)> = None;
    for m in legal {
        board.make_move(m);
        let child_pos = match to_shakmaty(board) {
            Some(p) => p,
            None => {
                board.unmake_move(m);
                continue;
            }
        };
        let child_wdl = match loaded.tables.probe_wdl(&child_pos) {
            Ok(v) => v,
            Err(_) => {
                board.unmake_move(m);
                continue;
            }
        };
        let s = classify_wdl(child_wdl, use_rule50);
        let d = match loaded.tables.probe_dtz(&child_pos) {
            Ok(v) => dtz_to_plies(v),
            Err(_) => 0,
        };
        board.unmake_move(m);
        if s == -1 {
            let ad = d.abs();
            match best_win {
                None => best_win = Some((m, d)),
                Some((_, bd)) => {
                    if ad < bd.abs() {
                        best_win = Some((m, d));
                    }
                }
            }
        } else if s == 0 {
            let ad = d.abs();
            match best_draw {
                None => best_draw = Some((m, d)),
                Some((_, bd)) => {
                    if ad < bd.abs() {
                        best_draw = Some((m, d));
                    }
                }
            }
        } else {
            match best_loss {
                None => best_loss = Some((m, d)),
                Some((_, bd)) => {
                    if d > bd {
                        best_loss = Some((m, d));
                    }
                }
            }
        }
    }
    if let Some((m, _)) = best_win {
        return Some(m);
    }
    if (cur == 0 || cur == -1)
        && let Some((m, _)) = best_draw
    {
        return Some(m);
    }
    if cur == -1
        && let Some((m, _)) = best_loss
    {
        return Some(m);
    }
    if cur == 1 {
        if let Some((m, _)) = best_draw {
            return Some(m);
        }
        if let Some((m, _)) = best_loss {
            return Some(m);
        }
    } else if cur == 0
        && let Some((m, _)) = best_loss
    {
        return Some(m);
    }
    fallback_best_move(board, &pos, &loaded)
}

#[cfg(test)]
pub(crate) static SYZYGY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::helpers::STARTING_FEN;

    fn clear_tables() {
        let _ = set_path("");
        set_probe_limit(7);
        set_probe_depth(1);
        set_50mr_rule(true);
    }

    #[test]
    fn probe_guards_roundtrip_with_clamp() {
        let _serial = SYZYGY_TEST_LOCK.lock().unwrap();
        set_probe_limit(10);
        assert_eq!(probe_limit(), 7);
        set_probe_limit(5);
        assert_eq!(probe_limit(), 5);
        set_probe_depth(3);
        assert_eq!(probe_depth(), 3);
        set_50mr_rule(false);
        assert!(!use_50mr());
        clear_tables();
        assert_eq!(probe_limit(), 7);
        assert_eq!(probe_depth(), 1);
        assert!(use_50mr());
    }

    #[test]
    fn no_tables_means_no_probe() {
        let _serial = SYZYGY_TEST_LOCK.lock().unwrap();
        clear_tables();
        assert_eq!(table_count(), 0);
        let board = BoardState::parse_fen(STARTING_FEN);
        assert!(probe_bound(&board, 0).is_none());
        let tiny = BoardState::parse_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1");
        assert!(probe_bound(&tiny, 0).is_none());
        assert!(root_move(&mut tiny.clone()).is_none());
        let empty = BoardState::new();
        assert!(probe_bound(&empty, 0).is_none());
        assert!(root_move(&mut empty.clone()).is_none());
    }

    #[test]
    fn bad_path_is_an_error() {
        let _serial = SYZYGY_TEST_LOCK.lock().unwrap();
        clear_tables();
        assert!(set_path("definitely/missing/tables").is_err());
        assert_eq!(table_count(), 0);
    }

    #[test]
    fn real_tables_probe_kqk() {
        let _serial = SYZYGY_TEST_LOCK.lock().unwrap();
        if !std::path::Path::new("tables/KQvK.rtbw").exists() {
            return;
        }
        clear_tables();
        let loaded = set_path("tables").expect("load tables");
        assert!(loaded > 0);
        assert_eq!(table_count(), loaded);

        let tiny = BoardState::parse_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1");
        assert_eq!(
            probe_bound(&tiny, 0),
            Some((0, TranspositionEntryType::Exact))
        );

        let start = BoardState::parse_fen(STARTING_FEN);
        assert!(probe_bound(&start, 0).is_none());
        assert!(root_move(&mut start.clone()).is_none());

        let win = BoardState::parse_fen("8/8/8/8/8/2K5/2Q5/7k w - - 0 1");
        assert_eq!(
            probe_bound(&win, 0),
            Some((TB_WIN, TranspositionEntryType::Beta))
        );
        let tb_move = root_move(&mut win.clone()).expect("table win has a move");
        assert!(win.is_legal(tb_move));

        assert_eq!(
            probe_bound(&win, 1),
            Some((TB_WIN - 1, TranspositionEntryType::Beta))
        );

        let stale = BoardState::parse_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
        assert_eq!(
            probe_bound(&stale, 0),
            Some((0, TranspositionEntryType::Exact))
        );
        assert!(root_move(&mut stale.clone()).is_none());

        let cursed = BoardState::parse_fen("8/8/8/8/8/2K5/2Q5/7k w - - 99 100");
        assert!(probe_bound(&cursed, 0).is_none());
        set_50mr_rule(false);
        assert_eq!(
            probe_bound(&cursed, 0),
            Some((TB_WIN, TranspositionEntryType::Beta))
        );
        set_50mr_rule(true);

        let blessed = BoardState::parse_fen("8/8/8/8/8/2k5/2q5/7K w - - 99 100");
        assert!(probe_bound(&blessed, 0).is_none());
        set_50mr_rule(false);
        assert_eq!(
            probe_bound(&blessed, 0),
            Some((-TB_WIN, TranspositionEntryType::Alpha))
        );

        clear_tables();
        assert_eq!(table_count(), 0);
    }
}
