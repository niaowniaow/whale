use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_uchar};
use std::sync::OnceLock;
use std::sync::atomic::AtomicBool;

use crate::board::state::BoardState;
use crate::common::move_list::MoveList;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::search::search_state::SearchState;

pub struct WhaleHandle {
    board: BoardState,
    state: SearchState,
}

static INIT_ONCE: OnceLock<()> = OnceLock::new();

fn ensure_init() {
    INIT_ONCE.get_or_init(|| {
        crate::init();
    });
}

fn handle_ref(h: *mut WhaleHandle) -> Option<&'static mut WhaleHandle> {
    if h.is_null() {
        return None;
    }
    unsafe { Some(&mut *h) }
}

fn read_cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p).to_str().ok().map(|s| s.to_owned()) }
}

fn resolve_move(board: &mut BoardState, uci: &str) -> Option<Move> {
    let parsed = Move::parse_long_algebraic(uci)?;
    let mut list = MoveList::new();
    board.generate_moves(&mut list);
    for m in list.iter() {
        if m.mv.source == parsed.source && m.mv.target == parsed.target {
            if parsed.move_type == MoveType::Quiet {
                let cand = m.mv;
                board.make_move(cand);
                let bad = board.is_in_check(board.side_to_move.other());
                board.unmake_move(cand);
                if !bad {
                    return Some(cand);
                }
            } else if (m.mv.move_type.value() & !8) == parsed.move_type.value() {
                let cand = m.mv;
                board.make_move(cand);
                let bad = board.is_in_check(board.side_to_move.other());
                board.unmake_move(cand);
                if !bad {
                    return Some(cand);
                }
            }
        }
    }
    None
}

fn move_to_uci(m: Move) -> String {
    let mut s = format!("{}{}", m.source, m.target);
    if let Some(p) = m.promotion_char() {
        s.push(p);
    }
    s
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_version() -> *const c_char {
    static V: &[u8] = b"Whale 0.1.0\0";
    V.as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_create() -> *mut WhaleHandle {
    ensure_init();
    let h = Box::new(WhaleHandle {
        board: BoardState::default(),
        state: SearchState::new(),
    });
    Box::into_raw(h)
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_destroy(h: *mut WhaleHandle) {
    if h.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(h));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_set_startpos(h: *mut WhaleHandle) -> c_int {
    match handle_ref(h) {
        None => -1,
        Some(x) => {
            x.board = BoardState::default();
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_set_fen(h: *mut WhaleHandle, fen: *const c_char) -> c_int {
    match (handle_ref(h), read_cstr(fen)) {
        (Some(x), Some(s)) => {
            if s.split(' ').count() < 4 {
                return -2;
            }
            x.board = BoardState::parse_fen(&s);
            0
        }
        _ => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_push_uci(h: *mut WhaleHandle, uci: *const c_char) -> c_int {
    match (handle_ref(h), read_cstr(uci)) {
        (Some(x), Some(s)) => match resolve_move(&mut x.board, s.trim()) {
            Some(m) => {
                x.board.make_move(m);
                0
            }
            None => -2,
        },
        _ => -1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_go_depth(
    h: *mut WhaleHandle,
    depth: c_uchar,
    out: *mut c_char,
    out_cap: usize,
) -> c_int {
    let x = match handle_ref(h) {
        None => return -1,
        Some(v) => v,
    };
    if out.is_null() || out_cap < 6 {
        return -1;
    }
    let cancel = AtomicBool::new(false);
    let mut debug = false;
    let d = depth.clamp(1, 64);
    let best = x
        .board
        .find_best_move(d, &cancel, &mut debug, &mut x.state, 1);
    if best == Move::NO_MOVE {
        return -2;
    }
    let s = move_to_uci(best);
    let bytes = s.as_bytes();
    if bytes.len() + 1 > out_cap {
        return -3;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out, bytes.len());
        *out.add(bytes.len()) = 0;
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_set_syzygy_path(path: *const c_char) -> c_int {
    match read_cstr(path) {
        None => -1,
        Some(s) => match crate::syzygy::set_path(&s) {
            Ok(_) => 0,
            Err(_) => -2,
        },
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn whale_tb_probe_wdl(h: *mut WhaleHandle) -> c_int {
    let x = match handle_ref(h) {
        None => return -100,
        Some(v) => v,
    };
    match crate::syzygy::probe_wdl(&x.board) {
        None => 100,
        Some(v) => v as c_int,
    }
}
