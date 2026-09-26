use super::inference::eval_with_net;
use super::position::{kings_present, to_sf};
use super::*;

pub fn collect_threats_lazy(
    pos: &SfnnPosition,
    perspective: Side,
    out: &mut [usize; MAX_THREAT_ACTIVE],
    active: bool,
) -> usize {
    if !active || !maintenance_active() {
        return 0;
    }
    collect_threats(pos, perspective, out)
}

pub fn collect_pairs_lazy(
    pos: &SfnnPosition,
    perspective: Side,
    out: &mut [usize; MAX_PAIR_ACTIVE],
    active: bool,
) -> usize {
    if !active || !maintenance_active() {
        return 0;
    }
    collect_pairs(pos, perspective, out)
}

pub fn threats_cached_len(pos: &SfnnPosition, perspective: Side) -> usize {
    if !maintenance_active() {
        return 0;
    }
    let mut buf = [0usize; MAX_THREAT_ACTIVE];
    collect_threats(pos, perspective, &mut buf)
}

pub fn pairs_cached_len(pos: &SfnnPosition, perspective: Side) -> usize {
    if !maintenance_active() {
        return 0;
    }
    let mut buf = [0usize; MAX_PAIR_ACTIVE];
    collect_pairs(pos, perspective, &mut buf)
}
#[derive(Clone, Debug)]
pub struct Sfnn16Accs {
    pub halfka: [[i16; L1]; 2],
    pub psqt: [[i32; N_BUCKETS]; 2],
    pub threat_psqt: [[i32; N_BUCKETS]; 2],
    pub generation: u64,
}

impl Sfnn16Accs {
    pub fn empty() -> Self {
        Self {
            halfka: [[0i16; L1]; 2],
            psqt: [[0i32; N_BUCKETS]; 2],
            threat_psqt: [[0i32; N_BUCKETS]; 2],
            generation: 0,
        }
    }
}

pub struct LoadedNets {
    pub net: Sfnn16Net,
}

static NETS: RwLock<Option<Arc<LoadedNets>>> = RwLock::new(None);
static PREVIOUS_NETS: RwLock<Vec<Arc<LoadedNets>>> = RwLock::new(Vec::new());
static ACTIVE_NET: std::sync::atomic::AtomicPtr<LoadedNets> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());
static NETS_GEN: AtomicU64 = AtomicU64::new(0);
static PENDING_PATH: RwLock<Option<String>> = RwLock::new(None);
#[cfg(test)]
pub(crate) static EVAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(target_arch = "x86_64")]
static HAS_AVX2: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| is_x86_feature_detected!("avx2"));

#[inline(always)]
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        *HAS_AVX2
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
pub fn active_loaded_net() -> Option<&'static LoadedNets> {
    let ptr = ACTIVE_NET.load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        unsafe { Some(&*ptr) }
    }
}

#[cfg(test)]
pub(super) fn loaded_nets() -> Option<Arc<LoadedNets>> {
    NETS.read().ok().and_then(|guard| guard.clone())
}

pub fn try_load_default_path() -> Option<&'static str> {
    if maintenance_active() {
        return Some("active");
    }

    let candidates = [
        "models/whale_big_1.nnue",
        "models/whale_big.nnue",
        "models/whale_medium.nnue",
        "models/whale_small.nnue",
        "whale_big.nnue",
        "whale_medium.nnue",
        "whale_small.nnue",
        "v16/quantised.bin",
        "quantised.bin",
        "resources/sfnn16-big-checkpoint.bin",
    ];
    candidates
        .into_iter()
        .find(|&path| std::path::Path::new(path).exists() && load_net(path).is_ok())
}

pub fn try_load_default() -> bool {
    try_load_default_path().is_some()
}

#[inline(always)]
pub fn maintenance_active() -> bool {
    !ACTIVE_NET.load(Ordering::Relaxed).is_null()
}

fn current_gen() -> u64 {
    NETS_GEN.load(Ordering::SeqCst)
}

pub fn load_net(path: &str) -> Result<(), &'static str> {
    let net =
        Sfnn16Net::load_file(path, true, L1).or_else(|_| Sfnn16Net::load_file(path, false, L1))?;
    let loaded = Arc::new(LoadedNets { net });
    let raw_ptr = Arc::as_ptr(&loaded) as *mut LoadedNets;
    let mut nets_guard = NETS.write().unwrap_or_else(|p| p.into_inner());
    *nets_guard = Some(loaded.clone());
    if let Ok(mut prev) = PREVIOUS_NETS.write() {
        prev.push(loaded);
    }
    ACTIVE_NET.store(raw_ptr, Ordering::Release);
    if let Ok(mut guard) = PENDING_PATH.write() {
        *guard = Some(path.to_string());
    }
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    crate::eval::nnue::clear_eval_cache();
    Ok(())
}

pub fn unload_nets() {
    ACTIVE_NET.store(std::ptr::null_mut(), Ordering::Release);
    let mut nets_guard = NETS.write().unwrap_or_else(|p| p.into_inner());
    *nets_guard = None;
    if let Ok(mut guard) = PENDING_PATH.write() {
        *guard = None;
    }
    NETS_GEN.fetch_add(1, Ordering::SeqCst);
    crate::eval::nnue::clear_eval_cache();
}

pub fn resolve_model_path(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "<empty>" || trimmed.eq_ignore_ascii_case("embedded") {
        return None;
    }
    if std::path::Path::new(trimmed).exists() {
        return Some(trimmed.to_string());
    }
    let candidates = [
        format!("models/{trimmed}"),
        format!("models/{trimmed}.nnue"),
        format!("models/whale_{trimmed}.nnue"),
    ];
    for candidate in candidates {
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    Some(trimmed.to_string())
}

pub fn active_model_name() -> String {
    if let Ok(guard) = PENDING_PATH.read()
        && let Some(ref p) = *guard
    {
        let path = std::path::Path::new(p);
        return path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(p)
            .to_string();
    }
    "embedded".to_string()
}

pub fn set_eval_file(which: &str, path: &str) -> Result<&'static str, &'static str> {
    if which.eq_ignore_ascii_case("EvalFileSmall") {
        return Ok("EvalFileSmall is deprecated and ignored (SFNNv16 uses a single network)");
    }
    if which.eq_ignore_ascii_case("EvalFile") || which.eq_ignore_ascii_case("Model") {
        let resolved = resolve_model_path(path);
        if let Ok(mut guard) = PENDING_PATH.write() {
            *guard = resolved;
        }
    } else {
        return Err("unknown eval file option");
    }
    try_activate()
}

fn try_activate() -> Result<&'static str, &'static str> {
    let path = match PENDING_PATH.read() {
        Ok(guard) => guard.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };
    match path {
        Some(p) => match load_net(&p) {
            Ok(()) => Ok("SFNNv16 network activated"),
            Err(e) => Err(e),
        },
        None => {
            unload_nets();
            Ok("SFNNv16 network deactivated")
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scatter_halfka_add_avx2(acc: &mut [i16], w: &[i16]) {
    unsafe {
        use std::arch::x86_64::*;
        let a_ptr = acc.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m256i;
        let n = acc.len() / 64;
        for i in 0..n {
            let idx = i * 4;
            let va0 = _mm256_loadu_si256(a_ptr.add(idx));
            let vw0 = _mm256_loadu_si256(w_ptr.add(idx));
            let va1 = _mm256_loadu_si256(a_ptr.add(idx + 1));
            let vw1 = _mm256_loadu_si256(w_ptr.add(idx + 1));
            let va2 = _mm256_loadu_si256(a_ptr.add(idx + 2));
            let vw2 = _mm256_loadu_si256(w_ptr.add(idx + 2));
            let va3 = _mm256_loadu_si256(a_ptr.add(idx + 3));
            let vw3 = _mm256_loadu_si256(w_ptr.add(idx + 3));

            _mm256_storeu_si256(a_ptr.add(idx), _mm256_add_epi16(va0, vw0));
            _mm256_storeu_si256(a_ptr.add(idx + 1), _mm256_add_epi16(va1, vw1));
            _mm256_storeu_si256(a_ptr.add(idx + 2), _mm256_add_epi16(va2, vw2));
            _mm256_storeu_si256(a_ptr.add(idx + 3), _mm256_add_epi16(va3, vw3));
        }
        for i in (n * 4)..(acc.len() / 16) {
            let va = _mm256_loadu_si256(a_ptr.add(i));
            let vw = _mm256_loadu_si256(w_ptr.add(i));
            _mm256_storeu_si256(a_ptr.add(i), _mm256_add_epi16(va, vw));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scatter_halfka_sub_avx2(acc: &mut [i16], w: &[i16]) {
    unsafe {
        use std::arch::x86_64::*;
        let a_ptr = acc.as_mut_ptr() as *mut __m256i;
        let w_ptr = w.as_ptr() as *const __m256i;
        let n = acc.len() / 64;
        for i in 0..n {
            let idx = i * 4;
            let va0 = _mm256_loadu_si256(a_ptr.add(idx));
            let vw0 = _mm256_loadu_si256(w_ptr.add(idx));
            let va1 = _mm256_loadu_si256(a_ptr.add(idx + 1));
            let vw1 = _mm256_loadu_si256(w_ptr.add(idx + 1));
            let va2 = _mm256_loadu_si256(a_ptr.add(idx + 2));
            let vw2 = _mm256_loadu_si256(w_ptr.add(idx + 2));
            let va3 = _mm256_loadu_si256(a_ptr.add(idx + 3));
            let vw3 = _mm256_loadu_si256(w_ptr.add(idx + 3));

            _mm256_storeu_si256(a_ptr.add(idx), _mm256_sub_epi16(va0, vw0));
            _mm256_storeu_si256(a_ptr.add(idx + 1), _mm256_sub_epi16(va1, vw1));
            _mm256_storeu_si256(a_ptr.add(idx + 2), _mm256_sub_epi16(va2, vw2));
            _mm256_storeu_si256(a_ptr.add(idx + 3), _mm256_sub_epi16(va3, vw3));
        }
        for i in (n * 4)..(acc.len() / 16) {
            let va = _mm256_loadu_si256(a_ptr.add(i));
            let vw = _mm256_loadu_si256(w_ptr.add(i));
            _mm256_storeu_si256(a_ptr.add(i), _mm256_sub_epi16(va, vw));
        }
    }
}

pub(super) fn scatter_halfka(
    tr: &SfnnTransformer,
    l1: usize,
    feats: &[usize],
    acc: &mut [i16],
    psqt: &mut [i32; N_BUCKETS],
    sign: i16,
) {
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = has_avx2() && l1.is_multiple_of(16);
    #[cfg(not(target_arch = "x86_64"))]
    let use_avx2 = false;

    for &f in feats {
        debug_assert!(f < PSQ_DIMS);
        let base = f * l1;
        let w_slice = &tr.weights[base..base + l1];
        let acc_slice = &mut acc[..l1];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if sign == 1 {
                    scatter_halfka_add_avx2(acc_slice, w_slice);
                } else {
                    scatter_halfka_sub_avx2(acc_slice, w_slice);
                }
            }
        } else {
            if sign == 1 {
                for (a, &w) in acc_slice.iter_mut().zip(w_slice) {
                    *a = a.wrapping_add(w);
                }
            } else {
                for (a, &w) in acc_slice.iter_mut().zip(w_slice) {
                    *a = a.wrapping_sub(w);
                }
            }
        }
        let pbase = f * N_BUCKETS;
        let pw_slice = &tr.psqt_w[pbase..pbase + N_BUCKETS];
        if use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::*;
                let p_ptr = psqt.as_mut_ptr() as *mut __m256i;
                let pw_ptr = pw_slice.as_ptr() as *const __m256i;
                let vp = _mm256_loadu_si256(p_ptr);
                let vpw = _mm256_loadu_si256(pw_ptr);
                let res = if sign == 1 {
                    _mm256_add_epi32(vp, vpw)
                } else {
                    _mm256_sub_epi32(vp, vpw)
                };
                _mm256_storeu_si256(p_ptr, res);
            }
        } else {
            if sign == 1 {
                for b in 0..N_BUCKETS {
                    psqt[b] = psqt[b].wrapping_add(pw_slice[b]);
                }
            } else {
                for b in 0..N_BUCKETS {
                    psqt[b] = psqt[b].wrapping_sub(pw_slice[b]);
                }
            }
        }
    }
}

pub fn refresh_perspective(
    pos: &SfnnPosition,
    nets: &LoadedNets,
    perspective: Side,
    accs: &mut Sfnn16Accs,
) {
    let p = perspective as usize;
    let mut feats = Vec::new();
    append_halfka(pos, perspective, &mut feats);
    let slot = &mut accs.halfka[p];
    slot.copy_from_slice(&nets.net.transformer.bias);
    let mut ps = [0i32; N_BUCKETS];
    scatter_halfka(&nets.net.transformer, L1, &feats, slot, &mut ps, 1);
    accs.psqt[p] = ps;
}

#[derive(Clone)]
pub struct FinnyEntry {
    pub halfka: [i16; L1],
    pub psqt: [i32; N_BUCKETS],
    pub occupied: u64,
    pub white: u64,
    pub mapping: [u8; 64],
    pub generation: u64,
    pub valid: bool,
}

impl Default for FinnyEntry {
    fn default() -> Self {
        Self {
            halfka: [0; L1],
            psqt: [0; N_BUCKETS],
            occupied: 0,
            white: 0,
            mapping: [6; 64],
            generation: 0,
            valid: false,
        }
    }
}

std::thread_local! {
    static FINNY_CACHE: RefCell<Option<Box<[[FinnyEntry; 2]; 64]>>> = const { RefCell::new(None) };
}

pub fn update_perspective_finny_or_refresh(
    pos: &SfnnPosition,
    nets: &LoadedNets,
    perspective: Side,
    accs: &mut Sfnn16Accs,
) {
    let pi = perspective as usize;
    let ksq = pos.king_square(perspective);
    if ksq >= 64 {
        return;
    }
    let current_generation = current_gen();

    FINNY_CACHE.with(|cache| {
        let mut cache_borrow = cache.borrow_mut();
        let cache_box = cache_borrow.get_or_insert_with(|| {
            let mut v = Vec::with_capacity(64);
            for _ in 0..64 {
                v.push([FinnyEntry::default(), FinnyEntry::default()]);
            }
            v.into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!())
        });
        let entry = &mut cache_box[ksq][pi];

        if entry.valid && entry.generation == current_generation {
            let cur_occ = pos.occupied();
            let entry_occ = entry.occupied;
            let removed_bb = entry_occ & !cur_occ;
            let added_bb = cur_occ & !entry_occ;
            let common_bb = cur_occ & entry_occ;

            let mut diff_count = (removed_bb | added_bb).count_ones() as usize;
            if diff_count <= 8 {
                let mut changed_common = 0u64;
                let mut temp = common_bb;
                while temp != 0 {
                    let s = temp.trailing_zeros() as usize;
                    temp &= temp - 1;
                    let cur_w = (pos.white >> s) & 1;
                    let entry_w = (entry.white >> s) & 1;
                    if cur_w != entry_w || pos.mapping[s] != entry.mapping[s] {
                        changed_common |= 1u64 << s;
                        diff_count += 1;
                        if diff_count > 8 {
                            break;
                        }
                    }
                }

                if diff_count <= 8 {
                    accs.halfka[pi] = entry.halfka;
                    accs.psqt[pi] = entry.psqt;
                    let slot = &mut accs.halfka[pi];
                    let psqt = &mut accs.psqt[pi];

                    let mut del_bb = removed_bb | changed_common;
                    while del_bb != 0 {
                        let s = del_bb.trailing_zeros() as usize;
                        del_bb &= del_bb - 1;
                        let is_w = (entry.white >> s) & 1 == 1;
                        let side = if is_w { Side::White } else { Side::Black };
                        let pt = entry.mapping[s] as usize;
                        if pt <= 5 {
                            let piece = Piece::ALL[pt];
                            if let Some(f) = halfka_index(perspective, side, piece, s, ksq) {
                                scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                            }
                        }
                    }

                    let mut add_bb = added_bb | changed_common;
                    while add_bb != 0 {
                        let s = add_bb.trailing_zeros() as usize;
                        add_bb &= add_bb - 1;
                        let is_w = (pos.white >> s) & 1 == 1;
                        let side = if is_w { Side::White } else { Side::Black };
                        let pt = pos.mapping[s] as usize;
                        if pt <= 5 {
                            let piece = Piece::ALL[pt];
                            if let Some(f) = halfka_index(perspective, side, piece, s, ksq) {
                                scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, 1);
                            }
                        }
                    }

                    entry.halfka = accs.halfka[pi];
                    entry.psqt = accs.psqt[pi];
                    entry.occupied = cur_occ;
                    entry.white = pos.white;
                    entry.mapping = pos.mapping;
                    return;
                }
            }
        }

        refresh_perspective(pos, nets, perspective, accs);
        entry.halfka = accs.halfka[pi];
        entry.psqt = accs.psqt[pi];
        entry.occupied = pos.occupied();
        entry.white = pos.white;
        entry.mapping = pos.mapping;
        entry.generation = current_generation;
        entry.valid = true;
    });
}
pub fn refresh_all(pos: &SfnnPosition, accs: &mut Sfnn16Accs) {
    if !kings_present(pos) {
        return;
    }
    let nets = active_loaded_net();
    if let Some(nets) = nets {
        refresh_perspective(pos, nets, Side::White, accs);
        refresh_perspective(pos, nets, Side::Black, accs);
        accs.threat_psqt = [[0i32; N_BUCKETS]; 2];
        accs.generation = current_gen();
    } else {
        accs.generation = current_gen();
    }
}

pub fn ensure_fresh(pos: &SfnnPosition, accs: &mut Sfnn16Accs) {
    if accs.generation != current_gen() {
        refresh_all(pos, accs);
    }
}

pub fn apply_queued(
    pos: &SfnnPosition,
    accs: &mut Sfnn16Accs,
    adds: &[Option<SfnnEvent>],
    dels: &[Option<SfnnEvent>],
    king_moved: &[bool; 2],
) {
    let nets = active_loaded_net();
    if let Some(nets) = nets {
        for (pi, perspective) in [Side::White, Side::Black].iter().enumerate() {
            if king_moved[pi] {
                update_perspective_finny_or_refresh(pos, nets, *perspective, accs);
                continue;
            }
            let ksq = pos.king_square(*perspective);
            let slot = &mut accs.halfka[pi];
            let psqt = &mut accs.psqt[pi];
            for opt in dels {
                if let Some((sq_sf, side, piece)) = *opt
                    && let Some(f) = halfka_index(*perspective, side, piece, sq_sf, ksq)
                {
                    scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                }
            }
            for opt in adds {
                if let Some((sq_sf, side, piece)) = *opt
                    && let Some(f) = halfka_index(*perspective, side, piece, sq_sf, ksq)
                {
                    scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, 1);
                }
            }
        }
        accs.generation = current_gen();
    }
}
pub struct SfnnEval {
    pub psqt: i32,
    pub positional: i32,
    pub combined: i32,
    pub used_small: bool,
}

pub fn evaluate_nets(pos: &SfnnPosition, accs: &mut Sfnn16Accs, stm: Side) -> Option<SfnnEval> {
    if !kings_present(pos) {
        return None;
    }
    let nets = active_loaded_net()?;
    ensure_fresh(pos, accs);

    let bucket = material_bucket(pos.piece_count());
    let refs: [&[i16]; 2] = [&accs.halfka[0], &accs.halfka[1]];
    let psqt_refs: [&[i32; N_BUCKETS]; 2] = [&accs.psqt[0], &accs.psqt[1]];

    let (psqt, positional) = eval_with_net(pos, &nets.net, refs, psqt_refs, stm, bucket);
    let combined = if nets.net.is_rudi {
        positional
    } else {
        psqt + positional
    };

    Some(SfnnEval {
        psqt,
        positional,
        combined,
        used_small: false,
    })
}

pub fn ensure_sfnn16_fresh(board: &mut BoardState, pos: &SfnnPosition) {
    let idx = board.history.index;
    let current_generation = current_gen();
    let nets = match active_loaded_net() {
        Some(n) => n,
        None => return,
    };

    if !kings_present(pos) {
        return;
    }

    for (p, &perspective) in [Side::White, Side::Black].iter().enumerate() {
        if board.history.sfnn16_computed[idx][p]
            && board.history.sfnn16[idx].generation == current_generation
        {
            continue;
        }

        let mut ancestor = None;
        for anc in (0..idx).rev() {
            if board.history.sfnn16_pending[anc + 1].king_moved[p]
                || board.history.sfnn16_pending[anc + 1].overflowed
            {
                break;
            }
            if board.history.sfnn16_computed[anc][p]
                && board.history.sfnn16[anc].generation == current_generation
            {
                ancestor = Some(anc);
                break;
            }
        }

        if let Some(anc) = ancestor {
            let ksq = pos.king_square(perspective);
            for k in (anc + 1)..=idx {
                if !board.history.sfnn16_computed[k][p]
                    || board.history.sfnn16[k].generation != current_generation
                {
                    board.history.sfnn16[k].halfka[p] = board.history.sfnn16[k - 1].halfka[p];
                    board.history.sfnn16[k].psqt[p] = board.history.sfnn16[k - 1].psqt[p];

                    let pending = board.history.sfnn16_pending[k];
                    let slot = &mut board.history.sfnn16[k].halfka[p];
                    let psqt = &mut board.history.sfnn16[k].psqt[p];

                    for &ev in &pending.dels[..pending.n_dels] {
                        if let Some((sq_sf, side, piece)) = ev
                            && let Some(f) = halfka_index(perspective, side, piece, sq_sf, ksq)
                        {
                            scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, -1);
                        }
                    }

                    for &ev in &pending.adds[..pending.n_adds] {
                        if let Some((sq_sf, side, piece)) = ev
                            && let Some(f) = halfka_index(perspective, side, piece, sq_sf, ksq)
                        {
                            scatter_halfka(&nets.net.transformer, L1, &[f], slot, psqt, 1);
                        }
                    }

                    board.history.sfnn16_computed[k][p] = true;
                    board.history.sfnn16[k].generation = current_generation;
                }
            }
        } else {
            let accs = &mut board.history.sfnn16[idx];
            update_perspective_finny_or_refresh(pos, nets, perspective, accs);
            board.history.sfnn16_computed[idx][p] = true;
            board.history.sfnn16[idx].generation = current_generation;
        }
    }
}

pub fn evaluate_board(board: &mut BoardState) -> Option<i16> {
    if !maintenance_active() {
        return None;
    }
    let pos = SfnnPosition::from_board(board);
    ensure_sfnn16_fresh(board, &pos);
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];

    evaluate_nets(&pos, accs, board.side_to_move).map(|e| e.combined.clamp(-29000, 29000) as i16)
}

pub fn evaluate_board_detailed(board: &mut BoardState) -> Option<SfnnEval> {
    if !maintenance_active() {
        return None;
    }
    let pos = SfnnPosition::from_board(board);
    ensure_sfnn16_fresh(board, &pos);
    let idx = board.history.index;
    if idx >= board.history.sfnn16.len() {
        return None;
    }
    let accs = &mut board.history.sfnn16[idx];
    evaluate_nets(&pos, accs, board.side_to_move)
}

pub type SfnnEvent = (usize, Side, Piece);

#[derive(Clone, Copy, Debug, Default)]
pub struct SfnnPending {
    pub adds: [Option<SfnnEvent>; 8],
    pub dels: [Option<SfnnEvent>; 8],
    pub n_adds: usize,
    pub n_dels: usize,
    pub king_moved: [bool; 2],
    pub overflowed: bool,
}

impl SfnnPending {
    pub fn push_add(&mut self, ev: SfnnEvent) {
        if self.n_adds < self.adds.len() {
            self.adds[self.n_adds] = Some(ev);
            self.n_adds += 1;
        } else {
            self.overflowed = true;
        }
    }

    pub fn push_del(&mut self, ev: SfnnEvent) {
        if self.n_dels < self.dels.len() {
            self.dels[self.n_dels] = Some(ev);
            self.n_dels += 1;
        } else {
            self.overflowed = true;
        }
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

pub fn note_add(pending: &mut SfnnPending, square: Square, side: Side, piece: Piece) {
    if piece == Piece::King {
        pending.king_moved[side as usize] = true;
    }
    pending.push_add((to_sf(square as usize), side, piece));
}

pub fn note_remove(pending: &mut SfnnPending, square: Square, side: Side, piece: Piece) {
    if piece == Piece::King {
        pending.king_moved[side as usize] = true;
    }
    pending.push_del((to_sf(square as usize), side, piece));
}

pub fn flush_pending(pos: &SfnnPosition, accs: &mut Sfnn16Accs, pending: &mut SfnnPending) {
    if !maintenance_active() {
        pending.clear();
        return;
    }
    if !kings_present(pos) {
        pending.clear();
        return;
    }
    if accs.generation != current_gen() {
        refresh_all(pos, accs);
        pending.clear();
        return;
    }
    if pending.overflowed {
        refresh_all(pos, accs);
        pending.clear();
        return;
    }
    apply_queued(
        pos,
        accs,
        &pending.adds[..pending.n_adds],
        &pending.dels[..pending.n_dels],
        &pending.king_moved,
    );
    pending.clear();
}

pub fn trainer_features(
    pieces: &[u64; 6],
    white: u64,
    black: u64,
    mapping: &[u8; 64],
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let pos = SfnnPosition {
        pieces: *pieces,
        white,
        black,
        mapping: *mapping,
    };
    let mut w_h = Vec::new();
    let mut b_h = Vec::new();
    append_halfka(&pos, Side::White, &mut w_h);
    append_halfka(&pos, Side::Black, &mut b_h);
    let mut w_t = Vec::new();
    let mut b_t = Vec::new();
    append_threats(&pos, Side::White, &mut w_t);
    append_threats(&pos, Side::Black, &mut b_t);
    append_pairs(&pos, Side::White, &mut w_t);
    append_pairs(&pos, Side::Black, &mut b_t);
    for v in w_t.iter_mut().chain(b_t.iter_mut()) {
        *v -= PSQ_DIMS;
    }
    (w_h, b_h, w_t, b_t)
}
