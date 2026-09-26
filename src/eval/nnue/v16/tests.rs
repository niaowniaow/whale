use super::arch::{LEB128_MAGIC, combine_hash, div_trunc, relu_hash, rotl1, unscramble_table};
use super::eval::{loaded_nets, scatter_halfka};
use super::inference::{
    add_threat_w_i16_avx2, eval_with_net, fc0_avx2, pairwise_transform_no_threats_avx2,
    pairwise_transform_rudi_avx2, pairwise_transform_threats_avx2, propagate,
};
use super::loader::{
    decode_leb128_i16, decode_leb128_i32, decode_leb128_i64, read_i32_le, read_leb128_section,
    read_rudi_i8, read_rudi_i16, read_rudi_i32, read_u32_le,
};
use super::position::{halfka_orient, kings_present, sf_piece_code, to_sf};
use super::threats::{
    knight_attacks_sf, pseudo_attacks_sf, sf_piece_type, threat_make_index, threat_orient,
};
use super::*;

#[cfg(target_arch = "x86_64")]
#[test]
fn regression_fc0_full_range_dot() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let mut arch = SfnnArch {
        fc0_bias: [0; FC0_OUT],
        fc0_w: vec![0; 32 * FC0_OUT],
        fc1_bias: [0; FC1_OUT],
        fc1_w: vec![0; FC1_IN * FC1_OUT],
        fc2_bias: 0,
        fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
        is_rudi: true,
    };
    arch.fc0_bias[0] = 3;
    arch.fc0_bias[1] = -5;
    arch.fc0_w[..2].fill(127);
    arch.fc0_w[32..34].fill(-128);
    let mut input = [0u8; 32];
    input[..2].fill(255);
    let mut actual = [0; FC0_OUT];
    unsafe { fc0_avx2(&arch, &input, &mut actual) };
    assert_eq!(actual[0], 3 + 255 * 127 + 255 * 127);
    assert_eq!(actual[1], -5 - 255 * 128 - 255 * 128);
    assert!(actual[2..].iter().all(|&value| value == 0));
}

#[test]
fn regression_rudi_division_identity() {
    for product in 0..=255 * 255 {
        assert_eq!((product * 257 + 257) >> 16, product / 255);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn regression_pairwise_clipping_parity() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let half = L1 / 2;
    let mut base = [0i16; L1];
    let mut threats = [0i32; L1];
    let mut actual = [0u8; L1 / 2];
    let values = [i16::MIN, -256, -1, 0, 1, 127, 254, 255, 256, i16::MAX];
    for a in 0..256 {
        for j in 0..half {
            base[j] = a;
            base[j + half] = (j % 256) as i16;
        }
        unsafe { pairwise_transform_rudi_avx2(&base, &threats, &mut actual, half, false) };
        for (j, &value) in actual.iter().enumerate() {
            assert_eq!(value, (i32::from(a) * (j % 256) as i32 / 255) as u8);
        }
    }
    for j in 0..L1 {
        base[j] = values[j % values.len()];
        threats[j] = ((j * 37) % 1024) as i32 - 512;
    }
    for with_threats in [false, true] {
        for rudi in [false, true] {
            unsafe {
                if rudi {
                    pairwise_transform_rudi_avx2(&base, &threats, &mut actual, half, with_threats);
                } else if with_threats {
                    pairwise_transform_threats_avx2(&base, &threats, &mut actual, half, 255);
                } else {
                    pairwise_transform_no_threats_avx2(&base, &mut actual, half, 255);
                }
            }
            for j in 0..half {
                let extra = |i| if with_threats { threats[i] } else { 0 };
                let a = (i32::from(base[j]) + extra(j)).clamp(0, 255);
                let b = (i32::from(base[j + half]) + extra(j + half)).clamp(0, 255);
                assert_eq!(actual[j], (a * b / if rudi { 255 } else { 512 }) as u8);
            }
        }
    }
}

#[test]
fn regression_rudi_converter_header() {
    let mut bytes = Vec::with_capacity(181_011_116);
    bytes.extend_from_slice(b"RUDI");
    bytes.extend_from_slice(&181_011_108u32.to_le_bytes());
    bytes.resize(181_011_116, 0);
    let net = Sfnn16Net::load_rudi(&bytes).unwrap();
    assert!(net.use_threats);
    assert_eq!(net.transformer.threat_w_i16.len(), THREAT_DIMS * L1);
    assert_eq!(net.transformer.pair_w_i16.len(), PAIR_DIMS * L1);
    bytes.push(0);
    assert!(Sfnn16Net::load_rudi(&bytes).is_err());
}

#[test]
fn regression_rudi_rejects_unknown_header() {
    let mut bytes = vec![0; 64 + 181_011_108];
    bytes[..4].copy_from_slice(b"RUDI");
    assert!(Sfnn16Net::load_rudi(&bytes).is_err());
}

#[test]
fn regression_rudi_dynamic_features() {
    let _eval_guard = EVAL_TEST_LOCK.lock().unwrap();
    let was_active = maintenance_active();
    if !was_active {
        load_net("models/whale_big_1.nnue").expect("bundled net loads");
    }
    let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let mut arch = SfnnArch {
        fc0_bias: [0; FC0_OUT],
        fc0_w: vec![0; 2 * FC0_OUT],
        fc1_bias: [0; FC1_OUT],
        fc1_w: vec![0; FC1_IN * FC1_OUT],
        fc2_bias: 0,
        fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
        is_rudi: true,
    };
    arch.fc0_w[0] = 64;
    arch.fc1_w[FC0_OUT] = 64;
    arch.fc2_w[0] = 64;
    let mut net = Sfnn16Net {
        l1: 2,
        use_threats: true,
        is_rudi: true,
        transformer: SfnnTransformer {
            threat_w_i16: vec![0; THREAT_DIMS * 2],
            pair_w_i16: vec![0; PAIR_DIMS * 2],
            ..Default::default()
        },
        stacks: vec![arch],
    };
    let mut threats = Vec::new();
    let mut pairs = Vec::new();
    append_threats(&pos, Side::White, &mut threats);
    append_pairs(&pos, Side::White, &mut pairs);
    assert!(!threats.is_empty());
    assert!(!pairs.is_empty());
    let base = [0, 255];
    let psqt = [0; N_BUCKETS];
    let eval = |net: &Sfnn16Net| eval_with_net(&pos, net, [&base; 2], [&psqt; 2], Side::White, 0);
    let baseline = eval(&net);
    let t = (threats[0] - PSQ_DIMS) * 2;
    net.transformer.threat_w_i16[t] = 255;
    assert_ne!(eval(&net), baseline);
    net.transformer.threat_w_i16[t] = 0;
    let p = (pairs[0] - PSQ_DIMS - THREAT_DIMS) * 2;
    net.transformer.pair_w_i16[p] = 255;
    assert_ne!(eval(&net), baseline);
    net.use_threats = false;
    assert_eq!(eval(&net), baseline);
    if !was_active {
        unload_nets();
    }
}

#[test]
fn test_eval_breakdown() {
    let _eval_guard = EVAL_TEST_LOCK.lock().unwrap();
    let mut loaded_any = false;
    for net_path in ["v16/nn-1a298aa575a0.nnue", "v16/whale.nnue"] {
        println!("=== Testing net: {} ===", net_path);
        if !std::path::Path::new(net_path).exists() {
            println!("Missing {net_path}, skipping (network files are git-ignored fixtures)");
            continue;
        }
        if let Err(e) = load_net(net_path) {
            println!("Failed to load {net_path}: {e}");
            continue;
        }
        loaded_any = true;
        let is_official = net_path.contains("nn-1a298aa575a0");
        let fens = [
            (
                "startpos",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ),
            (
                "1. e4 e5 2. Ke2 (B)",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 1 2",
            ),
            (
                "1. e4 e5 2. Nf3 (B)",
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
            ),
            (
                "1. e4 (B)",
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            ),
        ];

        for (name, fen) in fens {
            let board = BoardState::parse_fen(fen);
            let pos = SfnnPosition::from_board(&board);
            let mut accs = Sfnn16Accs::empty();
            refresh_all(&pos, &mut accs);
            if let Some(eval) = evaluate_nets(&pos, &mut accs, board.side_to_move) {
                if is_official && name == "startpos" {
                    assert_eq!(eval.combined, -1);
                }
                let cp = (eval.combined as i64 * 100) / 256;
                println!(
                    "{:<22} | STM: {:?} | psqt: {:>6} | pos: {:>6} | comb: {:>6} | cp: {:>5}",
                    name, board.side_to_move, eval.psqt, eval.positional, eval.combined, cp
                );
            }
        }
    }
    if !loaded_any {
        println!("No network fixtures present; skipping breakdown assertions");
        return;
    }
    let nets = loaded_nets().expect("network should be active after load");
    let stack = &nets.net.stacks[7];
    println!(
        "fc0_bias[30] = {}, fc0_bias[31] = {}",
        stack.fc0_bias[30], stack.fc0_bias[31]
    );
    let w30 = &stack.fc0_w[30 * 1024..(30 + 1) * 1024];
    let w31 = &stack.fc0_w[31 * 1024..(31 + 1) * 1024];
    let sum_w30_us: i64 = w30[..512].iter().map(|&x| x as i64).sum();
    let sum_w30_them: i64 = w30[512..].iter().map(|&x| x as i64).sum();
    let sum_w31_us: i64 = w31[..512].iter().map(|&x| x as i64).sum();
    let sum_w31_them: i64 = w31[512..].iter().map(|&x| x as i64).sum();
    println!("w30 us sum = {}, them sum = {}", sum_w30_us, sum_w30_them);
    println!("w31 us sum = {}, them sum = {}", sum_w31_us, sum_w31_them);
}

#[test]
fn bench_eval_speed() {
    if !std::path::Path::new("v16/nn-1a298aa575a0.nnue").exists() {
        return;
    }
    load_net("v16/nn-1a298aa575a0.nnue").unwrap();
    let board = BoardState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let mut accs = Sfnn16Accs::empty();
    refresh_all(&pos, &mut accs);
    let start = std::time::Instant::now();
    let iters = 10_000;
    for _ in 0..iters {
        let _ = evaluate_nets(&pos, &mut accs, board.side_to_move);
    }
    let el = start.elapsed();
    println!(
        "BASELINE: {} evals in {:?} ({:.2} us/eval, {:.0} NPS)",
        iters,
        el,
        (el.as_micros() as f64) / (iters as f64),
        (iters as f64) / el.as_secs_f64()
    );
}

#[test]
fn square_helpers_use_sf_numbering() {
    assert_eq!(to_sf(52), 12);
    assert_eq!(to_sf(to_sf(36)), 36);

    assert_eq!(halfka_orient(0), 7);
    assert_eq!(halfka_orient(4), 0);
    assert_eq!(sf_piece_code(Side::White, Piece::Pawn), 1);
    assert_eq!(sf_piece_code(Side::White, Piece::King), 6);
    assert_eq!(sf_piece_code(Side::Black, Piece::Pawn), 9);
    assert_eq!(sf_piece_code(Side::Black, Piece::King), 14);
}

#[test]
fn material_bucket_boundaries() {
    assert_eq!(material_bucket(0), 0);
    assert_eq!(material_bucket(1), 0);
    assert_eq!(material_bucket(4), 0);
    assert_eq!(material_bucket(5), 1);
    assert_eq!(material_bucket(32), 7);
    assert_eq!(material_bucket(1000), N_BUCKETS - 1);
}

#[test]
fn simple_eval_symmetry_and_material() {
    use crate::common::helpers::STARTING_FEN;

    let board = BoardState::parse_fen(STARTING_FEN);
    let pos = SfnnPosition::from_board(&board);
    assert_eq!(pos.piece_count(), 32);
    assert_eq!(pos.occupied(), pos.white | pos.black);
    assert_eq!(pos.king_square(Side::White), 4);
    assert_eq!(simple_eval(&pos, Side::White), 0);
    assert_eq!(simple_eval(&pos, Side::Black), 0);

    let mut up = BoardState::parse_fen(STARTING_FEN);
    up.add_piece(Square::E4, Side::White, Piece::Queen, false);
    let up_pos = SfnnPosition::from_board(&up);
    assert_eq!(simple_eval(&up_pos, Side::White), QUEEN_VALUE);
    assert_eq!(simple_eval(&up_pos, Side::Black), -QUEEN_VALUE);
}

#[test]
fn hash_utils_are_deterministic() {
    assert_eq!(combine_hash(&[]), 0);
    assert_eq!(rotl1(1), 2);
    assert_eq!(rotl1(u32::MAX), u32::MAX);
    assert_ne!(affine_hash(32, 0), affine_hash(31, 0));
    assert_ne!(affine_hash(32, 0), affine_hash(32, 1));
    assert_ne!(relu_hash(0), relu_hash(1));
    assert_eq!(arch_hash(1024), arch_hash(1024));
    assert_ne!(transformer_hash(true, 1024), transformer_hash(false, 1024));
    assert_eq!(
        network_hash(true, 1024),
        transformer_hash(true, 1024) ^ arch_hash(1024)
    );
}

#[test]
fn div_trunc_matches_rust_division() {
    assert_eq!(div_trunc(7, 2), 3);
    assert_eq!(div_trunc(-7, 2), -3);
    assert_eq!(div_trunc(7, -2), -3);
    assert_eq!(div_trunc(0, 5), 0);
}

#[test]
fn read_le_primitives() {
    let mut pos = 0;
    assert_eq!(read_u32_le(&[1, 0, 0, 0, 9], &mut pos), Ok(1));
    assert_eq!(pos, 4);
    assert_eq!(read_u32_le(&[9], &mut pos), Err("truncated u32"));
    let mut neg = 0;
    assert_eq!(read_i32_le(&[0xFF, 0xFF, 0xFF, 0xFF], &mut neg), Ok(-1));

    let mut data = Vec::new();
    data.extend_from_slice(LEB128_MAGIC);
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&[0x01, 0x7F]);
    let mut spos = 0;
    assert_eq!(
        read_leb128_section(&data, &mut spos, 2, decode_leb128_i16),
        Ok(vec![1, -1])
    );
    assert!(read_leb128_section(&[], &mut 0, 1, decode_leb128_i16).is_err());
    assert!(read_leb128_section(&[0u8; 17], &mut 0, 1, decode_leb128_i16).is_err());
}

#[test]
fn decode_leb128_values_and_errors() {
    let mut pos = 0;
    assert_eq!(decode_leb128_i64(&[0x05], &mut pos, 64), Ok(5));
    let mut pos = 0;
    assert_eq!(decode_leb128_i64(&[0xAC, 0x02], &mut pos, 64), Ok(300));
    let mut pos = 0;
    assert_eq!(decode_leb128_i64(&[0x7F], &mut pos, 64), Ok(-1));
    let mut pos = 0;
    assert_eq!(decode_leb128_i32(&[0xFF, 0x7F], &mut pos), Ok(-1));
    let mut pos = 0;
    assert!(decode_leb128_i64(&[0x80], &mut pos, 64).is_err());
    let mut pos = 0;
    assert!(decode_leb128_i16(&[0x80, 0x80, 0x80, 0x80], &mut pos).is_err());
}

#[test]
fn read_rudi_primitives() {
    let mut off = 0;
    assert_eq!(read_rudi_i8(&[0x01, 0xFF], &mut off, 2), Ok(vec![1, -1]));
    assert_eq!(off, 2);
    assert!(read_rudi_i8(&[0x01], &mut 0, 2).is_err());

    let mut off = 0;
    assert_eq!(
        read_rudi_i16(&[0x01, 0x00, 0xFF, 0xFF], &mut off, 2),
        Ok(vec![1, -1])
    );
    assert!(read_rudi_i16(&[0x01], &mut 0, 1).is_err());

    let mut off = 0;
    assert_eq!(
        read_rudi_i32(&[0x01, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF], &mut off, 2),
        Ok(vec![1, -1])
    );
    assert!(read_rudi_i32(&[], &mut 0, 1).is_err());

    let mut off = 0;
    assert_eq!(read_rudi_i8(&[0x01], &mut off, 0), Ok(vec![]));
}

#[test]
fn unscramble_table_is_permutation() {
    let mut inv = unscramble_table(2, 8);
    inv.sort_unstable();
    assert_eq!(inv, (0..16).collect::<Vec<_>>());
    assert_eq!(unscramble_table(1, 4), vec![0, 1, 2, 3]);
}

#[test]
fn sfnn_pending_queue_and_overflow() {
    let ev = (12usize, Side::White, Piece::Pawn);
    let mut pending = SfnnPending::default();
    for _ in 0..8 {
        pending.push_add(ev);
    }
    assert_eq!(pending.n_adds, 8);
    assert!(!pending.overflowed);
    pending.push_add(ev);
    assert!(pending.overflowed);

    let mut pending = SfnnPending::default();
    for _ in 0..8 {
        pending.push_del(ev);
    }
    pending.push_del(ev);
    assert!(pending.overflowed);
    pending.clear();
    assert_eq!(pending.n_adds, 0);
    assert_eq!(pending.n_dels, 0);
    assert!(!pending.overflowed);

    let mut pending = SfnnPending::default();
    note_add(&mut pending, Square::E2, Side::White, Piece::King);
    assert!(pending.king_moved[Side::White as usize]);
    assert_eq!(pending.adds[0], Some((12, Side::White, Piece::King)));
    note_remove(&mut pending, Square::E7, Side::Black, Piece::Pawn);
    assert!(!pending.king_moved[Side::Black as usize]);
    assert_eq!(
        pending.dels[0],
        Some((to_sf(Square::E7 as usize), Side::Black, Piece::Pawn))
    );
}

#[test]
fn load_paths_reject_garbage() {
    assert!(Sfnn16Net::load_bytes(&[], true, L1).is_err());
    assert!(Sfnn16Net::load_bytes(&[0, 0, 0, 0], true, L1).is_err());
    assert!(Sfnn16Net::load_rudi(&[]).is_err());
    assert!(Sfnn16Net::load_file("definitely/missing.nnue", true, L1).is_err());
    assert!(Sfnn16Net::load_file("definitely/missing.nnue", false, L1).is_err());
    assert!(set_eval_file("Bogus", "x").is_err());
    assert!(set_eval_file("EvalFileSmall", "x").is_ok());
}

#[test]
fn halfka_features_cover_all_pieces() {
    use crate::common::helpers::STARTING_FEN;

    assert_eq!(
        halfka_index(Side::White, Side::White, Piece::None, 0, 4),
        None
    );

    let king_idx = halfka_index(Side::White, Side::White, Piece::King, 4, 4).unwrap();
    assert!(king_idx >= 640);

    let board = BoardState::parse_fen(STARTING_FEN);
    let pos = SfnnPosition::from_board(&board);
    for perspective in [Side::White, Side::Black] {
        let mut out = Vec::new();
        append_halfka(&pos, perspective, &mut out);
        assert_eq!(out.len(), 32);
        let mut again = Vec::new();
        append_halfka(&pos, perspective, &mut again);
        assert_eq!(out, again);
    }

    assert!({
        let mut out = Vec::new();
        append_halfka(
            &SfnnPosition {
                pieces: [0; 6],
                white: 0,
                black: 0,
                mapping: [6; 64],
            },
            Side::White,
            &mut out,
        );
        out.is_empty()
    });
}

#[test]
fn threat_and_pair_collectors_agree() {
    let quiet = SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    let mut counted = 0;
    for_each_pair(&quiet, |_, _, _, _| counted += 1);
    assert_eq!(counted, 0);

    let board = BoardState::parse_fen(
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    );
    let pos = SfnnPosition::from_board(&board);
    for perspective in [Side::White, Side::Black] {
        let mut listed = Vec::new();
        append_threats(&pos, perspective, &mut listed);
        let mut buf = [0usize; MAX_THREAT_ACTIVE];
        let n = collect_threats(&pos, perspective, &mut buf);
        assert_eq!(&listed, &buf[..n]);
        assert!(!listed.is_empty());
        for &idx in &listed {
            assert!((PSQ_DIMS..PSQ_DIMS + THREAT_DIMS).contains(&idx));
        }

        let mut plist = Vec::new();
        append_pairs(&pos, perspective, &mut plist);
        let mut pbuf = [0usize; MAX_PAIR_ACTIVE];
        let pn = collect_pairs(&pos, perspective, &mut pbuf);
        assert_eq!(&plist, &pbuf[..pn]);
        for &idx in &plist {
            assert!((PSQ_DIMS + THREAT_DIMS..PSQ_DIMS + THREAT_DIMS + PAIR_DIMS).contains(&idx));
        }
    }

    let rel = pair_index_for(Side::White, Side::White, 12, 20, Side::White, 4).unwrap();
    let abs = pair_make_index(Side::White, Side::White, 12, 20, Side::White, 4).unwrap();
    assert_eq!(rel, abs - (PSQ_DIMS + THREAT_DIMS));
    assert!(rel < PAIR_DIMS);

    let (w_h, b_h, w_t, b_t) = trainer_features(&pos.pieces, pos.white, pos.black, &pos.mapping);
    assert_eq!(w_h.len(), 32);
    assert_eq!(b_h.len(), 32);
    assert!(!w_t.is_empty());
    assert!(!b_t.is_empty());
}

#[test]
fn whale_family_models_load() {
    for name in ["whale_big", "whale_medium", "whale_small"] {
        let path = format!("models/{name}.nnue");
        if !std::path::Path::new(&path).exists() {
            println!("Missing {path}, skipping");
            continue;
        }
        let net = Sfnn16Net::load_file(&path, true, L1)
            .or_else(|_| Sfnn16Net::load_file(&path, false, L1))
            .unwrap_or_else(|e| panic!("{path} failed to load: {e}"));
        assert!(!net.stacks.is_empty(), "{path} has no stacks");
        println!(
            "{name}: l1={} rudi={} threats={} stacks={}",
            net.l1,
            net.is_rudi,
            net.use_threats,
            net.stacks.len()
        );
    }
}

#[test]
fn test_finny_cache_consistency() {
    let _eval_guard = EVAL_TEST_LOCK.lock().unwrap();
    if !try_load_default() {
        return;
    }
    let nets = loaded_nets().unwrap();
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
        "8/8/8/4k3/8/8/4K3/8 w - - 0 1",
    ];
    for fen in fens {
        let board = BoardState::parse_fen(fen);
        let pos = SfnnPosition::from_board(&board);
        for perspective in [Side::White, Side::Black] {
            let mut acc_direct = Sfnn16Accs::empty();
            refresh_perspective(&pos, &nets, perspective, &mut acc_direct);

            let mut acc_finny = Sfnn16Accs::empty();
            update_perspective_finny_or_refresh(&pos, &nets, perspective, &mut acc_finny);
            assert_eq!(
                acc_finny.halfka[perspective as usize],
                acc_direct.halfka[perspective as usize]
            );
            assert_eq!(
                acc_finny.psqt[perspective as usize],
                acc_direct.psqt[perspective as usize]
            );

            let mut acc_finny_cached = Sfnn16Accs::empty();
            update_perspective_finny_or_refresh(&pos, &nets, perspective, &mut acc_finny_cached);
            assert_eq!(
                acc_finny_cached.halfka[perspective as usize],
                acc_direct.halfka[perspective as usize]
            );
            assert_eq!(
                acc_finny_cached.psqt[perspective as usize],
                acc_direct.psqt[perspective as usize]
            );
        }
    }
}

#[test]
fn halfka_black_perspective_and_ownership() {
    let own = halfka_index(Side::White, Side::White, Piece::Pawn, 12, 4).unwrap();
    let enemy = halfka_index(Side::White, Side::Black, Piece::Pawn, 12, 4).unwrap();
    assert_eq!(enemy.wrapping_sub(own), 64);
    assert!(own < PSQ_DIMS && enemy < PSQ_DIMS);

    let black = halfka_index(Side::Black, Side::Black, Piece::Pawn, 12, 4).unwrap();
    assert!(black < PSQ_DIMS);
    assert_ne!(own, black);

    let left = halfka_index(Side::White, Side::White, Piece::Knight, 0, 0).unwrap();
    let right = halfka_index(Side::White, Side::White, Piece::Knight, 0, 7).unwrap();
    assert_ne!(left, right);

    for &pt in &Piece::ALL {
        if pt == Piece::None {
            continue;
        }
        assert!(halfka_index(Side::White, Side::White, pt, 27, 4).is_some());
        assert!(halfka_index(Side::Black, Side::Black, pt, 27, 60).is_some());
    }
}

#[test]
fn append_halfka_skips_empty_mapping() {
    let pos = SfnnPosition {
        pieces: [0; 6],
        white: 1,
        black: 0,
        mapping: [6; 64],
    };
    let mut out = Vec::new();
    append_halfka(&pos, Side::White, &mut out);
    assert!(out.is_empty());
}

#[test]
fn threat_orient_covers_all_sides() {
    assert_eq!(threat_orient(Side::White, 0), 0);
    assert_eq!(threat_orient(Side::White, 7), 7);
    assert_eq!(threat_orient(Side::Black, 0), 56);
    assert_eq!(threat_orient(Side::Black, 7), 63);
    assert_eq!(threat_orient(Side::Both, 0), 0);
    assert_eq!(threat_orient(Side::Both, 63), 0);
    assert_eq!(sf_piece_type(0), 1);
    assert_eq!(sf_piece_type(5), 6);
}

#[test]
fn for_each_threat_covers_piece_types() {
    let cases = [
        "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1",
        "4k3/8/8/3n4/8/2N5/8/4K3 w - - 0 1",
        "4k3/8/8/3b4/4B3/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/3R4/8/8/3rK3 w - - 0 1",
        "4k3/8/8/3q4/4Q3/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/3p4/4K3 w - - 0 1",
    ];
    for fen in cases {
        let pos = SfnnPosition::from_board(&BoardState::parse_fen(fen));
        let mut count = 0;
        for_each_threat(&pos, |_, _, _, _| count += 1);
        assert!(count > 0, "no threat edges for {fen}");
    }

    let quiet = SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    let mut count = 0;
    for_each_threat(&quiet, |_, _, _, _| count += 1);
    assert_eq!(count, 0);
}

#[test]
fn threat_index_excluded_semi_and_perspective() {
    let ksq = 4;

    assert_eq!(threat_index_for(Side::White, 1, 12, 19, 9, ksq), None);

    assert_eq!(threat_index_for(Side::White, 2, 10, 4, 6, ksq), None);

    assert_eq!(threat_index_for(Side::White, 2, 10, 20, 10, ksq), None);
    let some = threat_index_for(Side::White, 2, 20, 10, 10, ksq);
    assert!(some.is_some());
    assert!(some.unwrap() < THREAT_DIMS);

    assert!(threat_index_for(Side::White, 2, 10, 20, 13, ksq).is_some());
    assert!(threat_index_for(Side::White, 2, 20, 10, 13, ksq).is_some());

    assert!(threat_index_for(Side::Black, 1, 12, 19, 9, 60).is_none());
    assert!(
        threat_index_for(Side::Black, 2, 20, 10, 10, 60).is_some()
            || threat_index_for(Side::Black, 2, 10, 20, 10, 60).is_some()
    );
}

#[test]
fn pair_orient_and_enumeration_variants() {
    for &persp in &[Side::White, Side::Black] {
        for &ksq in &[0usize, 7, 56, 63] {
            for &(c, pc) in &[
                (Side::White, Side::White),
                (Side::White, Side::Black),
                (Side::Black, Side::Black),
            ] {
                let abs = pair_make_index(persp, c, 12, 20, pc, ksq).unwrap();
                assert!(
                    (PSQ_DIMS + THREAT_DIMS..PSQ_DIMS + THREAT_DIMS + PAIR_DIMS).contains(&abs)
                );
                let rel = pair_index_for(persp, c, 12, 20, pc, ksq).unwrap();
                assert_eq!(rel, abs - (PSQ_DIMS + THREAT_DIMS));
            }
        }
    }

    let near =
        SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/2P5/3P4/4K3 w - - 0 1"));
    let mut n = 0;
    for_each_pair(&near, |_, _, _, _| n += 1);
    assert!(n > 0);
    let far = SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/P6P/4K3 w - - 0 1"));
    let mut m = 0;
    for_each_pair(&far, |_, _, _, _| m += 1);
    assert_eq!(m, 0);

    let mixed =
        SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/2p5/3P4/4K3 w - - 0 1"));
    let mut k = 0;
    for_each_pair(&mixed, |_, _, _, _| k += 1);
    assert!(k > 0);
}

#[test]
fn read_leb128_section_extra_errors() {
    let mut only_magic = Vec::new();
    only_magic.extend_from_slice(LEB128_MAGIC);
    assert!(read_leb128_section(&only_magic, &mut 0, 1, decode_leb128_i16).is_err());

    assert!(read_leb128_section(&[0u8; 17], &mut 0, 1, decode_leb128_i16).is_err());

    let mut data = Vec::new();
    data.extend_from_slice(LEB128_MAGIC);
    data.extend_from_slice(&1u32.to_le_bytes());
    data.push(0x80);
    assert!(read_leb128_section(&data, &mut 0, 1, decode_leb128_i16).is_err());
}

#[test]
fn decode_leb128_overflow_and_truncation() {
    assert!(decode_leb128_i64(&[], &mut 0, 32).is_err());

    assert!(decode_leb128_i64(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80], &mut 0, 32).is_err());

    assert_eq!(decode_leb128_i64(&[0xAC, 0x02], &mut 0, 32), Ok(300));

    assert_eq!(decode_leb128_i16(&[0x7E], &mut 0), Ok(-2));
}

fn build_valid_arch_bytes(fc0_in: usize) -> Vec<u8> {
    let hash = arch_hash(fc0_in as u32);
    let padded0 = fc0_in.next_multiple_of(32);
    let mut data = Vec::new();
    data.extend_from_slice(&hash.to_le_bytes());
    for _ in 0..FC0_OUT {
        data.extend_from_slice(&0i32.to_le_bytes());
    }
    data.extend(std::iter::repeat_n(0u8, FC0_OUT * padded0));
    for _ in 0..FC1_OUT {
        data.extend_from_slice(&0i32.to_le_bytes());
    }
    data.extend(std::iter::repeat_n(0u8, FC1_OUT * 64));
    data.extend_from_slice(&0i32.to_le_bytes());
    data.extend(std::iter::repeat_n(0u8, FC0_OUT * 2 + FC1_OUT * 2));
    data
}

#[test]
fn sfnn_arch_load_success_and_truncations() {
    let data = build_valid_arch_bytes(32);
    let mut pos = 0;
    let arch = SfnnArch::load(&data, &mut pos, 32, arch_hash(32)).unwrap();
    assert_eq!(pos, data.len());
    assert!(!arch.is_rudi);

    assert!(SfnnArch::load(&data, &mut 0, 32, 0x12345678).is_err());

    for cut in [1usize, 100, 500, 1500, 3000] {
        let mut p = 0;
        assert!(
            SfnnArch::load(&data[..data.len() - cut], &mut p, 32, arch_hash(32)).is_err(),
            "cut {cut} should fail"
        );
    }
}

#[test]
fn sfnn16_load_bytes_early_errors() {
    assert!(Sfnn16Net::load_bytes(&[0, 0, 0, 0], true, 2).is_err());

    let mut bad_hash = Vec::new();
    bad_hash.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
    bad_hash.extend_from_slice(&0u32.to_le_bytes());
    assert!(Sfnn16Net::load_bytes(&bad_hash, true, 2).is_err());

    assert!(Sfnn16Net::load_bytes(&bad_hash, false, 2).is_err());

    let mut desc = Vec::new();
    desc.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
    desc.extend_from_slice(&network_hash(true, 2).to_le_bytes());
    desc.extend_from_slice(&100u32.to_le_bytes());
    assert!(Sfnn16Net::load_bytes(&desc, true, 2).is_err());

    let mut th = Vec::new();
    th.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
    th.extend_from_slice(&network_hash(true, 2).to_le_bytes());
    th.extend_from_slice(&0u32.to_le_bytes());
    th.extend_from_slice(&0u32.to_le_bytes());
    assert!(Sfnn16Net::load_bytes(&th, true, 2).is_err());

    let mut ok = Vec::new();
    ok.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
    ok.extend_from_slice(&network_hash(true, 2).to_le_bytes());
    ok.extend_from_slice(&0u32.to_le_bytes());
    ok.extend_from_slice(&transformer_hash(true, 2).to_le_bytes());
    assert!(Sfnn16Net::load_bytes(&ok, true, 2).is_err());

    assert!(set_eval_file("Bogus", "x").is_err());
    assert!(set_eval_file("EvalFileSmall", "x").is_ok());
}

#[test]
fn load_rudi_extra_rejections() {
    let mut bad = Vec::new();
    bad.extend_from_slice(b"RUDI");
    bad.extend_from_slice(&123u32.to_le_bytes());
    bad.extend(std::iter::repeat_n(0u8, 123));
    assert!(Sfnn16Net::load_rudi(&bad).is_err());

    let mut short = Vec::new();
    short.extend_from_slice(b"RUDI");
    short.extend_from_slice(&181_011_108u32.to_le_bytes());
    short.extend(std::iter::repeat_n(0u8, 100));
    assert!(Sfnn16Net::load_rudi(&short).is_err());

    assert!(Sfnn16Net::load_rudi(&[1u8; 100]).is_err());

    assert!(read_rudi_i16(&[0x01], &mut 0, 1).is_err());
    assert!(read_rudi_i32(&[0x01, 0x02], &mut 0, 1).is_err());
}

#[test]
fn propagate_zero_nets_are_deterministic() {
    let mk = |rudi: bool| SfnnArch {
        fc0_bias: [0; FC0_OUT],
        fc0_w: vec![0; 32 * FC0_OUT],
        fc1_bias: [0; FC1_OUT],
        fc1_w: vec![0; FC1_IN * FC1_OUT],
        fc2_bias: 0,
        fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
        is_rudi: rudi,
    };
    let input = [0u8; 32];

    assert_eq!(propagate(&mk(true), 32, &input), -4480);

    assert_eq!(propagate(&mk(false), 32, &input), 0);

    assert_eq!(
        propagate(&mk(true), 32, &input),
        propagate(&mk(true), 32, &input)
    );
}

#[test]
fn scatter_add_sub_roundtrip() {
    let l1 = 32usize;
    let tr = SfnnTransformer {
        bias: vec![0; l1],
        weights: vec![3i16; 2 * l1],
        threat_w: Vec::new(),
        threat_w_i16: Vec::new(),
        psqt_w: vec![5i32; 2 * N_BUCKETS],
        threat_psqt_w: Vec::new(),
        pair_w: Vec::new(),
        pair_w_i16: Vec::new(),
        pair_psqt_w: Vec::new(),
    };
    let mut acc = [0i16; 32];
    let mut psqt = [0i32; N_BUCKETS];
    scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, 1);
    assert!(acc.iter().all(|&v| v == 3));
    assert!(psqt.iter().all(|&v| v == 5));
    scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, -1);
    assert!(acc.iter().all(|&v| v == 0));
    assert!(psqt.iter().all(|&v| v == 0));

    let l1s = 15usize;
    let trs = SfnnTransformer {
        bias: vec![0; l1s],
        weights: vec![2i16; 2 * l1s],
        threat_w: Vec::new(),
        threat_w_i16: Vec::new(),
        psqt_w: vec![1i32; 2 * N_BUCKETS],
        threat_psqt_w: Vec::new(),
        pair_w: Vec::new(),
        pair_w_i16: Vec::new(),
        pair_psqt_w: Vec::new(),
    };
    let mut accs = [0i16; 15];
    let mut ps = [0i32; N_BUCKETS];
    scatter_halfka(&trs, l1s, &[1], &mut accs, &mut ps, 1);
    scatter_halfka(&trs, l1s, &[1], &mut accs, &mut ps, -1);
    assert!(accs.iter().all(|&v| v == 0));
    assert!(ps.iter().all(|&v| v == 0));
}

#[test]
fn kingless_paths_are_safe_without_global_mutation() {
    let board = BoardState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    assert!(!kings_present(&pos));

    let mut accs = Sfnn16Accs::empty();
    assert!(evaluate_nets(&pos, &mut accs, Side::White).is_none());

    let mut accs = Sfnn16Accs::empty();
    refresh_all(&pos, &mut accs);
    ensure_fresh(&pos, &mut accs);

    let mut pending = SfnnPending::default();
    note_add(&mut pending, Square::E2, Side::White, Piece::Pawn);
    flush_pending(&pos, &mut accs, &mut pending);
    assert_eq!(pending.n_adds, 0);

    let start = BoardState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let spos = SfnnPosition::from_board(&start);
    let mut accs = Sfnn16Accs::empty();
    apply_queued(&spos, &mut accs, &[], &[], &[false, false]);
    apply_queued(&spos, &mut accs, &[], &[], &[true, true]);
}

#[test]
fn eval_with_net_small_nets_cover_all_branches() {
    let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let base = [10i16, 20];
    let psqt = [3i32; N_BUCKETS];
    let mk = |rudi: bool, threats: bool| Sfnn16Net {
        l1: 2,
        use_threats: threats,
        is_rudi: rudi,
        transformer: SfnnTransformer {
            bias: Vec::new(),
            weights: Vec::new(),
            threat_w: if !rudi && threats {
                vec![0i8; THREAT_DIMS * 2]
            } else {
                Vec::new()
            },
            threat_w_i16: vec![0i16; THREAT_DIMS * 2],
            psqt_w: Vec::new(),
            threat_psqt_w: if !rudi && threats {
                vec![0i32; THREAT_DIMS * N_BUCKETS]
            } else {
                Vec::new()
            },
            pair_w: Vec::new(),
            pair_w_i16: if rudi {
                vec![0i16; PAIR_DIMS * 2]
            } else {
                Vec::new()
            },
            pair_psqt_w: Vec::new(),
        },
        stacks: vec![SfnnArch {
            fc0_bias: [0; FC0_OUT],
            fc0_w: vec![0; 2 * FC0_OUT],
            fc1_bias: [0; FC1_OUT],
            fc1_w: vec![0; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
            is_rudi: rudi,
        }],
    };

    let rudi_t = mk(true, true);
    let rudi_nt = mk(true, false);
    let (psqt_t, pos_t) = eval_with_net(&pos, &rudi_t, [&base; 2], [&psqt; 2], Side::White, 0);
    let (psqt_nt, pos_nt) = eval_with_net(&pos, &rudi_nt, [&base; 2], [&psqt; 2], Side::White, 0);
    assert_eq!(
        (psqt_t, pos_t),
        eval_with_net(&pos, &rudi_t, [&base; 2], [&psqt; 2], Side::White, 0)
    );

    let sf_nt = mk(false, false);
    let _ = eval_with_net(&pos, &sf_nt, [&base; 2], [&psqt; 2], Side::White, 0);
    let _ = eval_with_net(&pos, &sf_nt, [&base; 2], [&psqt; 2], Side::Black, 0);

    assert_eq!(
        (psqt_nt, pos_nt),
        eval_with_net(&pos, &rudi_nt, [&base; 2], [&psqt; 2], Side::White, 0)
    );

    let (w_h, b_h, w_t, b_t) = trainer_features(&pos.pieces, pos.white, pos.black, &pos.mapping);
    assert!(!w_h.is_empty() && !b_h.is_empty());
    assert!(!w_t.is_empty() && !b_t.is_empty());
    assert_eq!(
        simple_eval(&pos, Side::White),
        -simple_eval(&pos, Side::Black)
    );
}

fn push_leb128_zeros_for_test(buf: &mut Vec<u8>, count: usize) {
    buf.extend_from_slice(LEB128_MAGIC);
    buf.extend_from_slice(&(count as u32).to_le_bytes());
    buf.extend(std::iter::repeat_n(0u8, count));
}

fn build_tiny_no_threats_for_test(l1: usize, version: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    push_leb128_zeros_for_test(&mut buf, l1);
    push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * l1);
    push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * N_BUCKETS);
    let arch = build_valid_arch_bytes(l1);
    for _ in 0..N_BUCKETS {
        buf.extend_from_slice(&arch);
    }
    buf
}

fn build_tiny_combined_threats_for_test(l1: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
    buf.extend_from_slice(&network_hash(true, l1 as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&transformer_hash(true, l1 as u32).to_le_bytes());
    push_leb128_zeros_for_test(&mut buf, l1);
    push_leb128_zeros_for_test(&mut buf, (THREAT_DIMS + PSQ_DIMS) * l1);
    push_leb128_zeros_for_test(&mut buf, (THREAT_DIMS + PSQ_DIMS) * N_BUCKETS);
    let arch = build_valid_arch_bytes(l1);
    for _ in 0..N_BUCKETS {
        buf.extend_from_slice(&arch);
    }
    buf
}

fn build_tiny_sf17_for_test(l1: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&SF_FILE_VERSION.to_le_bytes());
    buf.extend_from_slice(&network_hash(true, l1 as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&transformer_hash(true, l1 as u32).to_le_bytes());
    push_leb128_zeros_for_test(&mut buf, l1);
    buf.extend(std::iter::repeat_n(0u8, THREAT_DIMS * l1));
    push_leb128_zeros_for_test(&mut buf, THREAT_DIMS * N_BUCKETS);
    buf.extend(std::iter::repeat_n(0u8, PAIR_DIMS * l1));
    push_leb128_zeros_for_test(&mut buf, PAIR_DIMS * N_BUCKETS);
    push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * l1);
    push_leb128_zeros_for_test(&mut buf, PSQ_DIMS * N_BUCKETS);
    let arch = build_valid_arch_bytes(l1);
    for _ in 0..N_BUCKETS {
        buf.extend_from_slice(&arch);
    }
    buf
}

fn dummy_full_loaded_nets_for_test() -> LoadedNets {
    LoadedNets {
        net: Sfnn16Net {
            l1: L1,
            use_threats: false,
            is_rudi: false,
            transformer: SfnnTransformer {
                bias: vec![7i16; L1],
                weights: vec![0i16; PSQ_DIMS * L1],
                threat_w: Vec::new(),
                threat_w_i16: Vec::new(),
                psqt_w: vec![0i32; PSQ_DIMS * N_BUCKETS],
                threat_psqt_w: Vec::new(),
                pair_w: Vec::new(),
                pair_w_i16: Vec::new(),
                pair_psqt_w: Vec::new(),
            },
            stacks: Vec::new(),
        },
    }
}

#[test]
fn propagate_rudi_scalar_clamp_determinism() {
    let l1 = 15usize;
    let mk = |bias: i32| SfnnArch {
        fc0_bias: [bias; FC0_OUT],
        fc0_w: vec![0i8; l1 * FC0_OUT],
        fc1_bias: [0i32; FC1_OUT],
        fc1_w: vec![1i8; FC1_IN * FC1_OUT],
        fc2_bias: 0,
        fc2_w: {
            let mut w = [0i8; FC0_OUT * 2 + FC1_OUT * 2];
            w[0] = 64;
            w
        },
        is_rudi: true,
    };
    let input = vec![10u8; l1];
    let low = propagate(&mk(-1_000_000), l1, &input);
    let mid = propagate(&mk(8160), l1, &input);
    let high = propagate(&mk(1_000_000), l1, &input);
    assert_eq!(low, propagate(&mk(-1_000_000), l1, &input));
    assert_eq!(mid, propagate(&mk(8160), l1, &input));
    assert_eq!(high, propagate(&mk(1_000_000), l1, &input));
    for v in [low, mid, high] {
        assert!((-464_000..=464_000).contains(&v), "rudi {v} out of range");
    }
    assert!(low < mid, "low {low} should be < mid {mid}");
    assert!(mid <= high, "mid {mid} should be <= high {high}");
    let mut arch = mk(8160);
    arch.fc0_w.iter_mut().for_each(|w| *w = 2);
    let out = propagate(&arch, l1, &input);
    assert!((-464_000..=464_000).contains(&out));
    assert_eq!(out, propagate(&arch, l1, &input));
}

#[test]
fn propagate_nonrudi_scalar_skip_and_loops() {
    let l1 = 15usize;
    let mut arch = SfnnArch {
        fc0_bias: [0i32; FC0_OUT],
        fc0_w: vec![0i8; l1 * FC0_OUT],
        fc1_bias: [0i32; FC1_OUT],
        fc1_w: vec![0i8; FC1_IN * FC1_OUT],
        fc2_bias: 0,
        fc2_w: [0i8; FC0_OUT * 2 + FC1_OUT * 2],
        is_rudi: false,
    };
    arch.fc0_bias[30] = 1000;
    arch.fc0_bias[31] = 200;
    let input = vec![0u8; l1];
    let out = propagate(&arch, l1, &input);
    assert_eq!(out, ((800i64 * 600 * 16) / (128 * 64 * 2)) as i32);
    assert_eq!(out, propagate(&arch, l1, &input));
    arch.fc0_bias[30] = 200;
    arch.fc0_bias[31] = 1000;
    let out2 = propagate(&arch, l1, &input);
    assert_ne!(out, out2);
    arch.fc0_bias = [50_000; FC0_OUT];
    arch.fc1_w.iter_mut().for_each(|w| *w = 1);
    arch.fc2_w.iter_mut().for_each(|w| *w = 1);
    let big = propagate(&arch, l1, &input);
    assert_eq!(big, propagate(&arch, l1, &input));
    arch.fc0_bias = [-50_000; FC0_OUT];
    let small = propagate(&arch, l1, &input);
    assert_ne!(big, small);
    arch.fc0_bias = [0; FC0_OUT];
    arch.fc0_w.iter_mut().for_each(|w| *w = 3);
    arch.fc1_w.iter_mut().for_each(|w| *w = 2);
    arch.fc2_w.iter_mut().for_each(|w| *w = 1);
    arch.fc2_bias = 5;
    let input2 = vec![7u8; l1];
    let full = propagate(&arch, l1, &input2);
    assert_eq!(full, propagate(&arch, l1, &input2));
}

#[test]
fn propagate_l1_1024_large_determinism() {
    let l1 = L1;
    let mk = |rudi: bool| SfnnArch {
        fc0_bias: [3i32; FC0_OUT],
        fc0_w: {
            let mut w = vec![0i8; l1 * FC0_OUT];
            w[0] = 1;
            w[1] = -1;
            w[l1] = 2;
            w
        },
        fc1_bias: [1i32; FC1_OUT],
        fc1_w: {
            let mut w = vec![0i8; FC1_IN * FC1_OUT];
            w[0] = 1;
            w[FC1_IN] = -2;
            w
        },
        fc2_bias: 7,
        fc2_w: {
            let mut w = [0i8; FC0_OUT * 2 + FC1_OUT * 2];
            w[0] = 1;
            w[64] = -1;
            w[127] = 2;
            w
        },
        is_rudi: rudi,
    };
    let input = vec![11u8; l1];
    for rudi in [true, false] {
        let arch = mk(rudi);
        let a = propagate(&arch, l1, &input);
        let b = propagate(&arch, l1, &input);
        assert_eq!(a, b);
        let zero_input = vec![0u8; l1];
        let c = propagate(&arch, l1, &zero_input);
        assert_eq!(c, propagate(&arch, l1, &zero_input));
    }
}

#[test]
fn sfnn16_load_bytes_tiny_no_threats_success() {
    for version in [SF_FILE_VERSION, SF17_FILE_VERSION] {
        let buf = build_tiny_no_threats_for_test(2, version);
        let net = Sfnn16Net::load_bytes(&buf, false, 2).unwrap();
        assert_eq!(net.l1, 2);
        assert!(!net.use_threats);
        assert!(!net.is_rudi);
        assert_eq!(net.transformer.bias.len(), 2);
        assert_eq!(net.transformer.weights.len(), PSQ_DIMS * 2);
        assert_eq!(net.transformer.psqt_w.len(), PSQ_DIMS * N_BUCKETS);
        assert_eq!(net.stacks.len(), N_BUCKETS);
    }
    let mut buf = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
    buf.push(0);
    assert_eq!(
        Sfnn16Net::load_bytes(&buf, false, 2).unwrap_err(),
        "trailing data after network"
    );
    let mut bad = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
    let arch_len = build_valid_arch_bytes(2).len();
    let arch_start = bad.len() - arch_len * N_BUCKETS;
    bad[arch_start] ^= 0xFF;
    assert!(Sfnn16Net::load_bytes(&bad, false, 2).is_err());
    let mut bad_magic = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
    bad_magic[16] ^= 0xFF;
    assert!(Sfnn16Net::load_bytes(&bad_magic, false, 2).is_err());
    let good = build_tiny_no_threats_for_test(2, SF_FILE_VERSION);
    for cut in [20usize, 100, 1000, 50_000, 150_000] {
        if cut < good.len() {
            assert!(
                Sfnn16Net::load_bytes(&good[..good.len() - cut], false, 2).is_err(),
                "cut {cut} should fail"
            );
        }
    }
}

#[test]
fn sfnn16_load_bytes_threat_combined_success() {
    let buf = build_tiny_combined_threats_for_test(2);
    let net = Sfnn16Net::load_bytes(&buf, true, 2).unwrap();
    assert_eq!(net.l1, 2);
    assert!(net.use_threats);
    assert_eq!(net.transformer.weights.len(), PSQ_DIMS * 2);
    assert_eq!(net.transformer.threat_w.len(), THREAT_DIMS * 2);
    assert_eq!(net.transformer.threat_psqt_w.len(), THREAT_DIMS * N_BUCKETS);
    assert_eq!(net.stacks.len(), N_BUCKETS);
    let mut trailing = buf.clone();
    trailing.push(0);
    assert!(Sfnn16Net::load_bytes(&trailing, true, 2).is_err());
    for cut in [20usize, 500, 50_000, 300_000] {
        assert!(
            Sfnn16Net::load_bytes(&buf[..buf.len() - cut], true, 2).is_err(),
            "cut {cut} should fail"
        );
    }
    let mut bad_arch = buf.clone();
    let arch_start = buf.len() - build_valid_arch_bytes(2).len();
    bad_arch[arch_start] ^= 0xFF;
    assert!(Sfnn16Net::load_bytes(&bad_arch, true, 2).is_err());
}

#[test]
fn sfnn16_load_bytes_sf17_success() {
    let buf = build_tiny_sf17_for_test(2);
    let net = Sfnn16Net::load_bytes(&buf, true, 2).unwrap();
    assert_eq!(net.l1, 2);
    assert!(net.use_threats);
    assert_eq!(net.transformer.threat_w.len(), THREAT_DIMS * 2);
    assert_eq!(net.transformer.pair_w.len(), PAIR_DIMS * 2);
    assert_eq!(net.transformer.weights.len(), PSQ_DIMS * 2);
    assert_eq!(net.stacks.len(), N_BUCKETS);
    let pair_offset = {
        let mut pos = 0;
        pos += 4 + 4 + 4;
        pos += LEB128_MAGIC.len() + 4 + 2;
        pos += THREAT_DIMS * 2;
        pos += LEB128_MAGIC.len() + 4 + THREAT_DIMS * N_BUCKETS;
        pos
    };
    let truncated = buf[..pair_offset + 10].to_vec();
    assert_eq!(
        Sfnn16Net::load_bytes(&truncated, true, 2).unwrap_err(),
        "truncated pair weights"
    );
    let mut trailing = buf.clone();
    trailing.extend_from_slice(&[0u8; 4]);
    assert!(Sfnn16Net::load_bytes(&trailing, true, 2).is_err());
}

#[test]
fn load_file_delegates_without_global() {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let rudi_path = dir.join(format!("whale_v16_test_rudi_{pid}.tmp"));
    let normal_path = dir.join(format!("whale_v16_test_normal_{pid}.tmp"));
    let mut bad_rudi = Vec::new();
    bad_rudi.extend_from_slice(b"RUDI");
    bad_rudi.extend_from_slice(&123u32.to_le_bytes());
    bad_rudi.extend(std::iter::repeat_n(0u8, 123));
    std::fs::write(&rudi_path, &bad_rudi).unwrap();
    let rudi_str = rudi_path.to_string_lossy().into_owned();
    assert!(Sfnn16Net::load_file(&rudi_str, true, L1).is_err());
    std::fs::write(&normal_path, b"bad!").unwrap();
    let normal_str = normal_path.to_string_lossy().into_owned();
    assert!(Sfnn16Net::load_file(&normal_str, true, L1).is_err());
    assert!(Sfnn16Net::load_file(&normal_str, false, 2).is_err());
    let _ = std::fs::remove_file(&rudi_path);
    let _ = std::fs::remove_file(&normal_path);
}

#[test]
fn eval_with_net_sf_threats_covers_branches() {
    let _eval_guard = EVAL_TEST_LOCK.lock().unwrap();
    let was_active = maintenance_active();
    if !was_active {
        load_net("models/whale_big_1.nnue").expect("bundled net loads");
    }
    let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let base = [10i16, 20];
    let psqt = [3i32; N_BUCKETS];
    let mk = |with_pairs: bool| Sfnn16Net {
        l1: 2,
        use_threats: true,
        is_rudi: false,
        transformer: SfnnTransformer {
            bias: Vec::new(),
            weights: Vec::new(),
            threat_w: vec![0i8; THREAT_DIMS * 2],
            threat_w_i16: Vec::new(),
            psqt_w: Vec::new(),
            threat_psqt_w: vec![0i32; THREAT_DIMS * N_BUCKETS],
            pair_w: if with_pairs {
                vec![0i8; PAIR_DIMS * 2]
            } else {
                Vec::new()
            },
            pair_w_i16: Vec::new(),
            pair_psqt_w: if with_pairs {
                vec![0i32; PAIR_DIMS * N_BUCKETS]
            } else {
                Vec::new()
            },
        },
        stacks: vec![
            SfnnArch {
                fc0_bias: [0; FC0_OUT],
                fc0_w: vec![127i8; 2 * FC0_OUT],
                fc1_bias: [0; FC1_OUT],
                fc1_w: vec![127i8; FC1_IN * FC1_OUT],
                fc2_bias: 0,
                fc2_w: [127i8; FC0_OUT * 2 + FC1_OUT * 2],
                is_rudi: false,
            };
            1
        ],
    };
    let mut net_pairs = mk(true);
    let net_nopairs = mk(false);
    let base_pairs = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
    let base_nopairs = eval_with_net(&pos, &net_nopairs, [&base; 2], [&psqt; 2], Side::White, 0);
    assert_eq!(
        base_pairs,
        eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0)
    );
    let mut threats = [0usize; MAX_THREAT_ACTIVE];
    let nt = collect_threats(&pos, Side::White, &mut threats);
    assert!(nt > 0);
    let t = (threats[0] - PSQ_DIMS) * 2;
    net_pairs.transformer.threat_w[t] = 127;
    net_pairs.transformer.threat_w[t + 1] = 127;
    let after_w = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
    assert_ne!(base_pairs, after_w);
    net_pairs.transformer.threat_w[t] = 0;
    net_pairs.transformer.threat_w[t + 1] = 0;
    net_pairs.transformer.threat_psqt_w[(threats[0] - PSQ_DIMS) * N_BUCKETS] = 64;
    let after_psqt = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
    assert_ne!(base_pairs, after_psqt);
    let mut pairs = [0usize; MAX_PAIR_ACTIVE];
    let np = collect_pairs(&pos, Side::White, &mut pairs);
    if np > 0 {
        let pt = (pairs[0] - PSQ_DIMS - THREAT_DIMS) * 2;
        net_pairs.transformer.pair_w[pt] = 127;
        net_pairs.transformer.pair_w[pt + 1] = 127;
        let after_pair = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::White, 0);
        assert_ne!(base_pairs, after_pair);
    }
    let _ = eval_with_net(&pos, &net_pairs, [&base; 2], [&psqt; 2], Side::Black, 0);
    let _ = eval_with_net(&pos, &net_nopairs, [&base; 2], [&psqt; 2], Side::Black, 0);
    let _ = base_nopairs;
    if !was_active {
        unload_nets();
    }
}

#[test]
fn eval_with_net_l1_1024_kings_only() {
    let board = BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let base = [0i16; L1];
    let psqt = [0i32; N_BUCKETS];
    let mk = |rudi: bool, threats: bool| Sfnn16Net {
        l1: L1,
        use_threats: threats,
        is_rudi: rudi,
        transformer: SfnnTransformer {
            bias: Vec::new(),
            weights: Vec::new(),
            threat_w: Vec::new(),
            threat_w_i16: Vec::new(),
            psqt_w: Vec::new(),
            threat_psqt_w: Vec::new(),
            pair_w: Vec::new(),
            pair_w_i16: Vec::new(),
            pair_psqt_w: Vec::new(),
        },
        stacks: vec![
            SfnnArch {
                fc0_bias: [0; FC0_OUT],
                fc0_w: vec![0; L1 * FC0_OUT],
                fc1_bias: [0; FC1_OUT],
                fc1_w: vec![0; FC1_IN * FC1_OUT],
                fc2_bias: 0,
                fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
                is_rudi: rudi,
            };
            1
        ],
    };
    for rudi in [true, false] {
        for threats in [true, false] {
            let net = mk(rudi, threats);
            for stm in [Side::White, Side::Black] {
                let a = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], stm, 0);
                let b = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], stm, 0);
                assert_eq!(a, b);
            }
        }
    }
}

#[test]
fn refresh_perspective_manual_full_net() {
    let loaded = dummy_full_loaded_nets_for_test();
    let board = BoardState::parse_fen("K7/8/8/8/8/8/8/k7 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let mut accs = Sfnn16Accs::empty();
    refresh_perspective(&pos, &loaded, Side::White, &mut accs);
    assert!(accs.halfka[0].iter().all(|&v| v == 7));
    assert!(accs.psqt[0].iter().all(|&v| v == 0));
    refresh_perspective(&pos, &loaded, Side::Black, &mut accs);
    assert!(accs.halfka[1].iter().all(|&v| v == 7));
    let mut accs2 = Sfnn16Accs::empty();
    refresh_perspective(&pos, &loaded, Side::White, &mut accs2);
    assert_eq!(accs.halfka[0], accs2.halfka[0]);
    assert_eq!(accs.psqt[0], accs2.psqt[0]);
    let board2 = BoardState::parse_fen("K7/8/8/8/2P5/8/8/k7 w - - 0 1");
    let pos2 = SfnnPosition::from_board(&board2);
    refresh_perspective(&pos2, &loaded, Side::White, &mut accs2);
    assert!(accs2.halfka[0].iter().all(|&v| v == 7));
}

#[test]
fn finny_incremental_manual_full_net() {
    let loaded = dummy_full_loaded_nets_for_test();
    let fens = [
        "K7/8/8/8/8/8/8/k7 w - - 0 1",
        "K7/8/8/8/8/8/1P6/k7 w - - 0 1",
        "K7/8/8/8/8/8/1N6/k7 w - - 0 1",
        "K6R/PPPPPPPP/8/8/8/8/pppppppp/k6r w - - 0 1",
    ];
    for fen in fens {
        let board = BoardState::parse_fen(fen);
        let pos = SfnnPosition::from_board(&board);
        for perspective in [Side::White, Side::Black] {
            let mut direct = Sfnn16Accs::empty();
            refresh_perspective(&pos, &loaded, perspective, &mut direct);
            let mut via_finny = Sfnn16Accs::empty();
            update_perspective_finny_or_refresh(&pos, &loaded, perspective, &mut via_finny);
            assert_eq!(
                direct.halfka[perspective as usize], via_finny.halfka[perspective as usize],
                "finny mismatch for {fen} {perspective:?} first touch"
            );
            let mut via_finny2 = Sfnn16Accs::empty();
            update_perspective_finny_or_refresh(&pos, &loaded, perspective, &mut via_finny2);
            assert_eq!(
                direct.halfka[perspective as usize], via_finny2.halfka[perspective as usize],
                "finny mismatch for {fen} {perspective:?} second touch"
            );
            assert_eq!(
                direct.psqt[perspective as usize],
                via_finny2.psqt[perspective as usize]
            );
        }
    }
    let board_b = BoardState::parse_fen(fens[1]);
    let mut pos_b = SfnnPosition::from_board(&board_b);
    pos_b.white |= 1u64 << 18;
    let mut acc_inc = Sfnn16Accs::empty();
    let loaded_ref = &loaded;
    for perspective in [Side::White, Side::Black] {
        let mut direct = Sfnn16Accs::empty();
        refresh_perspective(&pos_b, loaded_ref, perspective, &mut direct);
        update_perspective_finny_or_refresh(&pos_b, loaded_ref, perspective, &mut acc_inc);
        assert_eq!(
            direct.halfka[perspective as usize],
            acc_inc.halfka[perspective as usize]
        );
    }
}

#[test]
fn scatter_empty_feats_safe() {
    let l1 = 8usize;
    let tr = SfnnTransformer {
        bias: vec![0; l1],
        weights: vec![1i16; 2 * l1],
        threat_w: Vec::new(),
        threat_w_i16: Vec::new(),
        psqt_w: vec![2i32; 2 * N_BUCKETS],
        threat_psqt_w: Vec::new(),
        pair_w: Vec::new(),
        pair_w_i16: Vec::new(),
        pair_psqt_w: Vec::new(),
    };
    let mut acc = [0i16; 8];
    let mut psqt = [0i32; N_BUCKETS];
    scatter_halfka(&tr, l1, &[], &mut acc, &mut psqt, 1);
    assert!(acc.iter().all(|&v| v == 0));
    assert!(psqt.iter().all(|&v| v == 0));
    scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, 1);
    assert!(acc.iter().all(|&v| v == 1));
    scatter_halfka(&tr, l1, &[0], &mut acc, &mut psqt, -1);
    assert!(acc.iter().all(|&v| v == 0));
}

#[test]
fn threat_pair_edge_cases_no_global() {
    assert_eq!(pseudo_attacks_sf(0, 0), 0);
    assert_eq!(pseudo_attacks_sf(7, 10), 0);
    assert_eq!(pseudo_attacks_sf(8, 10), 0);
    assert_eq!(pseudo_attacks_sf(15, 10), 0);
    let kingless_attack =
        SfnnPosition::from_board(&BoardState::parse_fen("8/8/8/3p4/4P3/8/8/8 w - - 0 1"));
    let mut n = 0;
    for_each_threat(&kingless_attack, |_, _, _, _| n += 1);
    assert!(n > 0);
    let mut inconsistent =
        SfnnPosition::from_board(&BoardState::parse_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    inconsistent.white |= 1u64 << 18;
    let mut skipped = 0;
    for_each_threat(&inconsistent, |_, _, _, _| skipped += 1);
    let _ = skipped;
    let mut dense_pieces = [0u64; 6];
    dense_pieces[Piece::Queen as usize] = u64::MAX;
    let dense = SfnnPosition {
        pieces: dense_pieces,
        white: 0x0000_0000_FFFF_FFFF,
        black: 0xFFFF_FFFF_0000_0000,
        mapping: [4u8; 64],
    };
    let mut tbuf = [0usize; MAX_THREAT_ACTIVE];
    let tn = collect_threats(&dense, Side::White, &mut tbuf);
    assert!(tn > 100 && tn <= MAX_THREAT_ACTIVE);
    let dense_p = SfnnPosition::from_board(&BoardState::parse_fen(
        "8/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP/8 w - - 0 1",
    ));
    let mut pbuf = [0usize; MAX_PAIR_ACTIVE];
    let pn = collect_pairs(&dense_p, Side::White, &mut pbuf);
    assert!(pn > 0);
    for &idx in &pbuf[..pn] {
        assert!((PSQ_DIMS + THREAT_DIMS..PSQ_DIMS + THREAT_DIMS + PAIR_DIMS).contains(&idx));
    }
    let mut found_none = false;
    let mut found_some = false;
    for attacker in [1usize, 2, 3, 4, 5] {
        for attacked in [1usize, 2, 3, 4, 5, 9, 10, 11, 12, 13] {
            for (from, to) in [(10usize, 20usize), (20, 10)] {
                match threat_index_for(Side::White, attacker, from, to, attacked, 4) {
                    None => found_none = true,
                    Some(idx) => {
                        found_some = true;
                        assert!(idx < THREAT_DIMS);
                    }
                }
            }
        }
    }
    assert!(found_none && found_some);
}

#[test]
fn kingless_evaluate_board_deterministic() {
    let board = BoardState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let mut accs = Sfnn16Accs::empty();
    assert!(evaluate_nets(&pos, &mut accs, Side::White).is_none());
    let gen_before = accs.generation;
    refresh_all(&pos, &mut accs);
    assert_eq!(accs.generation, gen_before);
    ensure_fresh(&pos, &mut accs);
    let mut pending = SfnnPending::default();
    note_add(&mut pending, Square::E2, Side::White, Piece::Pawn);
    flush_pending(&pos, &mut accs, &mut pending);
    assert_eq!(pending.n_adds, 0);
    apply_queued(&pos, &mut accs, &[], &[], &[false, false]);
    let mut board_mut = BoardState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");
    assert!(evaluate_board(&mut board_mut).is_none());
    assert!(evaluate_board_detailed(&mut board_mut).is_none());
}

#[test]
fn sfnn_arch_load_various_paddings() {
    for fc0_in in [2usize, 15, 32, 33, 64] {
        let data = build_valid_arch_bytes(fc0_in);
        let mut pos = 0;
        let arch = SfnnArch::load(&data, &mut pos, fc0_in, arch_hash(fc0_in as u32)).unwrap();
        assert_eq!(pos, data.len());
        assert_eq!(arch.fc0_w.len(), FC0_OUT * fc0_in.next_multiple_of(32));
    }
}

#[test]
fn read_rudi_overflow_guards() {
    assert_eq!(
        read_rudi_i16(&[], &mut 0, usize::MAX),
        Err("bad whale length")
    );
    assert_eq!(
        read_rudi_i32(&[], &mut 0, usize::MAX),
        Err("bad whale length")
    );
}

#[test]
fn cover_for_each_threat_skips_empty_mapping() {
    let pawn_pos = SfnnPosition {
        pieces: {
            let mut p = [0u64; 6];
            p[Piece::Pawn as usize] = 1u64 << 12;
            p[Piece::King as usize] = (1u64 << 4) | (1u64 << 60);
            p
        },
        white: (1u64 << 12) | (1u64 << 4),
        black: (1u64 << 19) | (1u64 << 60),
        mapping: {
            let mut m = [6u8; 64];
            m[12] = Piece::Pawn as u8;
            m[4] = Piece::King as u8;
            m[60] = Piece::King as u8;
            m[19] = 6;
            m
        },
    };

    let mut pawn_edges = 0;
    for_each_threat(&pawn_pos, |_, _, _, _| pawn_edges += 1);
    let mut pawn_fixed = pawn_pos;
    pawn_fixed.mapping[19] = Piece::Pawn as u8;
    pawn_fixed.pieces[Piece::Pawn as usize] |= 1u64 << 19;
    let mut pawn_fixed_edges = 0;
    for_each_threat(&pawn_fixed, |_, _, _, _| pawn_fixed_edges += 1);
    assert!(pawn_fixed_edges > pawn_edges);

    assert_ne!(knight_attacks_sf(10) & (1u64 << 20), 0);
    let knight_pos = SfnnPosition {
        pieces: {
            let mut p = [0u64; 6];
            p[Piece::Knight as usize] = 1u64 << 10;
            p[Piece::King as usize] = (1u64 << 4) | (1u64 << 60);
            p
        },
        white: (1u64 << 10) | (1u64 << 4),
        black: (1u64 << 20) | (1u64 << 60),
        mapping: {
            let mut m = [6u8; 64];
            m[10] = Piece::Knight as u8;
            m[4] = Piece::King as u8;
            m[60] = Piece::King as u8;
            m[20] = 6;
            m
        },
    };
    let mut knight_edges = 0;
    for_each_threat(&knight_pos, |_, _, _, _| knight_edges += 1);
    let mut knight_fixed = knight_pos;
    knight_fixed.mapping[20] = Piece::Pawn as u8;
    knight_fixed.pieces[Piece::Pawn as usize] |= 1u64 << 20;
    let mut knight_fixed_edges = 0;
    for_each_threat(&knight_fixed, |_, _, _, _| knight_fixed_edges += 1);
    assert!(knight_fixed_edges > knight_edges);
}

#[test]
fn cover_threat_index_never_overflows() {
    let mut max_seen = 0usize;
    for perspective in [Side::White, Side::Black] {
        for attacker in 0..16usize {
            for from in 0..64usize {
                for to in 0..64usize {
                    for attacked in 0..16usize {
                        for ksq in [0usize, 4, 60] {
                            if let Some(idx) =
                                threat_make_index(perspective, attacker, from, to, attacked, ksq)
                            {
                                max_seen = max_seen.max(idx);
                                assert!(idx < THREAT_DIMS);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(max_seen < THREAT_DIMS);
}

#[test]
fn cover_load_bytes_truncation_hits_pp_and_combined() {
    let sf17 = build_tiny_sf17_for_test(2);

    let mut saw_err = 0;
    let mut len = sf17.len();
    while len > 100 {
        len = len.saturating_sub(20_000);
        if Sfnn16Net::load_bytes(&sf17[..len], true, 2).is_err() {
            saw_err += 1;
        }
        if len <= 100 {
            break;
        }
    }
    assert!(saw_err > 0);

    let pp_start = 38 + THREAT_DIMS * 2 + (16 + 4 + THREAT_DIMS * N_BUCKETS) + PAIR_DIMS * 2;
    assert!(Sfnn16Net::load_bytes(&sf17[..pp_start + 10], true, 2).is_err());
    assert!(Sfnn16Net::load_bytes(&sf17[..pp_start + 30], true, 2).is_err());

    let combined = build_tiny_combined_threats_for_test(2);
    let early = 38 + 16 + 4 + 1000;
    assert!(Sfnn16Net::load_bytes(&combined[..early], true, 2).is_err());
    let mid = combined.len() - 50_000;
    assert!(Sfnn16Net::load_bytes(&combined[..mid], true, 2).is_err());
}

#[test]
fn cover_load_rudi_raw_payload() {
    let raw = vec![0u8; 181_011_108];
    let net = Sfnn16Net::load_rudi(&raw).expect("raw zero payload should parse");
    assert!(net.use_threats);
    assert_eq!(net.transformer.threat_w_i16.len(), THREAT_DIMS * L1);
    assert_eq!(net.transformer.pair_w_i16.len(), PAIR_DIMS * L1);
}

#[test]
fn cover_finny_changed_common_overflow() {
    let loaded = dummy_full_loaded_nets_for_test();
    let board = BoardState::parse_fen("6k1/8/8/8/8/P7/PPPPPPPP/1K6 w - - 0 1");
    let pos_a = SfnnPosition::from_board(&board);
    let occ = pos_a.occupied();
    assert!(occ.count_ones() >= 11);
    let mut pos_b = SfnnPosition {
        pieces: pos_a.pieces,
        white: pos_a.white,
        black: pos_a.black,
        mapping: pos_a.mapping,
    };
    let mut flipped = 0;
    for (s, cell) in pos_b.mapping.iter_mut().enumerate() {
        if ((occ >> s) & 1) == 1 && *cell == Piece::Pawn as u8 && flipped < 9 {
            *cell = Piece::Knight as u8;
            flipped += 1;
        }
    }
    assert_eq!(flipped, 9);
    let mut acc = Sfnn16Accs::empty();
    update_perspective_finny_or_refresh(&pos_a, &loaded, Side::White, &mut acc);
    update_perspective_finny_or_refresh(&pos_b, &loaded, Side::White, &mut acc);
    let mut direct = Sfnn16Accs::empty();
    refresh_perspective(&pos_b, &loaded, Side::White, &mut direct);
    assert_eq!(
        acc.halfka[Side::White as usize],
        direct.halfka[Side::White as usize]
    );
    assert_eq!(
        acc.psqt[Side::White as usize],
        direct.psqt[Side::White as usize]
    );
}

#[test]
fn cover_finny_skips_invalid_mapping() {
    let loaded = dummy_full_loaded_nets_for_test();

    let mut pos_a =
        SfnnPosition::from_board(&BoardState::parse_fen("6k1/8/8/8/8/8/1P6/1K6 w - - 0 1"));
    pos_a.white |= 1u64 << 18;

    assert_eq!(pos_a.mapping[18], 6);
    let pos_b = SfnnPosition::from_board(&BoardState::parse_fen("6k1/8/8/8/8/8/1P6/1K6 w - - 0 1"));
    let mut acc = Sfnn16Accs::empty();
    update_perspective_finny_or_refresh(&pos_a, &loaded, Side::White, &mut acc);
    update_perspective_finny_or_refresh(&pos_b, &loaded, Side::White, &mut acc);
    let mut direct = Sfnn16Accs::empty();
    refresh_perspective(&pos_b, &loaded, Side::White, &mut direct);
    assert_eq!(
        acc.halfka[Side::White as usize],
        direct.halfka[Side::White as usize]
    );

    let pos_c = SfnnPosition::from_board(&BoardState::parse_fen("6k1/8/8/8/8/8/1P6/1K6 w - - 0 1"));
    let mut pos_d = SfnnPosition {
        pieces: pos_c.pieces,
        white: pos_c.white | (1u64 << 18),
        black: pos_c.black,
        mapping: pos_c.mapping,
    };
    pos_d.mapping[18] = 6;
    let mut acc2 = Sfnn16Accs::empty();
    update_perspective_finny_or_refresh(&pos_c, &loaded, Side::White, &mut acc2);
    update_perspective_finny_or_refresh(&pos_d, &loaded, Side::White, &mut acc2);
    let mut direct2 = Sfnn16Accs::empty();
    refresh_perspective(&pos_d, &loaded, Side::White, &mut direct2);
    assert_eq!(
        acc2.halfka[Side::White as usize],
        direct2.halfka[Side::White as usize]
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn cover_add_threat_w_i16_avx2_direct() {
    let mut buf = [0i32; L1];
    let w = [1i16; L1];
    unsafe { add_threat_w_i16_avx2(&mut buf, &w) };
    assert!(buf.iter().all(|&v| v == 1));
    let w2 = [-1i16; L1];
    unsafe { add_threat_w_i16_avx2(&mut buf, &w2) };
    assert!(buf.iter().all(|&v| v == 0));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn cover_rudi_1024_threat_pair_avx2() {
    let board = BoardState::parse_fen("4k3/8/8/8/2p5/2n5/1PP5/4K3 w - - 0 1");
    let pos = SfnnPosition::from_board(&board);
    let base = [5i16; L1];
    let psqt = [7i32; N_BUCKETS];
    let net = Sfnn16Net {
        l1: L1,
        use_threats: true,
        is_rudi: true,
        transformer: SfnnTransformer {
            bias: Vec::new(),
            weights: Vec::new(),
            threat_w: Vec::new(),
            threat_w_i16: vec![1i16; THREAT_DIMS * L1],
            psqt_w: Vec::new(),
            threat_psqt_w: Vec::new(),
            pair_w: Vec::new(),
            pair_w_i16: vec![2i16; PAIR_DIMS * L1],
            pair_psqt_w: Vec::new(),
        },
        stacks: vec![SfnnArch {
            fc0_bias: [0; FC0_OUT],
            fc0_w: vec![0; L1 * FC0_OUT],
            fc1_bias: [0; FC1_OUT],
            fc1_w: vec![0; FC1_IN * FC1_OUT],
            fc2_bias: 0,
            fc2_w: [0; FC0_OUT * 2 + FC1_OUT * 2],
            is_rudi: true,
        }],
    };
    let mut tbuf = [0usize; MAX_THREAT_ACTIVE];
    let mut pbuf = [0usize; MAX_PAIR_ACTIVE];
    assert!(collect_threats(&pos, Side::White, &mut tbuf) > 0);
    assert!(collect_pairs(&pos, Side::White, &mut pbuf) > 0);
    let a = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], Side::White, 0);
    let b = eval_with_net(&pos, &net, [&base; 2], [&psqt; 2], Side::White, 0);
    assert_eq!(a, b);
}
