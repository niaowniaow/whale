use crate::common::random;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Result, Write};
use std::process::exit;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;

use crate::board::state::BoardState;
use crate::common::castle::Castle as WhaleCastle;
use crate::common::move_list::MoveList;
use crate::common::move_type::MoveType;
use crate::common::moves::Move;
use crate::common::piece::Piece as WhalePiece;
use crate::common::side::Side as WhaleSide;
use crate::common::square::Square as WhaleSquare;
use crate::search::search_state::SearchState;
use crate::teacher::StockfishTeacher;

use bullet_lib::game::formats::viriformat::{
    chess::board::Board as ViriBoard,
    chess::chessmove::{Move as ViriMove, MoveFlags as ViriMoveFlags},
    chess::piece::{Colour as ViriColour, Piece as ViriPiece, PieceType as ViriPieceType},
    chess::types::{CastlingRights as ViriCastlingRights, Square as ViriSquare},
    dataformat::Game as ViriGame,
};

pub struct SelfPlayPosition {
    pub side_to_move: WhaleSide,
    pub mv: Move,
    pub engine_eval: i16,
    pub fen: String,
    pub behavior_state: u8,
    pub behavior_intent: u8,
    pub behavior_pressure: i32,
    pub behavior_opp_cpi: i32,
    pub behavior_own_cpi: i32,
    pub behavior_urgency: u8,
    pub behavior_risk: i32,
    pub behavior_musttry: bool,
    pub behavior_concession: i16,
}

pub fn behavior_state_id(state: crate::world::position_state::PositionState) -> u8 {
    use crate::world::position_state::PositionState;
    match state {
        PositionState::Defend => 0,
        PositionState::Stabilize => 1,
        PositionState::Improve => 2,
        PositionState::Press => 3,
        PositionState::Attack => 4,
        PositionState::Reset => 5,
        PositionState::Crush => 6,
        PositionState::Convert => 7,
    }
}

pub fn behavior_intent_id(intent: crate::world::intent::SearchIntent) -> u8 {
    use crate::world::intent::SearchIntent;
    match intent {
        SearchIntent::Survival => 0,
        SearchIntent::Stabilization => 1,
        SearchIntent::Improvement => 2,
        SearchIntent::Pressure => 3,
        SearchIntent::Attack => 4,
        SearchIntent::Verification => 5,
        SearchIntent::Recovery => 6,
        SearchIntent::Conversion => 7,
    }
}

pub fn behavior_urgency_id(urgency: crate::risk::Urgency) -> u8 {
    use crate::risk::Urgency;
    match urgency {
        Urgency::Low => 0,
        Urgency::Medium => 1,
        Urgency::High => 2,
        Urgency::MustTry => 3,
    }
}

pub fn move_to_uci(m: &Move) -> String {
    let promo = m
        .promotion_char()
        .map(|c| c.to_string())
        .unwrap_or_default();
    format!("{}{}{}", m.source, m.target, promo)
}

pub fn outcome_for_side(outcome: WhaleSide, side: WhaleSide) -> f32 {
    if outcome == WhaleSide::Both {
        return 0.5;
    }
    if outcome == side { 1.0 } else { 0.0 }
}

pub const DATAGEN_TEMPERATURE_PLIES: usize = 12;

pub const DATAGEN_TEMPERATURE_OPENING: f32 = 1.5;

pub const DATAGEN_TEMPERATURE_MID: f32 = 1.0;

pub const DATAGEN_EPSILON_PER_MILLE: u64 = 15;

pub const DATAGEN_RESIGN_THRESHOLD_CP: i16 = 900;

pub const DATAGEN_RESIGN_EARLIEST_PLY: usize = 20;

pub const DATAGEN_DRAW_THRESHOLD_CP: i16 = 20;

pub const DATAGEN_DRAW_CONSECUTIVE_PLIES: usize = 12;

pub const DATAGEN_MAX_PLIES: usize = 400;

#[inline(always)]
pub fn temperature_for_ply(ply: usize) -> f32 {
    if ply < 8 {
        DATAGEN_TEMPERATURE_OPENING
    } else if ply < DATAGEN_TEMPERATURE_PLIES {
        DATAGEN_TEMPERATURE_MID
    } else {
        0.0
    }
}

pub fn tempered_index(visits: &[u64], temperature: f32) -> usize {
    if visits.is_empty() {
        return 0;
    }
    if visits.len() == 1 {
        return 0;
    }
    if temperature <= 0.0 {
        let mut best = 0;
        for i in 1..visits.len() {
            if visits[i] > visits[best] {
                best = i;
            }
        }
        return best;
    }
    let inv = 1.0 / temperature;
    let mut weights: Vec<f32> = Vec::with_capacity(visits.len());
    let mut sum: f32 = 0.0;
    for v in visits {
        let w = (*v as f32 + 1.0).powf(inv);
        weights.push(w);
        sum += w;
    }
    if sum <= 0.0 {
        return (random::next_u64() as usize) % visits.len();
    }
    let r = (random::next_u64() as f64 / u64::MAX as f64) * sum as f64;
    let mut acc: f64 = 0.0;
    for i in 0..weights.len() {
        acc += weights[i] as f64;
        if r < acc {
            return i;
        }
    }
    visits.len() - 1
}

#[inline(always)]
pub fn epsilon_roll_per_mille(per_mille: u64) -> bool {
    random::next_u64() % 1000 < per_mille
}

#[inline(always)]
pub fn should_epsilon_pick() -> bool {
    epsilon_roll_per_mille(DATAGEN_EPSILON_PER_MILLE)
}

pub fn random_legal_move(board: &BoardState) -> Option<Move> {
    let mut list = MoveList::new();
    board.generate_moves(&mut list);
    let mut legal: Vec<Move> = Vec::new();
    for i in 0..list.len() {
        let m = list[i].mv;
        if board.is_legal(m) {
            legal.push(m);
        }
    }
    if legal.is_empty() {
        return None;
    }
    let idx = (random::next_u64() as usize) % legal.len();
    Some(legal[idx])
}

pub fn choose_exploration_move(board: &BoardState, best: Move, ply: usize) -> Move {
    let temp = temperature_for_ply(ply);
    if temp > 0.0 {
        let mut list = MoveList::new();
        board.generate_moves(&mut list);
        let mut cands: Vec<Move> = Vec::new();
        cands.push(best);
        for i in 0..list.len() {
            let m = list[i].mv;
            if m == best {
                continue;
            }
            if board.is_legal(m) {
                cands.push(m);
            }
        }
        if cands.len() > 1 {
            let mut visits: Vec<u64> = Vec::with_capacity(cands.len());
            visits.push(200);
            for _ in 1..cands.len() {
                visits.push(5);
            }
            let idx = tempered_index(&visits, temp);
            if idx < cands.len() {
                return cands[idx];
            }
        }
        return best;
    }
    if epsilon_roll_per_mille(DATAGEN_EPSILON_PER_MILLE) {
        if let Some(r) = random_legal_move(board) {
            return r;
        }
    }
    best
}

#[inline(always)]
pub fn resign_winner(score_stm: i16, ply: usize, stm: WhaleSide) -> Option<WhaleSide> {
    if ply >= DATAGEN_RESIGN_EARLIEST_PLY && score_stm <= -DATAGEN_RESIGN_THRESHOLD_CP {
        Some(stm.other())
    } else {
        None
    }
}

#[inline(always)]
pub fn draw_streak_next(score_stm: i16, streak: usize) -> usize {
    if score_stm.abs() <= DATAGEN_DRAW_THRESHOLD_CP {
        streak.saturating_add(1)
    } else {
        0
    }
}

#[inline(always)]
pub fn is_draw_adjudicated(streak: usize) -> bool {
    streak >= DATAGEN_DRAW_CONSECUTIVE_PLIES
}

pub fn tb_rescore_outcome(
    initial: &BoardState,
    positions: &[SelfPlayPosition],
    outcome: WhaleSide,
) -> WhaleSide {
    let mut corrected = outcome;
    let mut replay = initial.clone();
    if replay.half_move_clock == 0 {
        if let Some(w) = crate::syzygy::probe_wdl(&replay) {
            if w == 1 {
                corrected = replay.side_to_move;
            } else if w == -1 {
                corrected = replay.side_to_move.other();
            } else {
                corrected = WhaleSide::Both;
            }
        }
    }
    for pos in positions {
        replay.make_move(pos.mv);
        if replay.half_move_clock != 0 {
            continue;
        }
        if let Some(w) = crate::syzygy::probe_wdl(&replay) {
            if w == 1 {
                corrected = replay.side_to_move;
            } else if w == -1 {
                corrected = replay.side_to_move.other();
            } else {
                corrected = WhaleSide::Both;
            }
        }
    }
    corrected
}

pub fn write_behavior_jsonl<W: Write>(
    positions: &[SelfPlayPosition],
    outcome: WhaleSide,
    writer: &mut W,
) -> Result<()> {
    for pos in positions {
        let side = match pos.side_to_move {
            WhaleSide::White => "w",
            WhaleSide::Black => "b",
            _ => "d",
        };
        let white_pov = if pos.side_to_move == WhaleSide::White {
            pos.engine_eval
        } else {
            -pos.engine_eval
        };
        writeln!(
            writer,
            "{{\"fen\":\"{}\",\"side\":\"{}\",\"move\":\"{}\",\"eval\":{},\"state\":{},\"intent\":{},\"pressure\":{},\"opp_cpi\":{},\"own_cpi\":{},\"urgency\":{},\"risk\":{},\"musttry\":{},\"concession\":{},\"result\":{}}}",
            pos.fen,
            side,
            move_to_uci(&pos.mv),
            white_pov,
            pos.behavior_state,
            pos.behavior_intent,
            pos.behavior_pressure,
            pos.behavior_opp_cpi,
            pos.behavior_own_cpi,
            pos.behavior_urgency,
            pos.behavior_risk,
            pos.behavior_musttry,
            pos.behavior_concession,
            outcome_for_side(outcome, pos.side_to_move),
        )?;
    }
    Ok(())
}

struct CompletedGame {
    initial_state: BoardState,
    initial_eval: i16,
    positions: Vec<SelfPlayPosition>,
    outcome: WhaleSide,
}

pub fn load_opening_book(file_path: &str) -> Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let fens: Vec<String> = reader
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();
    if fens.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, "Opening book is empty"));
    }
    Ok(fens)
}

pub fn board_state_to_viriboard(whale_state: &BoardState) -> ViriBoard {
    let mut viriboard = ViriBoard::new();

    *viriboard.turn_mut() = match whale_state.side_to_move {
        WhaleSide::White => ViriColour::White,
        WhaleSide::Black => ViriColour::Black,
        _ => panic!("Invalid turn side"),
    };

    *viriboard.ep_sq_mut() = if whale_state.en_passant_square == WhaleSquare::NoSquare {
        None
    } else {
        let viri_ep_idx = (whale_state.en_passant_square as u8) ^ 56;
        ViriSquare::new(viri_ep_idx)
    };

    let mut castling = ViriCastlingRights::NONE;
    if whale_state.castle.contains(WhaleCastle::WHITE_SHORT) {
        castling.wk = Some(ViriSquare::H1);
    }
    if whale_state.castle.contains(WhaleCastle::WHITE_LONG) {
        castling.wq = Some(ViriSquare::A1);
    }
    if whale_state.castle.contains(WhaleCastle::BLACK_SHORT) {
        castling.bk = Some(ViriSquare::H8);
    }
    if whale_state.castle.contains(WhaleCastle::BLACK_LONG) {
        castling.bq = Some(ViriSquare::A8);
    }
    *viriboard.castling_rights_mut() = castling;

    *viriboard.halfmove_clock_mut() = whale_state.half_move_clock;
    viriboard.set_fullmove_clock((1 + whale_state.move_count / 2) as u16);

    for whale_idx in 0..64 {
        let piece = whale_state.piece_mapping[whale_idx];
        if piece != WhalePiece::None {
            let side = if whale_state.occupancies[WhaleSide::White].get_bit(whale_idx) == 1 {
                ViriColour::White
            } else {
                ViriColour::Black
            };

            let viri_pt = match piece {
                WhalePiece::Pawn => ViriPieceType::Pawn,
                WhalePiece::Knight => ViriPieceType::Knight,
                WhalePiece::Bishop => ViriPieceType::Bishop,
                WhalePiece::Rook => ViriPieceType::Rook,
                WhalePiece::Queen => ViriPieceType::Queen,
                WhalePiece::King => ViriPieceType::King,
                WhalePiece::None => unreachable!(),
            };

            let viri_sq_idx = (whale_idx as u8) ^ 56;
            let viri_sq = ViriSquare::new_clamped(viri_sq_idx);

            viriboard.add_piece(viri_sq, ViriPiece::new(side, viri_pt));
        }
    }

    viriboard.regenerate_zobrist();
    viriboard.regenerate_threats();

    viriboard
}

pub fn map_whale_move(m: &Move) -> ViriMove {
    let from_viri = ViriSquare::new_clamped((m.source as u8) ^ 56);
    let to_viri = if m.move_type == MoveType::Castle {
        let rook_sq = match m.target {
            WhaleSquare::G1 => WhaleSquare::H1,
            WhaleSquare::C1 => WhaleSquare::A1,
            WhaleSquare::G8 => WhaleSquare::H8,
            WhaleSquare::C8 => WhaleSquare::A8,
            _ => m.target,
        };
        ViriSquare::new_clamped((rook_sq as u8) ^ 56)
    } else {
        ViriSquare::new_clamped((m.target as u8) ^ 56)
    };
    match m.move_type {
        MoveType::EnPassant => {
            ViriMove::new_with_flags(from_viri, to_viri, ViriMoveFlags::EnPassant)
        }
        MoveType::Castle => ViriMove::new_with_flags(from_viri, to_viri, ViriMoveFlags::Castle),
        MoveType::QueenPromotion | MoveType::QueenPromotionCapture => {
            ViriMove::new_with_promo(from_viri, to_viri, ViriPieceType::Queen)
        }
        MoveType::RookPromotion | MoveType::RookPromotionCapture => {
            ViriMove::new_with_promo(from_viri, to_viri, ViriPieceType::Rook)
        }
        MoveType::BishopPromotion | MoveType::BishopPromotionCapture => {
            ViriMove::new_with_promo(from_viri, to_viri, ViriPieceType::Bishop)
        }
        MoveType::KnightPromotion | MoveType::KnightPromotionCapture => {
            ViriMove::new_with_promo(from_viri, to_viri, ViriPieceType::Knight)
        }
        _ => ViriMove::new(from_viri, to_viri),
    }
}

pub fn write_game_to_binpack<W: Write>(
    initial_state: &BoardState,
    initial_eval: i16,
    positions: &[SelfPlayPosition],
    outcome: WhaleSide,
    writer: &mut W,
) -> Result<()> {
    let start_board = board_state_to_viriboard(initial_state);

    let initial_white_pov_eval = if initial_state.side_to_move == WhaleSide::White {
        initial_eval
    } else {
        -initial_eval
    };

    let wdl_outcome = match outcome {
        WhaleSide::White => 2,
        WhaleSide::Black => 0,
        _ => 1,
    };

    let initial_position = start_board.to_marlinformat(initial_white_pov_eval, wdl_outcome, 0);

    let mut game = ViriGame {
        initial_position,
        moves: Vec::with_capacity(positions.len()),
    };

    for pos in positions {
        let viri_move = map_whale_move(&pos.mv);
        let white_pov_eval = if pos.side_to_move == WhaleSide::White {
            pos.engine_eval
        } else {
            -pos.engine_eval
        };
        game.add_move(viri_move, white_pov_eval);
    }

    game.serialise_into(writer)
}

#[derive(Debug, Clone, Default)]
pub struct DatagenMetadata {
    pub games_completed: usize,
    pub total_positions: usize,
    pub white_wins: usize,
    pub black_wins: usize,
    pub draws: usize,
}

pub fn parse_metadata_json(content: &str) -> DatagenMetadata {
    let mut meta = DatagenMetadata::default();
    let cleaned = content.trim().trim_start_matches('{').trim_end_matches('}');
    for entry in cleaned.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        if let Some((key_part, val_part)) = entry.split_once(':') {
            let key = key_part.trim().trim_matches('"').trim();
            let val_str = val_part.trim().trim_matches('"').trim();
            if let Ok(val) = val_str.parse::<usize>() {
                match key {
                    "games_completed" => meta.games_completed = val,
                    "total_positions" => meta.total_positions = val,
                    "white_wins" => meta.white_wins = val,
                    "black_wins" => meta.black_wins = val,
                    "draws" => meta.draws = val,
                    _ => {}
                }
            }
        }
    }
    meta
}

pub fn read_metadata(output_path: &str) -> DatagenMetadata {
    let meta_path = format!("{}.meta", output_path);
    match std::fs::read_to_string(&meta_path) {
        Ok(c) => parse_metadata_json(&c),
        Err(_) => DatagenMetadata::default(),
    }
}

pub fn write_metadata(output_path: &str, meta: &DatagenMetadata) -> Result<()> {
    let meta_path = format!("{}.meta", output_path);
    let temp_path = format!("{}.meta.tmp", output_path);
    let content = format!(
        "{{\n  \"games_completed\": {},\n  \"total_positions\": {},\n  \"white_wins\": {},\n  \"black_wins\": {},\n  \"draws\": {}\n}}\n",
        meta.games_completed, meta.total_positions, meta.white_wins, meta.black_wins, meta.draws
    );
    std::fs::write(&temp_path, content)?;
    std::fs::rename(&temp_path, &meta_path)
}

pub fn run(output_path: &str, num_games: usize, book_path: &str, depth: u8, num_threads: usize) {
    run_with_teacher(output_path, num_games, book_path, depth, num_threads, None);
}

pub fn run_with_teacher(
    output_path: &str,
    num_games: usize,
    book_path: &str,
    depth: u8,
    num_threads: usize,
    teacher_path: Option<&str>,
) {
    let overall_start = std::time::Instant::now();
    let initial_metadata = read_metadata(output_path);

    println!("Loading opening book from '{}'...", book_path);
    let book_fens = match load_opening_book(book_path) {
        Ok(fens) => fens,
        Err(e) => {
            println!("Error loading opening book: {}", e);
            return;
        }
    };
    println!("Loaded {} starting positions from book.", book_fens.len());

    let file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
    {
        Ok(f) => f,
        Err(e) => {
            println!("Error opening output file: {}", e);
            return;
        }
    };
    let mut writer = BufWriter::new(file);
    let behavior_path = format!("{}.behavior.jsonl", output_path);
    let behavior_file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&behavior_path)
    {
        Ok(f) => f,
        Err(e) => {
            println!("Error opening behavior file: {}", e);
            return;
        }
    };
    let mut behavior_writer = BufWriter::new(behavior_file);

    let num_threads = num_threads.clamp(1, 256);
    if num_games == 0 {
        println!("Nothing to do: number of games is 0.");
        return;
    }

    let (tx, rx) = mpsc::sync_channel(256);

    let games_per_thread = num_games / num_threads;
    let remainder = num_games % num_threads;

    let book_fens_ref = &book_fens;

    std::thread::scope(|s| {
        for t in 0..num_threads {
            let tx = tx.clone();
            let thread_games = games_per_thread + if t < remainder { 1 } else { 0 };

            s.spawn(move || {
                let cancellation_token = AtomicBool::new(false);
                let mut debug_mode = false;
                let mut search_state = SearchState::new();
                let mut teacher = teacher_path.map(|path| match StockfishTeacher::new(path) {
                    Ok(teacher) => teacher,
                    Err(error) => panic!("failed to start Stockfish teacher '{path}': {error}"),
                });

                for _ in 0..thread_games {
                    let starting_fen = random::choose(book_fens_ref).unwrap();
                    let mut board_state = BoardState::parse_fen(starting_fen);
                    let initial_state = board_state.clone();

                    search_state.tt.clear();
                    search_state.reset_heuristics();

                    search_state.reset_search();
                    let _ = board_state.find_best_move(
                        depth,
                        &cancellation_token,
                        &mut debug_mode,
                        &mut search_state,
                        1,
                    );
                    let initial_eval = teacher
                        .as_mut()
                        .map(|teacher| {
                            teacher
                                .evaluate(&board_state, depth)
                                .unwrap_or_else(|error| {
                                    panic!("Stockfish teacher evaluation failed: {error}")
                                })
                        })
                        .unwrap_or(search_state.score);

                    let mut positions = Vec::new();
                    let mut draw_streak: usize = 0;
                    let outcome;

                    loop {
                        if board_state.is_draw() {
                            outcome = WhaleSide::Both;
                            break;
                        }
                        if positions.len() >= DATAGEN_MAX_PLIES {
                            outcome = WhaleSide::Both;
                            break;
                        }

                        search_state.reset_search();
                        let best_move = board_state.find_best_move(
                            depth,
                            &cancellation_token,
                            &mut debug_mode,
                            &mut search_state,
                            1,
                        );

                        if best_move == Move::NO_MOVE {
                            if board_state.is_in_check(board_state.side_to_move) {
                                outcome = board_state.side_to_move.other();
                            } else {
                                outcome = WhaleSide::Both;
                            }
                            break;
                        }

                        let score = teacher
                            .as_mut()
                            .map(|teacher| {
                                teacher
                                    .evaluate(&board_state, depth)
                                    .unwrap_or_else(|error| {
                                        panic!("Stockfish teacher evaluation failed: {error}")
                                    })
                            })
                            .unwrap_or(search_state.score);

                        let ply = positions.len();
                        if let Some(winner) = resign_winner(score, ply, board_state.side_to_move) {
                            positions.push(SelfPlayPosition {
                                side_to_move: board_state.side_to_move,
                                mv: best_move,
                                engine_eval: score,
                                fen: board_state.to_fen(),
                                behavior_state: behavior_state_id(search_state.last_state),
                                behavior_intent: behavior_intent_id(search_state.last_intent),
                                behavior_pressure: search_state.pressure_state.pressure,
                                behavior_opp_cpi: search_state.prev_opp_cpi.unwrap_or(0),
                                behavior_own_cpi: search_state.prev_own_cpi.unwrap_or(0),
                                behavior_urgency: behavior_urgency_id(search_state.last_urgency),
                                behavior_risk: search_state.last_risk,
                                behavior_musttry: search_state.last_musttry,
                                behavior_concession: search_state
                                    .last_concession
                                    .map(|c| c.swing_cp)
                                    .unwrap_or(0),
                            });
                            outcome = winner;
                            break;
                        }
                        draw_streak = draw_streak_next(score, draw_streak);
                        if is_draw_adjudicated(draw_streak) {
                            positions.push(SelfPlayPosition {
                                side_to_move: board_state.side_to_move,
                                mv: best_move,
                                engine_eval: score,
                                fen: board_state.to_fen(),
                                behavior_state: behavior_state_id(search_state.last_state),
                                behavior_intent: behavior_intent_id(search_state.last_intent),
                                behavior_pressure: search_state.pressure_state.pressure,
                                behavior_opp_cpi: search_state.prev_opp_cpi.unwrap_or(0),
                                behavior_own_cpi: search_state.prev_own_cpi.unwrap_or(0),
                                behavior_urgency: behavior_urgency_id(search_state.last_urgency),
                                behavior_risk: search_state.last_risk,
                                behavior_musttry: search_state.last_musttry,
                                behavior_concession: search_state
                                    .last_concession
                                    .map(|c| c.swing_cp)
                                    .unwrap_or(0),
                            });
                            outcome = WhaleSide::Both;
                            break;
                        }
                        let chosen = choose_exploration_move(&board_state, best_move, ply);

                        positions.push(SelfPlayPosition {
                            side_to_move: board_state.side_to_move,
                            mv: chosen,
                            engine_eval: score,
                            fen: board_state.to_fen(),
                            behavior_state: behavior_state_id(search_state.last_state),
                            behavior_intent: behavior_intent_id(search_state.last_intent),
                            behavior_pressure: search_state.pressure_state.pressure,
                            behavior_opp_cpi: search_state.prev_opp_cpi.unwrap_or(0),
                            behavior_own_cpi: search_state.prev_own_cpi.unwrap_or(0),
                            behavior_urgency: behavior_urgency_id(search_state.last_urgency),
                            behavior_risk: search_state.last_risk,
                            behavior_musttry: search_state.last_musttry,
                            behavior_concession: search_state
                                .last_concession
                                .map(|c| c.swing_cp)
                                .unwrap_or(0),
                        });

                        board_state.make_move(chosen);
                    }

                    let outcome = tb_rescore_outcome(&initial_state, &positions, outcome);

                    let game_data = CompletedGame {
                        initial_state,
                        initial_eval,
                        positions,
                        outcome,
                    };
                    let _ = tx.send(game_data);
                }
            });
        }

        drop(tx);

        s.spawn(move || {
            let mut games_written = 0;
            let mut total_positions = 0;
            let mut white_wins = 0;
            let mut black_wins = 0;
            let mut draws = 0;
            let mut batch_positions = 0;
            let mut last_batch_time = overall_start;

            while let Ok(game) = rx.recv() {
                if let Err(e) = write_game_to_binpack(
                    &game.initial_state,
                    game.initial_eval,
                    &game.positions,
                    game.outcome,
                    &mut writer,
                ) {
                    println!("Error writing game to file: {}", e);
                    break;
                }
                if let Err(e) =
                    write_behavior_jsonl(&game.positions, game.outcome, &mut behavior_writer)
                {
                    println!("Error writing behavior file: {}", e);
                    break;
                }
                games_written += 1;
                total_positions += game.positions.len();
                batch_positions += game.positions.len();

                match game.outcome {
                    WhaleSide::White => white_wins += 1,
                    WhaleSide::Black => black_wins += 1,
                    _ => draws += 1,
                }

                if games_written % 2500 == 0 {
                    let _ = writer.flush();
                    let _ = behavior_writer.flush();
                    let updated_metadata = DatagenMetadata {
                        games_completed: initial_metadata.games_completed + games_written,
                        total_positions: initial_metadata.total_positions + total_positions,
                        white_wins: initial_metadata.white_wins + white_wins,
                        black_wins: initial_metadata.black_wins + black_wins,
                        draws: initial_metadata.draws + draws,
                    };
                    if let Err(e) = write_metadata(output_path, &updated_metadata) {
                        println!("Error writing metadata file: {}", e);
                    }

                    let avg_len = total_positions as f64 / games_written as f64;
                    let batch_duration = last_batch_time.elapsed();
                    let overall_duration = overall_start.elapsed();
                    let batch_secs = batch_duration.as_secs_f64();
                    let overall_secs = overall_duration.as_secs_f64();
                    last_batch_time = std::time::Instant::now();

                    println!("----------------------------------------");
                    println!("Games completed (run) : {}", games_written);
                    println!("Games left            : {}", num_games - games_written);
                    println!("Total positions (run) : {}", total_positions);
                    println!("Total White Wins (run): {}", white_wins);
                    println!("Total Black Wins (run): {}", black_wins);
                    println!("Total Draws (run)     : {}", draws);
                    println!("Average Game Length   : {:.2}", avg_len);
                    println!("Batch Time            : {:.2?}", batch_duration);
                    if batch_secs > 0.0 {
                        println!("Batch Games/sec       : {:.2}", 2500.0 / batch_secs);
                        println!(
                            "Batch Positions/sec   : {:.2}",
                            batch_positions as f64 / batch_secs
                        );
                    }
                    println!("Overall Time          : {:.2?}", overall_duration);
                    if overall_secs > 0.0 {
                        println!(
                            "Overall Games/sec     : {:.2}",
                            games_written as f64 / overall_secs
                        );
                        println!(
                            "Overall Positions/sec : {:.2}",
                            total_positions as f64 / overall_secs
                        );
                    }
                    if overall_secs > 0.0 && games_written < num_games {
                        let games_left = num_games - games_written;
                        let games_per_sec = games_written as f64 / overall_secs;
                        if games_per_sec > 0.0 {
                            let eta_secs = games_left as f64 / games_per_sec;
                            let eta_duration = std::time::Duration::from_secs_f64(eta_secs);
                            println!("Estimated Time Left  : {:.2?}", eta_duration);
                        }
                    }
                    batch_positions = 0;
                    println!("--- Cumulative Stats (Total in File) ---");
                    println!(
                        "Games completed       : {}",
                        updated_metadata.games_completed
                    );
                    println!(
                        "Total positions       : {}",
                        updated_metadata.total_positions
                    );
                    println!("Total White Wins      : {}", updated_metadata.white_wins);
                    println!("Total Black Wins      : {}", updated_metadata.black_wins);
                    println!("Total Draws           : {}", updated_metadata.draws);
                    println!("----------------------------------------");
                }
            }

            let _ = writer.flush();
            let _ = behavior_writer.flush();
            let updated_metadata = DatagenMetadata {
                games_completed: initial_metadata.games_completed + games_written,
                total_positions: initial_metadata.total_positions + total_positions,
                white_wins: initial_metadata.white_wins + white_wins,
                black_wins: initial_metadata.black_wins + black_wins,
                draws: initial_metadata.draws + draws,
            };
            if let Err(e) = write_metadata(output_path, &updated_metadata) {
                println!("Error writing metadata file: {}", e);
            }

            let avg_len = if games_written > 0 {
                total_positions as f64 / games_written as f64
            } else {
                0.0
            };
            let overall_duration = overall_start.elapsed();
            let overall_secs = overall_duration.as_secs_f64();
            println!("----------------------------------------");
            println!("Final Data Generation Summary:");
            println!("Games completed (run) : {}", games_written);
            println!("Games left            : {}", num_games - games_written);
            println!("Total positions (run) : {}", total_positions);
            println!("Total White Wins (run): {}", white_wins);
            println!("Total Black Wins (run): {}", black_wins);
            println!("Total Draws (run)     : {}", draws);
            println!("Average Game Length   : {:.2}", avg_len);
            println!("Overall Time          : {:.2?}", overall_duration);
            if overall_secs > 0.0 {
                println!(
                    "Overall Games/sec     : {:.2}",
                    games_written as f64 / overall_secs
                );
                println!(
                    "Overall Positions/sec : {:.2}",
                    total_positions as f64 / overall_secs
                );
            }
            println!("--- Cumulative Stats (Total in File) ---");
            println!(
                "Games completed       : {}",
                updated_metadata.games_completed
            );
            println!(
                "Total positions       : {}",
                updated_metadata.total_positions
            );
            println!("Total White Wins      : {}", updated_metadata.white_wins);
            println!("Total Black Wins      : {}", updated_metadata.black_wins);
            println!("Total Draws           : {}", updated_metadata.draws);
            println!("----------------------------------------");
            println!(
                "Data generation finished. Successfully wrote {} games to '{}'.",
                games_written, output_path,
            );
        });
    });
    exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn behavior_jsonl_roundtrip_fields() {
        use crate::common::helpers::STARTING_FEN;
        let board = BoardState::parse_fen(STARTING_FEN);
        let mut moves = crate::common::move_list::MoveList::new();
        board.generate_moves(&mut moves);
        let mv = moves[0].mv;
        let positions = vec![SelfPlayPosition {
            side_to_move: WhaleSide::White,
            mv,
            engine_eval: 24,
            fen: board.to_fen(),
            behavior_state: 3,
            behavior_intent: 3,
            behavior_pressure: 12,
            behavior_opp_cpi: 18,
            behavior_own_cpi: 22,
            behavior_urgency: 2,
            behavior_risk: 40,
            behavior_musttry: true,
            behavior_concession: 25,
        }];
        let mut buf: Vec<u8> = Vec::new();
        write_behavior_jsonl(&positions, WhaleSide::White, &mut buf).unwrap();
        let line = String::from_utf8(buf).unwrap();
        assert!(line.contains("\"move\":\""));
        assert!(line.contains("\"state\":3"));
        assert!(line.contains("\"musttry\":true"));
        assert!(line.contains("\"concession\":25"));
        assert!(line.contains("rnbqkbnr"));
        assert!(line.contains("\"result\":1"));
        assert_eq!(outcome_for_side(WhaleSide::Black, WhaleSide::White), 0.0);
        assert_eq!(outcome_for_side(WhaleSide::Both, WhaleSide::Black), 0.5);
        assert_eq!(
            behavior_state_id(crate::world::position_state::PositionState::Reset),
            5
        );
        assert_eq!(
            behavior_intent_id(crate::world::intent::SearchIntent::Recovery),
            6
        );
        assert_eq!(behavior_urgency_id(crate::risk::Urgency::MustTry), 3);
        assert_eq!(move_to_uci(&mv).len(), 4);
    }

    #[test]
    fn parse_metadata_handles_multiline_json() {
        let json = r#"{
  "games_completed": 150,
  "total_positions": 12500,
  "white_wins": 60,
  "black_wins": 50,
  "draws": 40
}
"#;
        let meta = parse_metadata_json(json);
        assert_eq!(meta.games_completed, 150);
        assert_eq!(meta.total_positions, 12500);
        assert_eq!(meta.white_wins, 60);
        assert_eq!(meta.black_wins, 50);
        assert_eq!(meta.draws, 40);
    }

    #[test]
    fn parse_metadata_handles_single_line_json() {
        let json = r#"{"games_completed": 42, "total_positions": 3000, "white_wins": 15, "black_wins": 12, "draws": 15}"#;
        let meta = parse_metadata_json(json);
        assert_eq!(meta.games_completed, 42);
        assert_eq!(meta.total_positions, 3000);
        assert_eq!(meta.white_wins, 15);
        assert_eq!(meta.black_wins, 12);
        assert_eq!(meta.draws, 15);
    }

    #[test]
    fn write_and_read_metadata_roundtrip() {
        let dir = std::env::temp_dir().join(format!("whale-datagen-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_out.binpack");
        let path_str = path.to_str().unwrap();

        let original = DatagenMetadata {
            games_completed: 100,
            total_positions: 7500,
            white_wins: 40,
            black_wins: 35,
            draws: 25,
        };

        write_metadata(path_str, &original).unwrap();
        let loaded = read_metadata(path_str);

        assert_eq!(loaded.games_completed, original.games_completed);
        assert_eq!(loaded.total_positions, original.total_positions);
        assert_eq!(loaded.white_wins, original.white_wins);
        assert_eq!(loaded.black_wins, original.black_wins);
        assert_eq!(loaded.draws, original.draws);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
