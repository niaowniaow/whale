import subprocess
import time
import sys
import os
import json
import chess
import chess.pgn

def send_cmd(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()

def wait_for(proc, target):
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if target in line:
            return line

def init_engine(binary_path, is_aprm=True):
    proc = subprocess.Popen(
        [binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1
    )
    send_cmd(proc, "uci")
    wait_for(proc, "uciok")
    
    if is_aprm:
        send_cmd(proc, "setoption name CPI_Enabled value true")
        send_cmd(proc, "setoption name State_Enabled value true")
        send_cmd(proc, "setoption name Risk_Enabled value true")
        send_cmd(proc, "setoption name Pressure_Enabled value true")
        send_cmd(proc, "setoption name Attack_Enabled value true")
        send_cmd(proc, "setoption name Conversion_Enabled value true")
        send_cmd(proc, "setoption name QS_Checks_Enabled value true")
    else:
        send_cmd(proc, "setoption name CPI_Enabled value false")
        send_cmd(proc, "setoption name State_Enabled value false")
        send_cmd(proc, "setoption name Risk_Enabled value false")
        send_cmd(proc, "setoption name Pressure_Enabled value false")
        send_cmd(proc, "setoption name Attack_Enabled value false")
        send_cmd(proc, "setoption name Conversion_Enabled value false")
        send_cmd(proc, "setoption name QS_Checks_Enabled value false")

    send_cmd(proc, "isready")
    wait_for(proc, "readyok")
    return proc

def query_move(proc, fen, movetime=300):
    send_cmd(proc, f"position fen {fen}")
    send_cmd(proc, f"go movetime {movetime}")
    bestmove = None
    last_score = "0.00"
    aprm_info = ""
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if "info string aprm" in line:
            aprm_info = line.replace("info string aprm", "").strip()
        if "score cp" in line:
            parts = line.split()
            if "cp" in parts:
                idx = parts.index("cp")
                if idx + 1 < len(parts):
                    try:
                        cp = int(parts[idx+1])
                        last_score = f"{cp/100.0:+.2f}"
                    except ValueError:
                        pass
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) > 1:
                bestmove = parts[1]
            break
    return bestmove, last_score, aprm_info

def play_game(engine_path, white_aprm=True, movetime=300, max_plies=120):
    proc_w = init_engine(engine_path, is_aprm=white_aprm)
    proc_b = init_engine(engine_path, is_aprm=not white_aprm)

    board = chess.Board()
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "APRM Validation Tournament"
    pgn_game.headers["White"] = "Whale-APRM-ON" if white_aprm else "Whale-Baseline-OFF"
    pgn_game.headers["Black"] = "Whale-Baseline-OFF" if white_aprm else "Whale-APRM-ON"

    node = pgn_game
    plies = 0

    while not board.is_game_over() and plies < max_plies:
        fen = board.fen()
        proc = proc_w if board.turn == chess.WHITE else proc_b
        mv_str, score, aprm = query_move(proc, fen, movetime)
        if not mv_str or mv_str == "(none)":
            break
        try:
            mv = chess.Move.from_uci(mv_str)
            if mv not in board.legal_moves:
                break
        except Exception:
            break

        board.push(mv)
        node = node.add_variation(mv)
        comment = f"eval={score}"
        if aprm:
            comment += f" | {aprm}"
        node.comment = comment
        plies += 1

    send_cmd(proc_w, "quit")
    send_cmd(proc_b, "quit")
    proc_w.terminate()
    proc_b.terminate()

    result = board.result(claim_draw=True)
    pgn_game.headers["Result"] = result
    return pgn_game, result

def main():
    engine_bin = sys.argv[1] if len(sys.argv) > 1 else "./target/release/whale.exe"
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    movetime = int(sys.argv[3]) if len(sys.argv) > 3 else 250

    if not os.path.exists(engine_bin):
        engine_bin = "./target/debug/whale.exe"

    print("=" * 65)
    print("      APRM TOURNAMENT HARNESS: APRM-ON vs APRM-OFF (Baseline)")
    print(f" Engine binary : {engine_bin}")
    print(f" Games count   : {num_games} | Movetime: {movetime}ms")
    print("=" * 65)

    out_pgn = "aprm_match.pgn"
    score_aprm = 0.0
    score_base = 0.0

    with open(out_pgn, "w", encoding="utf-8") as f:
        for g_idx in range(1, num_games + 1):
            aprm_is_white = (g_idx % 2 == 1)
            print(f"Game {g_idx}/{num_games}: APRM as {'White' if aprm_is_white else 'Black'} ... ", end="", flush=True)
            game, res = play_game(engine_bin, white_aprm=aprm_is_white, movetime=movetime)
            f.write(str(game) + "\n\n")

            if res == "1-0":
                if aprm_is_white:
                    score_aprm += 1.0
                else:
                    score_base += 1.0
            elif res == "0-1":
                if aprm_is_white:
                    score_base += 1.0
                else:
                    score_aprm += 1.0
            else:
                score_aprm += 0.5
                score_base += 0.5

            print(f"Result: {res} (APRM {score_aprm} - {score_base} Base)")

    print("\n" + "#" * 65)
    print(f"FINAL SCORE: APRM-ON: {score_aprm} | Baseline-OFF: {score_base}")
    print(f"Games saved to '{out_pgn}'. Running 16-Metrics Game Analyzer...")
    print("#" * 65)

    os.system(f"python tools/aprm_game_analyzer.py {out_pgn} --json aprm_metrics.json")
    try:
        with open("aprm_metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception:
        metrics = {}
    report = {
        "engine": engine_bin,
        "games": num_games,
        "movetime_ms": movetime,
        "aprm_score": score_aprm,
        "baseline_score": score_base,
        "metrics": metrics,
    }
    with open("sprt_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("Wrote evidence report to 'sprt_report.json'.")

if __name__ == "__main__":
    main()
