import json
import math
import os
import subprocess
import sys

import chess
import chess.engine
import chess.pgn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "baseline", "phase0.json")
RESULT = os.path.join(ROOT, "baseline", "phase0_result.json")
PGN_OUT = os.path.join(ROOT, "baseline", "phase0.pgn")


def resolve(name, default):
    v = os.environ.get(name)
    if v:
        return v
    return os.path.join(ROOT, default)


def load_book(path, limit):
    fens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                fens.append(line.split(";")[0].strip())
            if len(fens) >= limit:
                break
    return fens


def elo_diff(score, total):
    if total <= 0:
        return 0.0
    if score <= 0:
        return -999.0
    if score >= total:
        return 999.0
    p = score / total
    return -400.0 * math.log10(1.0 / p - 1.0)


def smoke_uci(binary):
    proc = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def wait(token):
        while True:
            line = proc.stdout.readline()
            if not line:
                return None
            if token in line.strip():
                return line.strip()

    send("uci")
    wait("uciok")
    send("isready")
    wait("readyok")
    send("quit")
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    return True


def main():
    with open(MANIFEST) as f:
        cfg = json.load(f)
    whale = resolve("WHALE_ENGINE", cfg["whale_binary"])
    rudim = os.environ.get("RUDIM_ENGINE")
    games = int(os.environ.get("PHASE0_GAMES", "4"))
    depth = int(os.environ.get("PHASE0_DEPTH", "12"))
    smoke_uci(whale)
    if not rudim or not os.path.exists(rudim):
        report = {
            "whale": whale,
            "whale_commit": cfg["whale_commit"],
            "rudim": rudim,
            "status": "blocked_missing_rudim",
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "elo_diff": 0.0,
            "pgn": None,
        }
        with open(RESULT, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2))
        return 3
    book = load_book(os.path.join(ROOT, "resources", "openings.epd"), games)
    if os.path.exists(PGN_OUT):
        os.remove(PGN_OUT)
    wins = 0
    draws = 0
    losses = 0
    for g in range(games):
        fen = book[g % len(book)] if book else chess.STARTING_FEN
        board = chess.Board(fen)
        white_is_whale = (g % 2 == 0)
        eng_w = chess.engine.SimpleEngine.popen_uci(whale)
        eng_b = chess.engine.SimpleEngine.popen_uci(rudim)
        try:
            eng_w.configure({"Hash": 16, "Threads": 1})
            eng_b.configure({"Hash": 16, "Threads": 1})
        except Exception:
            pass
        limit = chess.engine.Limit(depth=depth)
        moves = []
        while not board.is_game_over(claim_draw=True) and len(moves) < 200:
            eng = eng_w if (board.turn == chess.WHITE) == white_is_whale else eng_b
            res = eng.play(board, limit)
            if res.move is None:
                break
            board.push(res.move)
            moves.append(res.move)
        eng_w.quit()
        eng_b.quit()
        outcome = board.outcome(claim_draw=True)
        whale_score = 0.5
        if outcome is not None and outcome.winner is not None:
            whale_won = (outcome.winner == chess.WHITE) == white_is_whale
            whale_score = 1.0 if whale_won else 0.0
        if whale_score == 1.0:
            wins += 1
        elif whale_score == 0.0:
            losses += 1
        else:
            draws += 1
        game = chess.pgn.Game()
        game.headers["Event"] = "whale phase0 baseline"
        game.headers["White"] = "whale" if white_is_whale else "rudim"
        game.headers["Black"] = "rudim" if white_is_whale else "whale"
        game.headers["Result"] = outcome.result() if outcome else "1/2-1/2"
        game.headers["FEN"] = fen
        node = game
        replay = chess.Board(fen)
        for m in moves:
            node = node.add_variation(m)
            replay.push(m)
        with open(PGN_OUT, "a") as f:
            print(game, file=f, end="\n\n")
    total = wins + draws + losses
    score = wins + 0.5 * draws
    report = {
        "whale": whale,
        "whale_commit": cfg["whale_commit"],
        "rudim": rudim,
        "status": "done",
        "games": total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "elo_diff": round(elo_diff(score, total), 1),
        "pgn": PGN_OUT,
    }
    with open(RESULT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
