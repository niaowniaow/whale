import argparse
import subprocess
import sys

import chess
import chess.pgn


def send_cmd(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def wait_for(proc, target):
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if target in line:
            return line


def init_engine(binary, options):
    proc = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    send_cmd(proc, "uci")
    wait_for(proc, "uciok")
    for name, value in options:
        send_cmd(proc, f"setoption name {name} value {value}")
    send_cmd(proc, "isready")
    wait_for(proc, "readyok")
    return proc


def query_move(proc, fen, movetime):
    send_cmd(proc, f"position fen {fen}")
    send_cmd(proc, f"go movetime {movetime}")
    bestmove = None
    score_cp = None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if "score cp" in line:
            parts = line.split()
            try:
                score_cp = int(parts[parts.index("cp") + 1])
            except (ValueError, IndexError):
                pass
        elif "score mate" in line:
            parts = line.split()
            try:
                mate_in = int(parts[parts.index("mate") + 1])
                score_cp = 30000 if mate_in > 0 else -30000
            except (ValueError, IndexError):
                pass
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) > 1:
                bestmove = parts[1]
            break
    return bestmove, score_cp


def adjudicate(scores_white, plies):
    if len(scores_white) >= 8:
        window = scores_white[-8:]
        if all(s is not None and s >= 700 for s in window):
            return "1-0"
        if all(s is not None and s <= -700 for s in window):
            return "0-1"
    if plies >= 100 and len(scores_white) >= 16:
        window = scores_white[-16:]
        if all(s is not None and abs(s) <= 25 for s in window):
            return "1/2-1/2"
    return None


def play_game(proc_a, proc_b, book_fen, a_is_white, movetime, max_plies):
    board = chess.Board(book_fen)
    moves = []
    plies = 0
    method = "natural"
    while not board.is_game_over() and plies < max_plies:
        proc = proc_a if (board.turn == chess.WHITE) == a_is_white else proc_b
        mv_str, score_cp = query_move(proc, board.fen(), movetime)
        score_white = None
        if score_cp is not None:
            score_white = score_cp if board.turn == chess.WHITE else -score_cp
        if not mv_str or mv_str == "(none)":
            break
        try:
            mv = chess.Move.from_uci(mv_str)
            if mv not in board.legal_moves:
                print(f"illegal move {mv_str} in {board.fen()}", flush=True)
                break
        except Exception:
            break
        board.push(mv)
        moves.append((mv.uci(), score_white))
        plies += 1
        adjudicated = adjudicate(
            [s for _, s in moves],
            plies,
        )
        if adjudicated:
            method = "adjudication"
            return adjudicated, moves, method
    if plies >= max_plies and not board.is_game_over():
        method = "plycap"
        return "1/2-1/2", moves, method
    return board.result(claim_draw=True), moves, method


def node_moves(board, scores_white):
    return (board, scores_white)


def parse_opt(text):
    name, _, value = text.partition("=")
    return name.strip(), value.strip()


def main():
    ap = argparse.ArgumentParser(description="A/B self-play match with per-side UCI options")
    ap.add_argument("--binary", default="./target/release/whale")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--movetime", type=int, default=300)
    ap.add_argument("--book", default="resources/openings.epd")
    ap.add_argument("--hash", type=int, default=16)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--off-opt", action="append", default=[],
                    help="NAME=value applied to side B only, e.g. BMO_Enabled=false")
    ap.add_argument("--out", default="sprt_match.pgn")
    ap.add_argument("--label-a", default="BASE")
    ap.add_argument("--label-b", default="CHALLENGER")
    args = ap.parse_args()

    with open(args.book) as f:
        book = [line.strip() for line in f if line.strip()]
    print(f"book: {args.book} ({len(book)} positions)", flush=True)

    base_opts = [("Threads", "1"), ("Hash", str(args.hash)), ("UseBook", "false")]
    chall_opts = base_opts + [parse_opt(o) for o in args.off_opt]

    proc_a = init_engine(args.binary, base_opts)
    proc_b = init_engine(args.binary, chall_opts)

    score_a = 0.0
    score_b = 0.0
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for g in range(1, args.games + 1):
                fen = book[(g - 1) % len(book)]
                a_is_white = (g % 2 == 1)
                res, moves, method = play_game(proc_a, proc_b, fen, a_is_white, args.movetime, args.max_plies)
                if res == "1-0":
                    if a_is_white:
                        score_a += 1.0
                    else:
                        score_b += 1.0
                elif res == "0-1":
                    if a_is_white:
                        score_b += 1.0
                    else:
                        score_a += 1.0
                else:
                    score_a += 0.5
                    score_b += 0.5
                game = chess.pgn.Game()
                game.headers["Event"] = f"{args.label_a} vs {args.label_b}"
                game.headers["White"] = args.label_a if a_is_white else args.label_b
                game.headers["Black"] = args.label_b if a_is_white else args.label_a
                game.headers["Result"] = res
                game.headers["FEN"] = fen
                game.headers["SetUp"] = "1"
                game.headers["Adjudication"] = method
                node = game
                for uci, score_white in moves:
                    node = node.add_variation(chess.Move.from_uci(uci))
                    if score_white is not None:
                        node.comment = f"eval={score_white / 100.0:+.2f}"
                f.write(str(game) + "\n\n")
                f.flush()
                print(f"game {g}/{args.games}: {res} ({args.label_a} {score_a} - {score_b} {args.label_b})",
                      flush=True)
    finally:
        for proc in (proc_a, proc_b):
            try:
                send_cmd(proc, "quit")
                proc.terminate()
            except Exception:
                pass

    print(f"FINAL: {args.label_a} {score_a} - {score_b} {args.label_b}")


if __name__ == "__main__":
    main()
