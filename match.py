import chess
import chess.engine
import chess.pgn
import sys
import os
import math
import time

MINE_EXE = os.path.abspath("target/release/rudim.exe")
ORIG_EXE = os.path.abspath("rudim-v305-orig.exe")

def calculate_elo(score, total_games):
    if score <= 0:
        return -999.0
    if score >= total_games:
        return 999.0
    p = score / total_games
    return -400.0 * math.log10(1.0 / p - 1.0)

def play_match(games=10, time_limit=2.0, depth=None):
    print("=" * 60)
    print("      RUDIM NEW (SFNNv16) VS RUDIM v3.0.5 ORIGINAL")
    print("=" * 60)
    if depth:
        print(f"Games: {games}, Fixed Depth: {depth}")
    else:
        print(f"Games: {games}, Time per move: {time_limit}s")
    print(f"Mine:   {MINE_EXE}")
    print(f"Orig:   {ORIG_EXE}")
    print("=" * 60)

    wins = 0
    draws = 0
    losses = 0

    for g in range(1, games + 1):
        board = chess.Board()
        mine_is_white = (g % 2 == 1)

        white_name = "Rudim-Mine" if mine_is_white else "Rudim-v3.0.5"
        black_name = "Rudim-v3.0.5" if mine_is_white else "Rudim-Mine"

        # Start engines
        mine_engine = chess.engine.SimpleEngine.popen_uci(MINE_EXE)
        mine_engine.configure({"EvalFile": os.path.abspath("v16/nn-1a298aa575a0.nnue")})
        orig_engine = chess.engine.SimpleEngine.popen_uci(ORIG_EXE)

        engines = {
            True: (mine_engine if mine_is_white else orig_engine),
            False: (orig_engine if mine_is_white else mine_engine),
        }

        print(f"\n[Game {g}/{games}] {white_name} (W) vs {black_name} (B)...", flush=True)

        move_count = 0
        limit = chess.engine.Limit(depth=depth) if depth else chess.engine.Limit(time=time_limit)

        game_moves = []
        while not board.is_game_over(claim_draw=True) and move_count < 200:
            current_engine = engines[board.turn]
            result = current_engine.play(board, limit)
            if result.move is None:
                break
            san = board.san(result.move)
            board.push(result.move)
            game_moves.append(result.move)
            move_count += 1
            if move_count % 10 == 0:
                print(f"  ply {move_count}: {san} ({result.move.uci()})", flush=True)

        mine_engine.quit()
        orig_engine.quit()

        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            res_str = "1/2-1/2 (Move limit reached)"
            draws += 1
        elif outcome.winner is None:
            res_str = f"1/2-1/2 ({outcome.termination.name})"
            draws += 1
        elif outcome.winner == chess.WHITE:
            if mine_is_white:
                res_str = f"1-0 (Mine won by {outcome.termination.name})"
                wins += 1
            else:
                res_str = f"0-1 (Orig won by {outcome.termination.name})"
                losses += 1
        else: # Black won
            if not mine_is_white:
                res_str = f"1-0 (Mine won as Black by {outcome.termination.name})"
                wins += 1
            else:
                res_str = f"0-1 (Orig won as Black by {outcome.termination.name})"
                losses += 1

        print(f"Result: {res_str}")

        # Save to PGN
        game = chess.pgn.Game()
        game.headers["Event"] = "Rudim NNUE Benchmark"
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = outcome.result() if outcome else "1/2-1/2"
        node = game
        b_replay = chess.Board()
        for m in game_moves:
            node = node.add_variation(m)
            b_replay.push(m)
        with open("match.pgn", "a") as f:
            print(game, file=f, end="\n\n")
        total = wins + draws + losses
        score = wins + 0.5 * draws
        elo = calculate_elo(score, total)
        print(f"Current Score: +{wins} ={draws} -{losses} ({score}/{total} points) | Elo Diff: {elo:+.1f}")

    print("\n" + "=" * 60)
    print("                    FINAL RESULT")
    print("=" * 60)
    print(f"Mine:   +{wins} ={draws} -{losses} ({score}/{games})")
    print(f"Orig:   +{losses} ={draws} -{wins} ({games - score}/{games})")
    elo = calculate_elo(score, games)
    print(f"Estimated Elo Difference: {elo:+.1f}")
    print("=" * 60)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "d7"
    if arg2.startswith("d"):
        play_match(games=n, depth=int(arg2[1:]))
    else:
        play_match(games=n, time_limit=float(arg2))

