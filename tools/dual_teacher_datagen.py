import argparse
import os
import subprocess
import time
import json
import math

def run_uci_eval(engine_proc, fen, movetime_ms=50):
    engine_proc.stdin.write(f"position fen {fen}\ngo movetime {movetime_ms}\n")
    engine_proc.stdin.flush()
    score = 0
    wdl = (500, 500, 0)
    bestmove = None
    while True:
        line = engine_proc.stdout.readline().strip()
        if not line:
            break
        if line.startswith("info"):
            parts = line.split()
            if "score" in parts:
                idx = parts.index("score")
                if parts[idx+1] == "cp":
                    score = int(parts[idx+2])
                elif parts[idx+1] == "mate":
                    m = int(parts[idx+2])
                    score = 30000 if m > 0 else -30000
            if "wdl" in parts:
                idx = parts.index("wdl")
                wdl = (int(parts[idx+1]), int(parts[idx+2]), int(parts[idx+3]))
        if line.startswith("bestmove"):
            bestmove = line.split()[1]
            break
    return score, wdl, bestmove

def distill_stockfish(sf_path, book_path, out_path, max_hours):
    print(f"Starting Stockfish distillation (Max: {max_hours} hours)...")
    start_time = time.time()
    max_seconds = max_hours * 3600
    
    proc = subprocess.Popen(
        [sf_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
        bufsize=1
    )
    proc.stdin.write("uci\nsetoption name Threads value 4\nisready\n")
    proc.stdin.flush()
    
    with open(book_path, "r") as f_in, open(out_path, "a", encoding="utf-8") as f_out:
        count = 0
        for line in f_in:
            if time.time() - start_time >= max_seconds:
                print(f"Reached time limit of {max_hours} hours. Gracefully stopping...")
                break
            fen = line.strip()
            if not fen or fen.startswith("#"):
                continue
            
            score, _, bestmove = run_uci_eval(proc, fen, movetime_ms=30)
            record = {
                "fen": fen,
                "teacher": "stockfish",
                "score": score,
                "bestmove": bestmove
            }
            f_out.write(json.dumps(record) + "\n")
            count += 1
            if count % 1000 == 0:
                f_out.flush()
                elapsed_min = (time.time() - start_time) / 60
                print(f"Stockfish: {count} positions distilled ({elapsed_min:.1f} min)")
                
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.terminate()
    print(f"Stockfish distillation complete. Saved {count} positions to {out_path}")

def distill_lc0(lc0_path, weights_path, book_path, out_path, max_hours):
    print(f"Starting Lc0 distillation on GPU (Max: {max_hours} hours)...")
    start_time = time.time()
    max_seconds = max_hours * 3600
    
    cmd = [lc0_path]
    if weights_path:
        cmd.extend([f"--weights={weights_path}"])
        
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
        bufsize=1
    )
    proc.stdin.write("uci\nisready\n")
    proc.stdin.flush()
    
    with open(book_path, "r") as f_in, open(out_path, "a", encoding="utf-8") as f_out:
        count = 0
        for line in f_in:
            if time.time() - start_time >= max_seconds:
                print(f"Reached time limit of {max_hours} hours. Gracefully stopping...")
                break
            fen = line.strip()
            if not fen or fen.startswith("#"):
                continue
            
            score, wdl, bestmove = run_uci_eval(proc, fen, movetime_ms=40)
            record = {
                "fen": fen,
                "teacher": "lc0",
                "score": score,
                "wdl": wdl,
                "bestmove": bestmove
            }
            f_out.write(json.dumps(record) + "\n")
            count += 1
            if count % 1000 == 0:
                f_out.flush()
                elapsed_min = (time.time() - start_time) / 60
                print(f"Lc0: {count} positions distilled ({elapsed_min:.1f} min)")
                
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.terminate()
    print(f"Lc0 distillation complete. Saved {count} positions to {out_path}")

def merge_datasets(sf_data_path, lc0_data_path, merged_out_path):
    print("Merging Stockfish and Lc0 datasets into Hybrid Training Dataset...")
    sf_data = {}
    if os.path.exists(sf_data_path):
        with open(sf_data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    sf_data[rec["fen"]] = rec

    merged_count = 0
    with open(merged_out_path, "w", encoding="utf-8") as f_out:
        if os.path.exists(lc0_data_path):
            with open(lc0_data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    lc0_rec = json.loads(line)
                    fen = lc0_rec["fen"]
                    
                    if fen in sf_data:
                        sf_rec = sf_data[fen]
                        blended_score = int(round(0.35 * sf_rec["score"] + 0.65 * lc0_rec["score"]))
                        bestmove = lc0_rec["bestmove"] or sf_rec["bestmove"]
                        wdl = lc0_rec.get("wdl", (500, 500, 0))
                    else:
                        blended_score = lc0_rec["score"]
                        bestmove = lc0_rec["bestmove"]
                        wdl = lc0_rec.get("wdl", (500, 500, 0))
                        
                    merged_record = {
                        "fen": fen,
                        "score": blended_score,
                        "wdl": wdl,
                        "bestmove": bestmove
                    }
                    f_out.write(json.dumps(merged_record) + "\n")
                    merged_count += 1

    print(f"Merging complete! Generated {merged_count} hybrid-labeled positions at {merged_out_path}")

def main():
    parser = argparse.ArgumentParser(description="Dual Teacher Distillation Pipeline")
    parser.add_argument("--mode", choices=["stockfish", "lc0", "merge"], required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--lc0", default="lc0")
    parser.add_argument("--lc0-weights", default="")
    parser.add_argument("--book", default="data/books/UHO_Lichess_4852_v1.epd")
    parser.add_argument("--sf-out", default="data/stockfish_distill.jsonl")
    parser.add_argument("--lc0-out", default="data/lc0_distill.jsonl")
    parser.add_argument("--merged-out", default="data/whale_hybrid_dataset.jsonl")
    parser.add_argument("--max-hours", type=float, default=10.0)
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    if args.mode == "stockfish":
        distill_stockfish(args.stockfish, args.book, args.sf_out, args.max_hours)
    elif args.mode == "lc0":
        distill_lc0(args.lc0, args.lc0_weights, args.book, args.lc0_out, args.max_hours)
    elif args.mode == "merge":
        merge_datasets(args.sf_out, args.lc0_out, args.merged_out)

if __name__ == "__main__":
    main()
