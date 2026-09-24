import argparse
import math
import os
import random
import sys

PARAMS = {
    "PawnValue": {"default": 100, "min": 60, "max": 160, "step": 2},
    "KnightValue": {"default": 320, "min": 250, "max": 420, "step": 4},
    "BishopValue": {"default": 330, "min": 250, "max": 430, "step": 4},
    "RookValue": {"default": 500, "min": 400, "max": 620, "step": 5},
    "QueenValue": {"default": 1000, "min": 850, "max": 1200, "step": 10},
    "TempoBonus": {"default": 20, "min": 0, "max": 60, "step": 2},
    "Razor_Margin": {"default": 350, "min": 100, "max": 800, "step": 20},
    "NMP_Verify_Margin": {"default": 150, "min": 0, "max": 500, "step": 15},
}


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def static_score(counts, theta):
    w = (
        counts["P"] * theta["PawnValue"]
        + counts["N"] * theta["KnightValue"]
        + counts["B"] * theta["BishopValue"]
        + counts["R"] * theta["RookValue"]
        + counts["Q"] * theta["QueenValue"]
    )
    b = (
        counts["p"] * theta["PawnValue"]
        + counts["n"] * theta["KnightValue"]
        + counts["b"] * theta["BishopValue"]
        + counts["r"] * theta["RookValue"]
        + counts["q"] * theta["QueenValue"]
    )
    diff = (w - b) + theta["TempoBonus"]
    return diff if counts["stm"] == "w" else -diff


def parse_fen_counts(fen):
    placement = fen.split()[0]
    counts = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0,
              "p": 0, "n": 0, "b": 0, "r": 0, "q": 0, "stm": "w"}
    for ch in placement:
        if ch in counts:
            counts[ch] += 1
    parts = fen.split()
    if len(parts) > 1 and parts[1] in ("w", "b"):
        counts["stm"] = parts[1]
    return counts


def load_dataset(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " bm " in line:
                continue
            if "," in line:
                fen, res = line.rsplit(",", 1)
            else:
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                fen, res = parts
            try:
                r = float(res.strip().rstrip(";"))
            except ValueError:
                continue
            if r not in (0.0, 0.5, 1.0):
                continue
            data.append((fen.strip(), r))
    return data


def mse(data, theta, k):
    tot = 0.0
    for fen, result in data:
        s = static_score(parse_fen_counts(fen), theta) / 400.0
        e = sigmoid(k * s)
        tot += (result - e) ** 2
    return tot / max(1, len(data))


def fit_k(data, theta):
    best_k, best_err = 1.0, float("inf")
    k = 0.2
    while k <= 4.0:
        err = mse(data, theta, k)
        if err < best_err:
            best_err, best_k = err, k
        k += 0.1
    return best_k, best_err


def coordinate_descent(data, theta, k, iters, seed=1):
    rng = random.Random(seed)
    theta = dict(theta)
    cur = mse(data, theta, k)
    names = list(PARAMS.keys())
    for it in range(iters):
        rng.shuffle(names)
        improved = False
        for name in names:
            cfg = PARAMS[name]
            for direction in (+1, -1):
                cand = dict(theta)
                cand[name] = max(cfg["min"], min(cfg["max"], theta[name] + direction * cfg["step"]))
                if cand[name] == theta[name]:
                    continue
                err = mse(data, theta=cand, k=k)
                if err < cur:
                    theta, cur = cand, err
                    improved = True
                    break
        print(f"  iter {it + 1:03d} mse={cur:.6f} theta={theta}", flush=True)
        if not improved:
            print("  converged (no single-step improvement).")
            break
    return theta, cur


def gen_sample(path, n, seed=7):
    rng = random.Random(seed)
    pieces = "PNBRQpnbrq"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(n):
            squares = ["1"] * 64
            for _ in range(rng.randint(4, 14)):
                sq = rng.randrange(64)
                squares[sq] = rng.choice(pieces)
            placement = "/".join("".join(squares[r * 8:(r + 1) * 8]) for r in range(8))
            rows = []
            for row in placement.split("/"):
                out, run = "", 0
                for ch in row:
                    if ch == "1":
                        run += 1
                    else:
                        if run:
                            out += str(run)
                            run = 0
                        out += ch
                if run:
                    out += str(run)
                rows.append(out or "8")
            stm = rng.choice("wb")
            fen = f"{'/'.join(rows)} {stm} - - 0 1"
            counts = parse_fen_counts(fen)
            true_theta = {k: v["default"] for k, v in PARAMS.items()}
            s = static_score(counts, true_theta) / 400.0
            p = sigmoid(1.0 * s)
            r = rng.random()
            result = 1.0 if r < p else (0.0 if r > p + 0.15 else 0.5)
            f.write(f"{fen},{result}\n")
    print(f"wrote {n} synthetic positions to {path}")


def main():
    ap = argparse.ArgumentParser(description="Texel-style tuner for classical weights")
    ap.add_argument("--data", default="", help="EPD/CSV file with fen,result lines")
    ap.add_argument("--sample", type=int, default=0, help="generate N synthetic lines and tune them")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fit-k-only", action="store_true")
    ap.add_argument("--out", default="", help="write tuned params as JSON")
    args = ap.parse_args()

    data_path = args.data
    if args.sample:
        data_path = data_path or "texel_sample.epd"
        gen_sample(data_path, args.sample, seed=args.seed)
    if not data_path or not os.path.exists(data_path):
        print("no dataset: pass --data <file> or --sample <n>", file=sys.stderr)
        return 2
    data = load_dataset(data_path)
    print(f"loaded {len(data)} positions from {data_path}")
    if not data:
        print("empty dataset", file=sys.stderr)
        return 2

    theta = {k: float(v["default"]) for k, v in PARAMS.items()}
    k, err = fit_k(data, theta)
    print(f"fitted K={k:.2f} mse={err:.6f}")
    if args.fit_k_only:
        return 0
    theta, err = coordinate_descent(data, theta, k, args.iters, seed=args.seed)
    print(f"final K={k:.2f} mse={err:.6f}")
    for name in PARAMS:
        print(f"  {name} = {int(theta[name])}  (setoption name {name} value {int(theta[name])})")
    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({k: int(v) for k, v in theta.items()} | {"TexelK": k}, f, indent=2)
        print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
