import argparse
import json
import os
import sys

try:
    import numpy as np
except ImportError:
    print("error: numpy is required (pip install numpy)", file=sys.stderr)
    sys.exit(2)

FEATURES = [
    "eval_stm",
    "pressure",
    "opp_cpi",
    "own_cpi",
    "urgency",
    "risk",
    "musttry",
    "concession",
    "state",
    "intent",
    "side",
]


def scale_row(rec):
    side = 1.0 if rec.get("side") == "w" else -1.0
    ev = float(rec.get("eval", 0))
    eval_stm = ev if side > 0 else -ev
    return [
        eval_stm / 100.0,
        float(rec.get("pressure", 0)) / 50.0,
        float(rec.get("opp_cpi", 0)) / 50.0,
        float(rec.get("own_cpi", 0)) / 50.0,
        float(rec.get("urgency", 0)),
        float(rec.get("risk", 0)) / 100.0,
        1.0 if rec.get("musttry") is True else 0.0,
        float(rec.get("concession", 0)) / 50.0,
        float(rec.get("state", 0)),
        float(rec.get("intent", 0)),
        side,
    ]


def load_records(path):
    rows, targets = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if "result" not in rec:
                continue
            try:
                y = float(rec["result"])
            except Exception:
                continue
            if y not in (0.0, 0.5, 1.0):
                continue
            rows.append(scale_row(rec))
            targets.append(y)
    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def train_logreg(X, y, epochs=200, lr=0.1, l2=1e-4, seed=1):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds < 1e-9] = 1.0
    Xs = (X - means) / stds
    w = np.zeros(d)
    b = 0.0
    order = np.arange(n)
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            p = sigmoid(float(Xs[i] @ w + b))
            err = p - y[i]
            w -= lr * (err * Xs[i] + l2 * w)
            b -= lr * err
    return w, b, means, stds


def logloss(Xs, y, w, b):
    p = sigmoid(Xs @ w + b)
    eps = 1e-9
    p = np.clip(p, eps, 1.0 - eps)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def main():
    parser = argparse.ArgumentParser(description="Fit learned win-probability head from behavior jsonl")
    parser.add_argument("input", help="behavior .jsonl with result labels")
    parser.add_argument("--out", default="heads_weights.json", help="weights JSON output")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    X, y = load_records(args.input)
    if len(X) < 50:
        print(f"error: only {len(X)} labeled rows, need >= 50", file=sys.stderr)
        sys.exit(2)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(X) * (1.0 - args.val_frac))
    Xtr, ytr = X[idx[:cut]], y[idx[:cut]]
    Xva, yva = X[idx[cut:]], y[idx[cut:]]

    w, b, means, stds = train_logreg(Xtr, ytr, epochs=args.epochs, lr=args.lr, l2=args.l2, seed=args.seed)
    Xts = (Xtr - means) / stds
    Xvs = (Xva - means) / stds
    tr_loss = logloss(Xts, ytr, w, b)
    va_loss = logloss(Xvs, yva, w, b)
    va_acc = float((((sigmoid(Xvs @ w + b) > 0.5).astype(float) - yva) == 0).mean())

    payload = {
        "features": FEATURES,
        "w": [float(v) for v in w],
        "b": float(b),
        "means": [float(v) for v in means],
        "stds": [float(v) for v in stds],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"rows={len(X)} train={len(Xtr)} val={len(Xva)}")
    print(f"train_logloss={tr_loss:.4f} val_logloss={va_loss:.4f} val_acc={va_acc:.3f}")
    print(f"Wrote weights to {args.out}")


if __name__ == "__main__":
    main()
