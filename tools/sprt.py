import argparse
import json
import math
import re
import sys

PAIR_SCORES = {
    "LL": 0.0,
    "LD": 0.25,
    "WL": 0.5,
    "DD": 0.5,
    "WD": 0.75,
    "WW": 1.0,
}


def bayeselo_to_proba(elo, draw_elo):
    pwin = 1.0 / (1.0 + pow(10.0, (-elo + draw_elo) / 400.0))
    ploss = 1.0 / (1.0 + pow(10.0, (elo + draw_elo) / 400.0))
    pdraw = max(0.0, 1.0 - pwin - ploss)
    return pwin, pdraw, ploss


def penta_probs(elo, draw_elo):
    w, d, l = bayeselo_to_proba(elo, draw_elo)
    return {
        "LL": l * l,
        "LD": 2.0 * l * d,
        "WL": 2.0 * w * l,
        "DD": d * d,
        "WD": 2.0 * w * d,
        "WW": w * w,
    }


def game_to_points(result):
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def flip_points(p):
    return 1.0 - p


def normalize_pair_key(first, second):
    if first is None or second is None:
        return None
    if first == 1.0 and second == 1.0:
        return "WW"
    if first == 0.0 and second == 0.0:
        return "LL"
    if first == 0.5 and second == 0.5:
        return "DD"
    if sorted([first, second]) == [0.0, 0.5]:
        return "LD"
    if sorted([first, second]) == [0.5, 1.0]:
        return "WD"
    return "WL"


def parse_pgn_results(path):
    results = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = re.match(r'\[Result\s+"([^"]+)"\]', line)
            if m and m.group(1) in ("1-0", "0-1", "1/2-1/2", "*"):
                if m.group(1) != "*":
                    results.append(m.group(1))
    return results


def results_to_white_points(results, white_is_player_one_pattern=None):
    points = []
    for i, r in enumerate(results):
        p = game_to_points(r)
        if white_is_player_one_pattern is not None:
            if not white_is_player_one_pattern(i):
                p = flip_points(p)
        points.append(p)
    return points


def points_to_pairs(points):
    pairs = []
    for i in range(0, len(points) - 1, 2):
        key = normalize_pair_key(points[i], points[i + 1])
        if key is not None:
            pairs.append(key)
    return pairs


def count_pairs(pairs):
    counts = {k: 0 for k in PAIR_SCORES}
    for p in pairs:
        counts[p] += 1
    return counts


def llr(counts, elo0, elo1, draw_elo):
    p0 = penta_probs(elo0, draw_elo)
    p1 = penta_probs(elo1, draw_elo)
    total = 0.0
    for k, n in counts.items():
        if n > 0:
            total += n * (math.log(max(p1[k], 1e-12)) - math.log(max(p0[k], 1e-12)))
    return total


def bounds(alpha=0.05, beta=0.05):
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    return lower, upper


def verdict(llr_value, lower, upper):
    if llr_value >= upper:
        return "H1"
    if llr_value <= lower:
        return "H0"
    return "continue"


def elo_estimate(counts, draw_elo=200.0):
    n = sum(counts.values())
    if n == 0:
        return 0.0, 0.0
    score = sum(PAIR_SCORES[k] * v for k, v in counts.items()) / n
    score = min(max(score, 0.001), 0.999)
    lo, hi = -800.0, 800.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        w, d, l = bayeselo_to_proba(mid, draw_elo)
        s = w + 0.5 * d
        if s < score:
            lo = mid
        else:
            hi = mid
    elo = (lo + hi) / 2.0
    var = 0.0
    for k, v in counts.items():
        var += v * (PAIR_SCORES[k] - score) ** 2
    var = var / n if n > 0 else 0.0
    err = 400.0 * math.log10(math.e) * math.sqrt(max(var, 1e-12) / max(n, 1)) * 1.96
    return elo, err


def analyze_pgn(path, elo0=0.0, elo1=5.0, draw_elo=200.0, alpha=0.05, beta=0.05,
                alternate_colors=True):
    results = parse_pgn_results(path)
    if alternate_colors:
        points = results_to_white_points(
            results, white_is_player_one_pattern=lambda i: i % 2 == 0)
    else:
        points = results_to_white_points(results)
    pairs = points_to_pairs(points)
    counts = count_pairs(pairs)
    value = llr(counts, elo0, elo1, draw_elo)
    lower, upper = bounds(alpha, beta)
    elo, err = elo_estimate(counts, draw_elo)
    return {
        "games": len(results),
        "pairs": len(pairs),
        "counts": counts,
        "llr": value,
        "lower": lower,
        "upper": upper,
        "verdict": verdict(value, lower, upper),
        "elo": elo,
        "elo_err": err,
        "elo0": elo0,
        "elo1": elo1,
        "draw_elo": draw_elo,
    }


def self_test():
    strong = ["WW"] * 60 + ["WD"] * 45 + ["DD"] * 30 + ["WL"] * 12 + ["LD"] * 3
    c = count_pairs(strong)
    v = llr(c, 0.0, 5.0, 200.0)
    _, upper = bounds()
    assert v > upper, f"expected H1, llr={v}"
    assert verdict(v, *bounds()) == "H1"
    weak = ["LL"] * 60 + ["LD"] * 45 + ["DD"] * 30 + ["WL"] * 12 + ["WD"] * 3
    c = count_pairs(weak)
    v = llr(c, 0.0, 5.0, 200.0)
    lower, _ = bounds()
    assert v < lower, f"expected H0, llr={v}"
    assert verdict(v, *bounds()) == "H0"
    even = ["WW"] * 10 + ["LL"] * 10 + ["DD"] * 40 + ["WD"] * 10 + ["LD"] * 10 + ["WL"] * 20
    c = count_pairs(even)
    v = llr(c, 0.0, 5.0, 200.0)
    lower, upper = bounds()
    assert lower < v < upper, f"expected continue, llr={v}"
    assert normalize_pair_key(1.0, 1.0) == "WW"
    assert normalize_pair_key(0.0, 0.0) == "LL"
    assert normalize_pair_key(0.5, 0.5) == "DD"
    assert normalize_pair_key(1.0, 0.5) == "WD"
    assert normalize_pair_key(0.0, 0.5) == "LD"
    assert normalize_pair_key(1.0, 0.0) == "WL"
    e, _ = elo_estimate(count_pairs(strong))
    assert e > 0.0, f"expected positive elo, got {e}"
    print("sprt self-test: all assertions passed")


def main():
    parser = argparse.ArgumentParser(description="Mini fishtest-style GSPRT (pentanomial)")
    parser.add_argument("pgn", nargs="?", default=None, help="PGN file with game results")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=5.0)
    parser.add_argument("--draw-elo", type=float, default=200.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--no-alternate", action="store_true",
                        help="do not assume alternating colors per game pair")
    parser.add_argument("--json", default=None, help="write JSON report to path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.pgn:
        print("error: provide a PGN file or --self-test", file=sys.stderr)
        sys.exit(2)

    report = analyze_pgn(
        args.pgn,
        elo0=args.elo0,
        elo1=args.elo1,
        draw_elo=args.draw_elo,
        alpha=args.alpha,
        beta=args.beta,
        alternate_colors=not args.no_alternate,
    )
    print(f"games={report['games']} pairs={report['pairs']} counts={report['counts']}")
    print(f"LLR={report['llr']:.2f} bounds=[{report['lower']:.2f},{report['upper']:.2f}] verdict={report['verdict']}")
    print(f"Elo={report['elo']:+.1f}+-{report['elo_err']:.1f} (draw_elo={report['draw_elo']})")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote JSON report to {args.json}")


if __name__ == "__main__":
    main()
