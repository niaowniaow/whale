import argparse
import os
import sys
import urllib.request
import zipfile

BENCHMARKS = {
    "data/benchmarks/defense.epd": [
        '4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1 c0 "Defensive in check";',
        '8/8/8/8/8/4k3/4r3/4K3 w - - 0 1 c0 "King trapped in check";',
        '4r1k1/5ppp/8/8/8/8/4PPPP/4KB1R b - - 0 1 c0 "Under pressure back rank";',
    ],
    "data/benchmarks/quiet.epd": [
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 c0 "Startpos quiet";',
        'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5 c0 "Giuoco Pianissimo quiet";',
        'rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5 c0 "QGD quiet equality";',
    ],
    "data/benchmarks/conversion.epd": [
        '8/8/8/8/8/5K2/5Q2/6k1 w - - 0 1 c0 "KQ vs K conversion";',
        '8/8/8/4k3/8/8/4K3/4R3 w - - 0 1 c0 "KR vs K conversion";',
        '8/8/8/4k3/8/5B2/4K3/4R3 w - - 0 1 c0 "R+B vs K winning conversion";',
    ],
    "data/benchmarks/must_try.epd": [
        'r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4 c0 "Scholar attack must-try";',
        'r2qkb1r/pp2pppp/2n1b3/2pn4/2BP4/5N2/PPP2PPP/RNBQK2R w KQkq - 0 7 c0 "Tactical fork must-try";',
    ],
    "data/benchmarks/opportunity.epd": [
        'rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3 c0 "Fools mate blunder";',
        'r1bqkb1r/pppp1ppp/2n5/4p3/4n3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4 c0 "Hanging pawn opportunity";',
    ],
    "data/benchmarks/pressure.epd": [
        'r1bqk2r/pppp1ppp/2n5/4p3/1bB1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 5 c0 "Pressure center space";',
        'rnbq1rk1/pp2ppbp/3p1np1/8/2PNP3/2N1BP2/PP4PP/R2QKB1R b KQ - 0 8 c0 "Dragon setup pressure";',
    ],
}

OPENINGS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pp1ppppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq c3 0 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]

UHO_URL = "https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip"


def write_fixtures():
    for path, lines in BENCHMARKS.items():
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {path} ({len(lines)} positions)")
    os.makedirs("resources", exist_ok=True)
    with open("resources/openings.epd", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(OPENINGS) + "\n")
    print(f"wrote resources/openings.epd ({len(OPENINGS)} positions)")


def fetch_uho(dest="data/books/UHO_Lichess_4852_v1.epd"):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"exists, skip: {dest}")
        return
    tmp = dest + ".zip"
    print(f"downloading UHO book (~50MB) ...")
    req = urllib.request.Request(UHO_URL, headers={"User-Agent": "whale-dl"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    with zipfile.ZipFile(tmp) as z:
        names = [n for n in z.namelist() if n.endswith(".epd")]
        z.extract(names[0], os.path.dirname(dest))
        os.replace(os.path.join(os.path.dirname(dest), names[0]), dest)
    os.remove(tmp)
    print(f"done: {dest}")


def main():
    parser = argparse.ArgumentParser(description="Recreate EPD fixtures / fetch UHO book")
    parser.add_argument("--uho", action="store_true", help="also download full UHO book")
    parser.add_argument("--uho-dest", default="data/books/UHO_Lichess_4852_v1.epd")
    args = parser.parse_args()
    write_fixtures()
    if args.uho:
        fetch_uho(args.uho_dest)


if __name__ == "__main__":
    main()
