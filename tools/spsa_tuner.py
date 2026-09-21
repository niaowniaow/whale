
import argparse
import math
import os
import random
import subprocess
import sys
import time

PARAMETERS = {
    "RFP_Margin": {"default": 110, "min": 50, "max": 300, "step": 10},
    "Futility_Margin": {"default": 120, "min": 50, "max": 300, "step": 10},
    "Singular_Margin": {"default": 2, "min": 1, "max": 5, "step": 1},
    "ProbCut_Margin": {"default": 170, "min": 50, "max": 400, "step": 15},
    "NMP_Base": {"default": 3, "min": 1, "max": 6, "step": 1},
    "NMP_Depth_Div": {"default": 3, "min": 2, "max": 8, "step": 1},
    "LMR_Base": {"default": 65, "min": 10, "max": 150, "step": 5},
    "LMR_Div": {"default": 215, "min": 100, "max": 350, "step": 10},
    "History_Weight": {"default": 2, "min": 1, "max": 4, "step": 1},

    "MustTryGain": {"default": 30, "min": 10, "max": 100, "step": 5},
    "MustTryRisk": {"default": 120, "min": 40, "max": 250, "step": 10},
    "ConvertScore": {"default": 250, "min": 100, "max": 600, "step": 25},
    "CrushScore": {"default": 600, "min": 300, "max": 1200, "step": 50},
    "DefendCPI": {"default": 120, "min": 40, "max": 250, "step": 10},
}

class SpsaTuner:
    def __init__(self, engine_path, games_per_eval=20, alpha=0.602, gamma=0.101, a=10.0, c=5.0):
        self.engine_path = engine_path
        self.games_per_eval = games_per_eval
        self.alpha = alpha
        self.gamma = gamma
        self.a = a
        self.c = c
        self.theta = {k: float(v["default"]) for k, v in PARAMETERS.items()}

    def perturb(self, k):
        ck = self.c / ((k + 1) ** self.gamma)
        delta = {}
        theta_plus = {}
        theta_minus = {}
        for param, cfg in PARAMETERS.items():
            d = 1.0 if random.random() > 0.5 else -1.0
            delta[param] = d
            p_val = self.theta[param]
            tp = round(p_val + ck * cfg["step"] * d)
            tm = round(p_val - ck * cfg["step"] * d)
            theta_plus[param] = max(cfg["min"], min(cfg["max"], tp))
            theta_minus[param] = max(cfg["min"], min(cfg["max"], tm))
        return delta, theta_plus, theta_minus, ck

    def format_options(self, params_dict):
        opts = []
        for k, v in params_dict.items():
            opts.append(f"option.{k}={int(v)}")
        return opts

    def run_fast_chess_match(self, theta_plus, theta_minus, book_path=None):
        cmd = [
            "fast-chess",
            "-engine", f"cmd={self.engine_path}", "name=EnginePlus", *self.format_options(theta_plus),
            "-engine", f"cmd={self.engine_path}", "name=EngineMinus", *self.format_options(theta_minus),
            "-each", "tc=5+0.05",
            "-rounds", str(self.games_per_eval // 2),
            "-repeat",
            "-concurrency", "4",
        ]
        if book_path and os.path.exists(book_path):
            cmd.extend(["-openings", f"file={book_path}", "format=epd", "order=random"])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = res.stdout
            plus_score = 0.0
            for line in output.splitlines():
                if "Score of EnginePlus vs EngineMinus:" in line:
                    parts = line.split(":")[-1].split("-")
                    w = float(parts[0].strip())
                    l = float(parts[1].strip())
                    d = float(parts[2].split()[0].strip())
                    plus_score = (w + 0.5 * d) / max(1.0, (w + l + d))
                    break
            return plus_score
        except Exception:
            return 0.5

    def step(self, k, book_path=None):
        ak = self.a / ((k + 1 + 10) ** self.alpha)
        delta, theta_plus, theta_minus, ck = self.perturb(k)
        score_plus = self.run_fast_chess_match(theta_plus, theta_minus, book_path)
        score_minus = 1.0 - score_plus
        diff = score_plus - score_minus

        for param, cfg in PARAMETERS.items():
            ghat = diff / (2.0 * ck * delta[param])
            self.theta[param] += ak * ghat * cfg["step"]
            self.theta[param] = max(cfg["min"], min(cfg["max"], round(self.theta[param])))

        print(f"Iteration {k+1:03d} | Diff: {diff:+.3f} | Current Params: {self.theta}")

def main():
    parser = argparse.ArgumentParser(description="Turnkey SPSA Tuner for Whale")
    parser.add_argument("--engine", default="target/release/whale.exe", help="Path to whale binary")
    parser.add_argument("--book", default="resources/openings.epd", help="Path to openings book")
    parser.add_argument("--iterations", type=int, default=100, help="Number of SPSA iterations")
    parser.add_argument("--games", type=int, default=20, help="Games per iteration")
    args = parser.parse_args()

    tuner = SpsaTuner(engine_path=args.engine, games_per_eval=args.games)
    print("Starting SPSA tuning session...")
    for it in range(args.iterations):
        tuner.step(it, book_path=args.book)
    print("\nTuning complete. Final parameters:")
    for k, v in tuner.theta.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
