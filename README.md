# 🐋 Whale

Whale is a fast chess engine that works with UCI, built using Rust.

It started as a fork of [znxftw/rudim](https://github.com/znxftw/rudim). We’ve completely rebuilt the core engine, including **Board Representation & Move Generation**, **Search Pipeline**, **Evaluation Architecture**, and **NNUE Subsystems**. We took inspiration from [Stockfish](https://github.com/official-stockfish/Stockfish) for algorithms and used high-performance Rust patterns from [Reckless](https://github.com/codedeliveryservice/Reckless).

We're training evaluation networks with [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch).

**Training Backend Research:** We’re exploring [Bullet](https://github.com/jw1912/bullet) as a possible replacement for `nnue-pytorch` in NNUE model training.

---

## Technical Architecture

### 1. Board Representation & Move Generation

* **High-Performance Bitboards:** A new 64-bit board model that tracks piece placements and colors.
* **Magic Bitboards:** Custom precomputed lookups for sliding piece attacks (bishops, rooks, queens) with a generator for runtime use (`--generate-magics`).
* **Phased Move Generation & Staged Picker:**

  * Gradually generate candidates: Hash/PV Move → Good Captures (SEE ≥ 0) → Killer Moves → Counter Moves → Quiet Moves → Bad Captures.
  * Check for legality directly in the search loop to skip illegal moves.
  * **Exact Boolean SEE:** `see_ge()` returns exactly `see() >= threshold`, short-circuiting as soon as the worst case (the opponent's recapture) already clears the bound, so pruning and ordering gates no longer pay for a full exchange evaluation.
* **Fast Zobrist Hashing & History:** Updates hash states quickly during moves, captures, promotions, castling, and en-passant; includes a check for three-fold repetition.
* **Syzygy Tablebase Support:** Endgame tablebase probing using `shakmaty-syzygy` (WDL bounds and optimal move extraction).

### 2. Search Engine & Novel Heuristics

The search engine uses a multi-threaded Principal Variation Search (PVS) with iterative deepening and 16 new search and pruning algorithms:

* **Depth Allocation & Root Search:**

  * **Disagreement-Allocated Depth (DAD):** Adjusts depth and time for tactical positions based on score disagreements.
  * **Speculative Persona Search (SPS):** Helper threads search under different strategies (Aggressive, Tactical, Solid, Standard) on top of the usual depth stagger. Personas only reshape pruning/LMR — never eval optimism — so shared Transposition Table entries stay sound across threads.

* **Heuristic & Adaptive Pruning (transparent formulas, SPSA-tunable thresholds):**

  * **Adaptive Late-move Pruner (ALP):** Scores safe pruning chances from move index, eval margin, history and momentum.
  * **Graph-guided Tree Pruner (GTP):** Ranks subtree importance by node quality averaged over tree neighbors.
  * **Eval Momentum (Δ static eval over 2 plies):** Feeds RFP/NMP/LMR margins; late-move LMR reduction also keys on sibling-cutoff rate at all-nodes.
  * **Reverse Futility Pruning (RFP)** & **ProbCut:** Dynamically adjusts margins.
  * **Null Move Pruning (NMP):** Verifies searches with adaptive reductions.

* **Extensions & Quiescence:**

  * **Threat-Conditioned Extension (TCE):** Predicts extensions for threats.
  * **Tactical Quiescence Control (LQT):** Prevents early termination during tactical shifts.

* **Memory, Ordering & Multi-Resolution Search:**

  * **Persistent Search Memory (PSM):** Shares context across sibling nodes to avoid unproductive lines.
  * **Bandit Move Ordering (BMO):** Adapts move ordering strategies.
  * **Coarse-to-Fine Selective Search (CFSS):** Filters out unpromising moves before detailed expansion.
  * **Runtime Annealing Search (RAS):** Dynamically adjusts search parameters.
  * **Speculative Move Pre-computation (SMP):** Prepares responses to expected opponent moves.
  * **Per-Node Threat Snapshot:** Checkers, pinners, lesser-piece threat maps and check squares are computed once per node and shared by legality tests, DCN evaluation, quiet-move scoring, TCE and LQT.
* **Transposition Table:** Two-tiered design for efficient storage.
* **Multi-Level History:** Tracks various move histories.

### 2b. Adaptive Pressure Layer (S1–S8)

Behavioral policy on top of search — TT-safe (shapes pruning/ordering/
extension/time only, never leaf eval), one UCI toggle per subsystem:

* **S1 State & Phase Controller** (`position_state.rs`): DEFEND → … → CONVERT classification at the root + `info string aprm state=…`.
* **S2 Counterplay Model** (`counterplay.rs`): node-local CPI from the per-node threat snapshot; widens RFP margin when the side to move has resources.
* **S3 Risk Envelope & Must-Try Gate** (`risk.rs`): gain + urgency + risk + counterplay gate; passed gates buy +25% root time.
* **S4 Concession Tracker** (`concession.rs`): eval-swing detection + Temporary → Permanent ladder.
* **S5 Pressure Planner** (`pressure.rs`): pressure trajectory + ΔCPI/ΔRisk efficiency.
* **S6 Attack & Conversion** (`attack.rs`, `conversion.rs`): urgency → verification budget; tunable convert/crush thresholds.
* **S8 Verification** (`multipv.rs`, `metrics.rs`, `tests/aprm.rs`, `tests/aprm_suite.rs`): UCI `MultiPV` (1–8) root lines with candidate classes + `info string aprm` diagnostics (CPI, freedom, plans, momentum, pressure, concession, musttry, convert, reset, simplify, verified) + 6-category Style Suite + Ultimate 12-step chain. Metric mapping in `docs/adaptive-pressure-metrics.md`.
* **Threshold tuning:** `MustTryGain`, `MustTryRisk`, `ConvertScore`, `CrushScore`, `DefendCPI` are UCI spins wired into `tools/spsa_tuner.py` (spec §24: tune, don't hard-code).

### 3. NNUE Evaluation

* **Primary Network:** Custom dual-accumulator architecture with efficient execution.
* **Depth-Conditioned NNUE (DCN):** Adjusts network representation based on tactical and strategic depth.
* **Robustness & Augmentation:**

  * **Adversarial Robustness Regularization (ARR):** Regularizes against noise in training.
  * **Counterfactual Move Augmentation (CMA):** Evaluates potential threats for long-range awareness.
* **SFNNv16 Dual-Net Architecture:** Supports large Stockfish 19 style setups.
* **Training Pipeline:** Utilizes [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch).

### 4. NNUE Training Backend Research

Whale is looking into new training setups to enhance its evaluation network training.

#### Bullet Training Backend

We’re checking out [Bullet](https://github.com/jw1912/bullet) as a potential alternative to [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch).

The focus is on:

* Evaluating Bullet as a new NNUE training backend.
* Comparing performance and resource use.
* Checking compatibility with Whale's NNUE setups.
* Exploring integration with current training data.
* Assessing replacing the existing `nnue-pytorch` workflow.

> **Status:** Research is ongoing. `nnue-pytorch` is still in use until we finish evaluating Bullet.

---

## Getting Started

### Prerequisites

* [Rust](https://www.rust-lang.org/) (stable version 1.75+)
* Cargo

### Building the Release Binary

```bash
cargo build --release
```

The optimized binary will be at:

* `target/release/whale` (Linux / macOS)
* `target/release/whale.exe` (Windows)

To build with NNUE training and datagen support:

```bash
cargo build --release --features train
```

With CUDA acceleration:

```bash
cargo build --release --features cuda
```

---

## CLI & Engine Modes

### UCI Interactive Mode

```bash
cargo run --release
```

### Engine Benchmark (NPS Measurement)

```bash
cargo run --release -- bench
```

### CPU Profiling

```bash
cargo run --release -- --profile
```

### Self-Play Binpack Datagen

```bash
cargo run --release --features train -- datagen <output.binpack> <games> <book.fen> [depth] [threads]
```

### Teacher-Supervised Datagen

```bash
cargo run --release --features train -- datagen-teacher <output.binpack> <games> <book.fen> [depth] [threads] <stockfish_binary>
```

### Recompute Magic Bitboards

```bash
cargo run --release -- --generate-magics
```

---

## UCI Options

| Option          |  Type  |    Default   | Description                                             |
| :-------------- | :----: | :----------: | :------------------------------------------------------ |
| `Hash`          |  spin  |      16      | Size of the Transposition Table in MB (1 to 2048 MB).  |
| `Threads`       |  spin  |       1      | Number of concurrent search threads (1 to 256).        |
| `Move Overhead` |  spin  |      10      | Latency buffer in milliseconds (0 to 5000 ms).         |
| `SyzygyPath`    | string |   `<empty>`  | Path to directory with `.rtbw` and `.rtbz` files.     |
| `EvalFile`      | string | `<embedded>` | Path to custom NNUE weights file.                       |
| `EvalFileSmall` | string |   `<empty>`  | Path to small NNUE net for dual-net mode.              |
| `Contempt`      |  spin  |       0      | Engine-relative draw aversion in cp (-200 to 200, Lc0-inspired). |
| `DrawScore`     |  spin  |       0      | Absolute draw value override in cp.                    |
| `ShowWDL`       | check  |     true     | Emit `wdl w d l` on search info lines.                 |
| `CFSS_Enabled`  | check  |     true     | Coarse-to-fine selective search.                       |
| `RAS_Enabled`   | check  |     true     | Runtime annealing LMR perturbation.                    |
| `BMO_Enabled`   | check  |     true     | Bandit move ordering arm selection.                    |
| `TCE_Enabled`   | check  |     true     | Threat-conditioned extensions.                         |
| `LQT_Enabled`   | check  |     true     | Tactical quiescence termination.                       |
| `SPS_Enabled`   | check  |     true     | Speculative persona search (helper threads).           |
| `DAD_Enabled`           | check  |     true     | Disagreement-allocated depth time factor.              |
| `Extension_Cap_Enabled` | check  |     true     | Consecutive extension cap (prevents tactical dive).    |
| `MultiPV`               |  spin  |       1      | Ranked root lines 1–8 (spec §28 behavioral verification). |
| `CPI_Enabled`           | check  |     true     | S2 counterplay model (CPI → RFP margin).               |
| `State_Enabled`         | check  |     true     | S1 state controller (root shaping + diagnostics).      |
| `Risk_Enabled`          | check  |     true     | S3 risk envelope + must-try time gate.                 |
| `Pressure_Enabled`      | check  |     true     | S5 pressure trajectory tracking.                       |
| `Attack_Enabled`        | check  |     true     | S6a attack urgency budgets.                            |
| `Conversion_Enabled`    | check  |     true     | S6b conversion signal in diagnostics.                  |
| `MustTryGain`  |  spin  |      30      | Min gain (cp) for must-try gate (10–100, SPSA).        |
| `MustTryRisk`  |  spin  |      120     | Max risk units for must-try gate (40–250, SPSA).       |
| `ConvertScore` |  spin  |      250     | Score (cp) to prefer simplification (100–600, SPSA).   |
| `CrushScore`   |  spin  |      600     | Score (cp) for forced conversion (300–1200, SPSA).     |
| `DefendCPI`    |  spin  |      120     | Opp CPI triggering defend state (40–250, SPSA).        |
| `DualNet`               | check  |     true     | Dual-Net optimistic evaluation gating.                 |
| `Clear Hash`            | button |       -      | Clears all entries in the Transposition Table.         |

---

### 2b. Whale Adaptive Architecture (§61 Domain Modules)

Whale structures its behavioral playing policy into clean, decoupled domain crates:

* **`perception`** ([`src/perception/`](file:///C:/Users/newo/Downloads/whale/src/perception/mod.rs)): Strategic snapshot, pawn islands, passed/isolated/backward pawns, open files, king shelter.
* **`world`** ([`src/world/`](file:///C:/Users/newo/Downloads/whale/src/world/mod.rs)): 7-state behavioral machine (`Defend`, `Stabilize`, `Improve`, `Press`, `Attack`, `Crush`, `Convert`), hysteresis, volatility levels, CPI calculations, and pressure trajectories.
* **`opponent`** ([`src/opponent/`](file:///C:/Users/newo/Downloads/whale/src/opponent/mod.rs)): Opponent modeling, defense/escape capacity, and proactive break detection (`plans.rs`).
* **`opportunity`** ([`src/opportunity/`](file:///C:/Users/newo/Downloads/whale/src/opportunity/mod.rs)): Weakness classification across 4 persistence levels (`Temporary`, `Latent`, `Structural`, `Permanent`).
* **`risk`** ([`src/risk/`](file:///C:/Users/newo/Downloads/whale/src/risk/mod.rs)): Risk envelope and Must-Try tactical gate.
* **`endgame`** ([`src/endgame/`](file:///C:/Users/newo/Downloads/whale/src/endgame/mod.rs)): Simplification incentives and anti-fortress conversion gates.
* **`root`** ([`src/root/`](file:///C:/Users/newo/Downloads/whale/src/root/mod.rs)): Multi-PV candidate classification, root attack commitment verification, and 16-metric search telemetry.

---

## 🚀 Cloud & Kaggle CPU Pipeline

A ready-to-run Jupyter notebook is provided in [`notebooks/whale_kaggle_pipeline.ipynb`](file:///C:/Users/newo/Downloads/whale/notebooks/whale_kaggle_pipeline.ipynb) for running on free Kaggle Linux CPU instances (4 vCPUs):
* Auto-installs Rust toolchain & Linux build tools.
* Builds Whale in Release mode.
* Performs SPSA parameter tuning with matplotlib convergence graphs.
* Runs fastchess SPRT match validation.
* Analyzes 16 Core Behavioral Metrics (MTR, PCR, CRI, RSI, ODI).
* Demos Multi-Head NNUE PyTorch training on CPU.

---

## Testing

To run the full verification suite — 700+ unit tests plus the APRM behavioral, transition, and EPD benchmark suites:

```bash
# Run unit & regression tests
cargo test --release --test aprm

# Run 12-step state transition chain & style categories
cargo test --release --test aprm_suite

# Run 6-category position benchmark suite (Defensive, Quiet, Opportunity, MustTry, Pressure, Conversion)
cargo test --release --test epd_suite
```


---

## Acknowledgements

* [znxftw/rudim](https://github.com/znxftw/rudim): Base repository.
* [Stockfish](https://github.com/official-stockfish/Stockfish): Ideas and benchmarks.
* [Reckless](https://github.com/codedeliveryservice/Reckless): Design references and optimization patterns.
* [Lc0](https://github.com/LeelaChessZero/lc0): Ideas for contempt, draw scoring and WDL reporting.
* [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch): NNUE training pipeline.
* [Bullet](https://github.com/jw1912/bullet): Alternative NNUE training backend under research.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
