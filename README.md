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
* **Fast Zobrist Hashing & History:** Updates hash states quickly during moves, captures, promotions, castling, and en-passant; includes a check for three-fold repetition.
* **Syzygy Tablebase Support:** Endgame tablebase probing using `shakmaty-syzygy` (WDL bounds and optimal move extraction).

### 2. Search Engine & Novel Heuristics

The search engine uses a multi-threaded Principal Variation Search (PVS) with iterative deepening and 16 new search and pruning algorithms:

* **Depth Allocation & Root Search:**

  * **Disagreement-Allocated Depth (DAD):** Adjusts depth and time for tactical positions based on score disagreements.
  * **Speculative Persona Search (SPS):** Helper threads search under different strategies (Aggressive, Tactical, Solid, Standard) on top of the usual depth stagger. Personas only reshape pruning/LMR — never eval optimism — so shared Transposition Table entries stay sound across threads.

* **Neural & Adaptive Pruning:**

  * **Adversarial Learned Pruner (ALP):** Predicts safe pruning chances based on various factors.
  * **GNN Tree Pruner (GTP):** Uses a graph neural network to prune less important branches.
  * **Sibling Cutoff Rate Pruning (SCR):** Prunes quiet moves based on real-time ratios.
  * **Positional Momentum Tracking (PMT):** Adjusts evaluations based on recent moves.
  * **Reverse Futility Pruning (RFP)** & **ProbCut:** Dynamically adjusts margins.
  * **Null Move Pruning (NMP):** Verifies searches with adaptive reductions.

* **Extensions & Quiescence:**

  * **Threat-Conditioned Extension (TCE):** Predicts extensions for threats.
  * **Learned Quiescence Termination (LQT):** Prevents early termination during tactical shifts.

* **Memory, Ordering & Multi-Resolution Search:**

  * **Persistent Search Memory (PSM):** Shares context across sibling nodes to avoid unproductive lines.
  * **Bandit Move Ordering (BMO):** Adapts move ordering strategies.
  * **Coarse-to-Fine Selective Search (CFSS):** Filters out unpromising moves before detailed expansion.
  * **Runtime Annealing Search (RAS):** Dynamically adjusts search parameters.
  * **Speculative Move Pre-computation (SMP):** Prepares responses to expected opponent moves.
  * **Transposition Table:** Two-tiered design for efficient storage.
  * **Multi-Level History:** Tracks various move histories.

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
| `Contempt`      |  spin  |       0      | Draw aversion in cp (-200 to 200, Lc0-inspired).       |
| `DrawScore`     |  spin  |       0      | Absolute draw value override in cp.                    |
| `ShowWDL`       | check  |     true     | Emit `wdl w d l` on search info lines.                 |
| `CFSS_Enabled`  | check  |     true     | Coarse-to-fine selective search.                       |
| `RAS_Enabled`   | check  |     true     | Runtime annealing LMR perturbation.                    |
| `BMO_Enabled`   | check  |     true     | Bandit move ordering arm selection.                    |
| `TCE_Enabled`   | check  |     true     | Threat-conditioned extensions.                         |
| `LQT_Enabled`   | check  |     true     | Learned quiescence termination.                        |
| `SPS_Enabled`   | check  |     true     | Speculative persona search (helper threads).           |
| `DAD_Enabled`   | check  |     true     | Disagreement-allocated depth time factor.              |
| `Clear Hash`    | button |       -      | Clears all entries in the Transposition Table.         |

---

## Testing

To run the full verification suite (233 unit tests, search integration, movegen, and perft validation):

```bash
cargo test --release
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
