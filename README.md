# 🐋 Whale

**A sample chess engine project to help you build a more complete chess engine with NNUE.**

Whale is a UCI chess engine written in Rust. It is intended as a readable, hackable starting point: board representation, move generation, alpha-beta search, NNUE evaluation, multi-threading, Syzygy tablebases and an opening book, all in one place so you can learn how the pieces fit together and extend them toward a stronger, more complete engine.

> **Origin:** Whale started as a fork of [znxftw/rudim](https://github.com/znxftw/rudim). The core engine (board, search, evaluation, NNUE plumbing) has since been rebuilt. See [Acknowledgements](#acknowledgements) for the projects it learns from.

---

## Feature tour

### Board & move generation (`src/board/`, `src/bitboard/`)

- 64-bit bitboards with precomputed attack tables and magic bitboards for sliding pieces (regenerate with `--generate-magics`).
- Phased move generation with a staged move picker: hash move → good captures → killers → counter moves → quiets → bad captures.
- Exact boolean SEE (`see_ge`) with an early-exit shortcut for clearly winning exchanges.
- Incremental Zobrist hashing, repetition detection, standard castling.

### Search (`src/search/`)

- Iterative deepening with aspiration windows and Principal Variation Search (NegaScout).
- Standard pruning and reductions: mate-distance pruning, razoring, reverse futility pruning, null-move pruning with zugzwang verification, ProbCut, internal iterative reduction, late-move pruning, futility pruning, SEE pruning, singular extensions, and a history-driven LMR table.
- Quiescence search over captures, promotions and selected checking moves.
- Two worker models: Lazy SMP (default) and opt-in split-at-root; search threads run on 16 MB stacks (`SEARCH_THREAD_STACK_SIZE`).
- Transposition table with generational replacement, mate-score adjustment and 50-move-rule downgrade.
- History families: killer moves, history, counter moves, capture history, continuation histories and correction histories.
- Experimental, individually toggleable heuristics (threat-conditioned extensions, bandit ordering, annealing, personas, …) so you can A/B each idea with SPRT.
- Game-state modules (`src/world/`, `src/risk/`, `src/opportunity/`, …) that shape pruning, extensions and time usage, plus `MultiPV` diagnostics.

### Evaluation (`src/eval/`)

- Embedded small NNUE network with incremental (dual-accumulator) updates and an eval cache.
- Optional large Stockfish-style network (`models/whale_*.nnue`) with threat/pair features, auto-loaded when present.
- Depth-conditioned blending, material-aware optimism (`src/eval/optimism.rs`), contempt and draw-score controls, WDL reporting.

### Endgames & openings

- Syzygy tablebase probing (WDL bounds) via `shakmaty-syzygy`; ship `tables/K*vK` for smoke tests.
- EPD opening book probed at the root (`data/book.epd`, `resources/openings.epd`).

---

## Getting started

### Prerequisites

- Recent stable Rust (1.88+) with Cargo. Install via [rustup](https://rustup.rs/).

### Build

```bash
cargo build --release
```

The binary lands at `target/release/whale` (`whale.exe` on Windows).

Optional features:

```bash
cargo build --release --features train   # self-play datagen + Bullet trainer plumbing
cargo build --release --features cuda    # train with CUDA
```

> The build downloads the default small NNUE weights once (`build.rs`). Keep network access enabled for the first build.

### Run

```bash
cargo run --release                 # UCI loop (talks to cutechess, Arena, …)
cargo run --release -- bench        # NPS benchmark
cargo run --release -- --profile    # CPU profiling run
cargo run --release -- --generate-magics
```

Datagen (needs `--features train`):

```bash
cargo run --release --features train -- datagen <out.binpack> <games> <book.fen> [depth] [threads]
cargo run --release --features train -- datagen-teacher <out.binpack> <games> <book.fen> [depth] [threads] <stockfish>
```

---

## UCI options

| Option | Type | Default | Description |
| :----- | :--: | :-----: | :---------- |
| `Hash` | spin | 16 | Transposition table size in MB (1–2048). |
| `Threads` | spin | 1 | Search threads (1–256). |
| `Move Overhead` | spin | 10 | Clock-latency buffer in ms. |
| `SyzygyPath` | string | `<empty>` | Directory with `.rtbw`/`.rtbz` files. |
| `EvalFile` | string | `<embedded>` | Custom NNUE weights file. |
| `EvalFileSmall` | string | `<empty>` | Small net (deprecated, single-net engine). |
| `Contempt` | spin | 0 | Draw aversion in cp (-200–200). |
| `DrawScore` | spin | 0 | Absolute draw value override in cp. |
| `ShowWDL` | check | true | Emit `wdl` on info lines. |
| `MultiPV` | spin | 1 | Ranked root lines (1–8). |
| `UseBook` / `BookFile` / `BookDepth` | check/string/spin | true/`<empty>`/30 | EPD opening book control. |
| `DualNet` | check | true | Gate the large network on cheap-net bounds. |
| `Clear Hash` | button | - | Clear the transposition table. |
| `CFSS_Enabled` | check | true | Coarse-to-fine selective search. |
| `RAS_Enabled` | check | true | Runtime annealing for LMR. |
| `BMO_Enabled` | check | true | Bandit move ordering. |
| `TCE_Enabled` | check | true | Threat-conditioned extensions. |
| `LQT_Enabled` | check | true | Tactical quiescence control. |
| `SPS_Enabled` | check | true | Speculative persona search threads. |
| `DAD_Enabled` | check | true | Disagreement-based time factor. |
| `Extension_Cap_Enabled` | check | true | Cap consecutive extensions. |
| `Razor_Enabled` / `Razor_Margin` | check/spin | true/350 | Frontier razoring. |
| `IIR_Enabled` | check | true | Internal iterative reduction. |
| `NMP_Verify` / `NMP_Verify_Margin` | check/spin | true/150 | Null-move zugzwang verification. |
| `Conspiracy_Enabled` / `Conspiracy_Tolerance` | check/spin | false/30 | Root conspiracy verification (experimental). |
| `SplitRoot_Enabled` | check | false | Split-at-root instead of Lazy SMP. |
| `CPI_Enabled` / `State_Enabled` / `Risk_Enabled` | check | true | Counterplay, state and risk shaping. |
| `Pressure_Enabled` / `Attack_Enabled` / `Conversion_Enabled` | check | true | Pressure, attack and conversion modules. |
| `MustTryGain` / `MustTryRisk` | spin | 30/120 | Must-try tactical gate thresholds. |
| `ConvertScore` / `CrushScore` / `DefendCPI` | spin | 250/600/120 | Conversion/defense thresholds. |

---

## Testing & quality

Full gate (format, lints, unit tests, integration suites, coverage):

```bash
make quality
```

Targeted runs:

```bash
cargo test --lib
cargo test --release --test aprm
cargo test --release --test aprm_suite
cargo test --release --test epd_suite
cargo test --release -- bench
```

> Test threads run search, so the quality targets set `RUST_MIN_STACK=16777216` (16 MB, same as production search threads). If you run `cargo test` by hand on deep-search tests and hit a stack overflow, export that variable first.

Tuning helpers live in `tools/`: SPSA tuner, Texel tuner, SPRT runner and match scripts. Validate every strength-related change with SPRT before keeping it.

---

## Project layout

```text
src/
  bitboard/   bitboards, magics, attack tables
  board/      state, movegen, make/unmake, SEE, FEN, history
  search/     negamax, quiescence, iterative deepening, LMR, TT use, heuristics
  eval/       NNUE nets, loaders, optimism, move ordering histories
  uci/        UCI loop, go/position/setoption, time management
  syzygy.rs   tablebase probing          opening/  EPD book
  world/ risk/ opportunity/ opponent/
  perception/ endgame/ root/            game-state shaping + diagnostics
  datagen.rs teacher.rs train.rs        self-play data + training (feature `train`)
tests/        perft, search, APRM suites, EPD benchmarks, eval equivalence
tools/        SPSA/Texel/SPRT/match scripts
models/       large NNUE weights        tables/  Syzygy files
```

---

## Acknowledgements

- [znxftw/rudim](https://github.com/znxftw/rudim): the original project Whale forked from.
- [Stockfish](https://github.com/official-stockfish/Stockfish): search and evaluation ideas, and the benchmark for correctness.
- [Reckless](https://github.com/codedeliveryservice/Reckless): high-performance Rust engine patterns.
- [Lc0](https://github.com/LeelaChessZero/lc0): contempt, draw scoring and WDL reporting ideas.
- [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch): NNUE training pipeline.
- [Bullet](https://github.com/jw1912/bullet): alternative NNUE training backend under research.

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
