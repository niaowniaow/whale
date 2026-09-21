# Adaptive Pressure — Metrics, Suite & Tuning (spec §29, §30)

How the 33-criteria spec is measured in this repo. Two layers; no fake
precision: per-search observables first, suite rates second.

## 1. Per-search observables (`src/search/metrics.rs::SearchMetrics`)

Collected from `SearchState` after a finished root search via
`metrics::collect(&state)`:

| Field | Source |
| :---- | :----- |
| `score`, `state` | root score, `PositionState` (S1) |
| `opp_cpi`, `opp_freedom`, `opp_breaks` | child position after bestmove (S2 + movegen + S5/§21) |
| `pressure`, `sustained_plies` | S5 trajectory |
| `concession_swing` | S4 latch (survives flat later iterations and `go`s) |
| `musttry_fired`, `verified` | S3 gate + B1 verification search |
| `simplify`, `reset` | S6b / S7 flags |
| `multipv_count` | S8 ranked lines |

Live view: `info string aprm ...` every iteration (state, cpi_us/opp,
freedom_us/them, plans_opp, momentum, pressure, sustained, vbudget,
concession, musttry, convert, reset, simplify, verified/refuted).

## 2. The 16 spec metrics — measured vs derived

| Metric | Status | How |
| :----- | :----- | :-- |
| MTR (Must-Try Recognition) | ✅ measured | `SuiteSummary::mtr` on must-try-labeled suite |
| PCR (Premature Attack) | ✅ measured | attack/crush state on quiet-labeled suite (lower better) |
| CRI (Conversion Reliability) | ✅ measured | simplify on convert-labeled suite |
| RSI (Recovery/Reset) | ✅ measured | reset on defensive-labeled suite |
| ODI proxy | ✅ measured | concession detected on opportunity-labeled suite |
| CDI (Counterplay Denial) | 🟡 derived | `opp_cpi` + freedom/breaks deltas per search; full CDI needs game histories |
| PPI/PEI (Pressure) | 🟡 derived | `pressure`, `sustained_plies`, ΔCPI/ΔRisk from observables |
| OMR (Opportunity Miss) | 🟡 derived | 1 − ODI proxy on opportunity-labeled suite |
| FRI/PCI (Freedom/Plan Constraint) | 🟡 derived | `opp_freedom`, `opp_breaks` trajectories |
| DSI (Defensive Stability) | ✅ harness/suite | defensive-labeled pass rate + eval-collapse check (`tools/aprm_game_analyzer.py`) |
| AUI/IOI/MAI/RAE | ✅ harness | multi-ply game log parser via `tools/aprm_game_analyzer.py` and `tools/run_sprt.py` |

## 3. Style Suite (spec §29 — `tests/aprm_suite.rs`)

Six entries mirroring the spec weights (fast: depths 1–3):

* Defensive — side to move in check → must classify `defend`.
* Quiet — startpos → must not enter `attack`/`crush`.
* Opportunity — Fool's mate blundered across two `go`s on one state →
  concession latch must fire (baselines survive searches via
  `best_previous_score`).
* Must-Try — mate-in-1 single iteration → gate + verification must fire.
* Pressure — startpos depth 3 → pressure must sustain (`sustained ≥ 1`).
* Conversion — KQ vs K → `simplify` must engage.

Plus the Ultimate 12-step chain (§31): one deterministic walk
improve → concession → amplify → pressure → urgency → gate →
verification budget → conversion value → reset → convert.

Run: `cargo test --test aprm_suite`.

## 4. Tuning the thresholds (spec §24 — tune, don't hard-code)

SPSA-tunable UCI options (`tools/spsa_tuner.py PARAMETERS`):

`MustTryGain` (30), `MustTryRisk` (120), `ConvertScore` (250),
`CrushScore` (600), `DefendCPI` (120) — plus the 9 classic search params.

Procedure per new wiring (mandatory, same as every heuristic here):

1. Ship behind its toggle (default = current behavior).
2. `tests/aprm*.rs` behavioral test first (no silent regression).
3. SPRT vs baseline (`tournament.yml` / fastchess) before changing defaults.
4. SPSA only after SPRT passes — thresholds are starting points, not dogma.

Remaining hard-coded internals (next SPSA round): `StateThresholds`
attack/convert scores, `RiskEnvelope` normal/elevated bands, urgency
window table, pressure weights (½, −1, −2, +3), verification tolerance
(30 cp), reset drop (50 cp). Each is a pure function of its inputs, so
exposing any of them as UCI is mechanical.
