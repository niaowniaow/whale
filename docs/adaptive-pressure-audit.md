# Adaptive Pressure Engine — Audit đối chiếu 33 tiêu chí

**Đối tượng:** Whale (Rust, fork `znxftw/rudim`) @ commit `21fe1ef`
**Câu hỏi:** *"Engine tôi có đáp ứng tiêu chí Adaptive Pressure không?"*
**Trả lời ngắn gọn: KHÔNG — chưa.** Whale hiện là một **alpha-beta search engine tốt** (NNUE
Chess768 + SFNNv16, Lazy SMP, ~15 heuristic pruning/extension), nhưng **chưa có behavioral
layer** mà spec mô tả. Ước lượng độ phủ:

| Mức | Số tiêu chí | Ghi chú |
| :-- | :---------: | :------ |
| ✅ Có, đúng như spec | **0 / 33** | không tiêu chí nào được implement như một cơ chế có chủ đích |
| 🟡 Có "nguyên liệu" dùng lại được | **8 / 33** | #2, #7, #13, #22, #24, #25, #26, #27 — momentum, optimism, NodeThreats, TCE/LQT, DAD, Syzygy, personas |
| ❌ Chưa có gì | **25 / 33** | toàn bộ state machine, risk envelope, counterplay/CPI, concession, must-try, conversion, metrics |

Bằng chứng `grep` trong `src/` (loại trừ vendor `lc0/`, `Reckless/`, `Stockfish/`):

```text
counterplay 0 | freedom 0 | urgency 0 | risk 0 | concession 0
must_try 0   | envelope 0 | plan 0     | multipv 0 | PositionState 0
state_machine 0 | king_safety 0 | simplif 0 | mobility 6 (chỉ trong lqt.rs)
pressure 1 (chỉ là comment trong node_threats.rs) | king_danger 9 (chỉ trong tce.rs)
```

→ Các khái niệm cốt lõi của spec **hoàn toàn không tồn tại trong code**, không phải "có nhưng yếu".

---

## 1. Nền tảng hiện có (nguyên liệu tái sử dụng được)

Đây là phần *đáng giá* của engine: hạ tầng search đủ tốt để gắn behavioral layer lên trên mà
không phải viết lại.

| Hạng mục | Vị trí | Có thể dùng cho |
| :------- | :----- | :-------------- |
| Threat snapshot / node (checkers, pinned 2 phía, threat map lesser-piece, check squares) | `src/board/node_threats.rs` | **CPI / Counterplay Denial** (tiêu chí 3, 13, 27) — đây là nguyên liệu đắt giá nhất, hiện chỉ dùng cho legality/eval/order/TCE |
| `momentum` = Δstatic_eval 2 ply | `negamax.rs:260-267` | Pressure accumulation (7) — hiện chỉ feed RFP/NMP/LMR |
| `optimism` (self-referential, đổi dấu theo score) | `iterative_deepening.rs:104-112` | Winning-vs-losing pressure (25) |
| `is_improving` (eval_stack) | `negamax.rs:351-357` | State classification (17) |
| DCN (depth regime + queen-threat + pawn count + bishop pair + FiLM gamma/beta) | `src/eval/nnue/dcn.rs` | Asymmetric tactical/strategic signal (12, 18) |
| TCE (pins, threatened pieces, king_danger, check count → extension 0/1/2) | `src/search/tce.rs` | Tactical trigger (13) |
| LQT (pins, discovered attacks, king distance, window) | `src/search/lqt.rs` | Tactical trigger (13) |
| Extension cap (không extend 3 ply liên tiếp) | `negamax.rs:785-809` | Risk control (4) — chống "tactical dive" |
| DAD + time management (falling_eval, best_move_instability, node effort, stability_depth) | `iterative_deepening.rs:211-279` | Search priority (27), conversion timing (24) |
| Contempt / DrawScore / WDL | `src/search/draw.rs` | Conversion (24) — nhưng có lỗi, xem §3 |
| Syzygy WDL + DTZ-optimal move | `src/syzygy.rs` | Conversion (14, 24) |
| SPS personas + UCI toggle cho **mọi** heuristic | `src/search/sps.rs`, `setoption.rs` | A/B tuning 8 subsystem mới (29) |
| `spsa_tuner.py`, `play_match.py`, `tournament.yml` (fastchess) | `tools/`, `.github/` | Benchmark infra (29) |
| Pattern test equivalence (`tce_equiv.rs`) | `tests/` | Verification harness cho module mới (29, 30) |

## 2. Bảng đối chiếu 33 tiêu chí

Ký hiệu: ❌ không tồn tại · 🟡 có nguyên liệu nhưng **sai mục đích / sai scope** · ✅ đúng spec.

| # | Tiêu chí | Verdict | Bằng chứng / lý do |
| :-: | :------- | :-----: | :----------------- |
| 1 | Core philosophy (adaptive, không cố định style) | ❌ | Behavior được điều khiển bởi *search statistics* (depth, move count, score disagreement), không bởi *position state*. |
| 2 | Position survival integrity + **Defensive Regret** | 🟡 | Không có DR. Có "an toàn" ở tầng search: pruning conservative, extension cap (`negamax.rs:800`), NMP guards, SEE ≥ -74 trong qsearch. Đây là chống blunder, không phải metric DR. |
| 3 | Counterplay Denial + **CPI** | ❌ | Không có metric counterplay. `NodeThreats.threats_them` / `check_squares` **đã được tính** nhưng chưa bao giờ dùng để đánh giá tài nguyên của đối thủ. |
| 4 | Own Risk Control (risk model) | ❌ | Không có mô hình risk. RFP/NMP/futility/LMP/ProbCut chỉ tối ưu node/NPS. |
| 5 | Opportunity Sensing (concession) | ❌ | Không phát hiện concession. Gần nhất: `structural_disagreement` = \|TT − static eval\| (`negamax.rs:269-276`) chỉ dùng để giảm LMR 1 nấc. |
| 6 | Weakness Persistence (temp/structural/permanent/latent) | ❌ | NNUE là **Chess768 piece-square thuần** (`eval/nnue/features.rs`) — không có pawn structure / mobility / king safety term nào để phân loại weakness. |
| 7 | Pressure Accumulation | 🟡 | `momentum` (Δeval 2 ply) + `eval_stack` tồn tại, nhưng chỉ là đầu vào pruning (RFP `negamax.rs:366`, NMP `nmp.rs:27,69`, LMR `lmr.rs:126-134`). Không có trajectory, không có persistence. |
| 8 | Pressure Efficiency (ΔCPI / ΔRisk) | ❌ | Không có 2 đại lượng để chia. |
| 9 | Initiative Ownership | ❌ | Không đo "ai phải phản ứng". |
| 10 | Attack Urgency | ❌ | Không có urgency. Có `DAD` tăng 25% time khi \|Δscore\| > 60 — theo *search*, không theo *window sắp đóng*. |
| 11 | Must-Try within Risk Range | ❌ | Không có `Must-Try`, không có `Risk Envelope`. Root không phân loại candidate. |
| 12 | Controlled Aggression | ❌ | "Aggressive persona" (`sps.rs:57`) chỉ đổi `futility_margin_mult` 140 + `lmr_base` 0.60 — không phải đánh giá attack gain vs damage. |
| 13 | Tactical Trigger (đổi mode tính toán) | 🟡 | Có extension theo tín hiệu (TCE, check, singular double, recapture) + LQT kéo dài qsearch + `in_check`/`gives_check` guards. **Thiếu**: tăng depth theo tín hiệu, mở rộng forcing search (qsearch **không sinh check**, chỉ capture + promoted-queen quiet — `quiescence.rs:160-307`). |
| 14 | Attack Conversion (kiểm tra attack tạo ra gì) | ❌ | Không có bookkeeping gain (king displacement, weakness, material). Chỉ có score TT. |
| 15 | Pressure → Attack → Pressure loop | ❌ | Không có loop; không có trạng thái để quay về. |
| 16 | Recovery / Reset | ❌ | Không có logic rút lui. Chỉ có: aspiration window, time management, `is_mate` → dừng. |
| 17 | State Adaptability (DEFEND→…→CONVERT) | ❌ | `PositionState` / state machine = 0 hit. `game_phase` là legacy phase counter. Persona được chọn theo **`thread_id % 4`** (`sps.rs:14`), không theo position. |
| 18 | Asymmetric Mistake Response | ❌ | Search đối xứng hoàn toàn. `Contempt` không dùng được cho exploitation vì không side-aware (§3.1). |
| 19 | Concession Amplification | ❌ | Không có. |
| 20 | Opponent Freedom Reduction | ❌ | Không đếm legal moves / breaks của đối thủ. NNUE không có mobility/space feature ⇒ eval không thể "ép" một cách tường minh. |
| 21 | Plan Restriction | ❌ | Không có khái niệm plan. CFSS (`cfss.rs:26`) chỉ prune *move của mình* theo coarse pass. |
| 22 | Patience / Premature Attack Rate | 🟡 | Không có "premature attack" vì **không có tầng attack**; pruning conservative gián tiếp tạo patience. Đây là "patience do không làm gì", không phải policy. |
| 23 | FOMO Control (passive ↔ reckless) | ❌ | Không có núm điều chỉnh giữa hai phía. |
| 24 | Conversion Integrity (ngưỡng + không convert sớm) | 🟡 | Có: Syzygy WDL/DTZ, dừng khi mate, time giảm khi eval stable. **Thiếu**: ngưỡng chuyển phase, simplification logic, counterplay removal. Spec yêu cầu ngưỡng **học/tune**, còn engine hard-code, và các bảng "learned" không hề được train (§3.3). |
| 25 | Winning vs Losing Pressure | 🟡 | `optimism` đổi dấu theo score ⇒ khi thắng thì optimism dương (tự tin hơn), khi thua thì âm. Nhưng nó **đối xứng** và ảnh hưởng eval/pruning, không phải risk budget cho must-try. |
| 26 | Evaluation Stability | 🟡 | Aspiration window giảm dao động, `eval_stack` giữ lịch sử eval. Không có cơ chế đảm bảo "advantage có chất lượng". |
| 27 | Search Priority | 🟡 | Có nhiều cơ chế đổi *effort* (DAD time factor, CFSS, TCE, LQT, SPS depth stagger). Tất cả key theo depth/disagreement/move count — **không có** "must-try ↑↑↑", "counterplay danger ↑", "quiet position → normal". |
| 28 | Multi-PV Behavioral Verification | ❌ | `multipv` = **0 hit trong `src/`**. Không thể xem top-N candidates & lý do. Có `smp_precompute.rs` (precompute replies) nhưng score bị vứt (`let _ = score;`), chỉ warm TT. |
| 29 | Style Benchmark Suite (20/20/20/15/15/10) | ❌ | Chỉ có 675 unit tests + perft/search/equivalence + fastchess Elo vs stable release. Không có suite hành vi, không có EPD theo nhóm tiêu chí. |
| 30 | 16 core metrics (DSI, CDI, ODI, AUI, MTR, RAE, PPI, PEI, IOI, MAI, PCR, OMR, FRI, PCI, CRI, RSI) | ❌ | **0/16** tồn tại dưới bất kỳ dạng nào (kể cả diagnostic). |
| 31 | Ultimate Behavior Test (12 bước full cycle) | ❌ | Không thể có khi thiếu state machine + métrics. |
| 32 | Personality chains (CALM→OBSERVE→…, DEFEND→ABSORB→…) | ❌ | Chuỗi persona chỉ khác pruning margin theo thread. |
| 33 | Định nghĩa chính thức | ❌ | Chỉ thoả phần "minimizes unnecessary positional risk" (nhờ pruning an toàn) và "converts efficiently" (nhờ Syzygy), phần còn lại chưa có. |

---

## 3. Phát hiện trong quá trình audit (cần sửa trước khi xây layer mới)

### 3.1. `Contempt` bị mất tính bất đối xứng (bug thật) — **ĐÃ SỬA**

**Trước (bug):** `apply_contempt` tính `side_sign` rồi vứt (`let _ = side_sign;`) và trừ một
lượng bị chặn ngầm ở `|c| <= 50` tại **mọi** node hoà:

```rust
let side_sign: i16 = if stm_is_white { 1 } else { -1 };
let _ = side_sign;                       // <-- bị vứt
score.saturating_add(d).saturating_sub(c.signum() * c.abs().min(50))
```

⇒ Chỉ còn là "postpone-draw gradient" kiểu `value_draw` của Stockfish: dịch draw cùng một
hướng cho cả hai bên, không thể hiện "engine muốn tránh hoà"; và `Contempt 200` hành xử y hệt
`Contempt 50` dù UCI quảng cáo dải ±200.

**Sau (fix):** contempt được tính **engine-relative**, zero-sum giữa hai bên:

```rust
let contempt = if stm == engine_side { c } else { -c };
score.saturating_add(d).saturating_sub(contempt)
```

* `SearchState` có field mới `engine_side: Side` (`search_state.rs`), được set ở
  `iterative_deepening::search()` từ **root side to move** (bên engine cầm quân) và được
  `clone_for_worker` copy nguyên vẹn ⇒ helper thread chấm draw cùng phía với primary, không
  làm lệch score trong TT chia sẻ (giữ đúng TT-SAFETY INVARIANT ở §3.2).
* Cả 4 call site đã cập nhật: `negamax.rs` (draw node + stalemate) và `quiescence.rs` (2 chỗ).
* Toàn bộ dải ±200 giờ có hiệu lực; giá trị ngoài dải bị clamp, không wrap.
* Helper cũ `stm_is_white` đã bị xoá (không còn nơi dùng).

Test bảo vệ hành vi (`cargo test --lib`):

```text
search::draw::tests::contempt_follows_the_engine_side ............... ok
search::draw::tests::contempt_uses_the_full_option_range ............ ok
search::draw::tests::contempt_shifts_draws .......................... ok
search::iterative_deepening::tests::contempt_is_engine_relative_at_the_root ... ok
search::iterative_deepening::tests::search_records_engine_side_from_root_stm .. ok
```

`contempt_is_engine_relative_at_the_root` là test end-to-end qua `search()`: trên thế cờ K vs K
(hoà chết) với `Contempt 50`, root score **bắt buộc âm** (~−50) dù engine cầm Trắng hay Đen —
đúng ngữ nghĩa "draw aversion hướng về engine" mà spec §18/§24 cần.


### 3.2. Persona (SPS) hiện **gần như vô hiệu** với cấu hình mặc định

* `persona_for_thread(thread_id % 4)` (`sps.rs:14`) — persona gắn với **thread**, không với position.
* Chỉ **helper thread** nhận persona (`iterative_deepening.rs:392-405`); `search_primary` luôn
  chạy `Standard` ⇒ nước đi cuối cùng (bestmove) đến từ search Standard.
* `Threads=1` (mặc định) ⇒ persona không chạy lần nào.
* Persona chỉ đổi margin/LMR, **không đổi eval** — đây là **TT-SAFETY INVARIANT** có chủ đích
  (`sps.rs:23-35`): mọi thứ ảnh hưởng leaf eval phải uniform giữa các thread, nếu không TT
  entries bị nhiễm offset. **Ràng buộc này áp dụng cho mọi "personality/state" mới**: hoặc chỉ
  được đổi pruning/extension/time, hoặc phải đưa state vào TT key.

### 3.3. Các bảng "learned" thực chất là **hằng số hard-code, chưa từng train**

`ALP`, `GTP`, `LQT`, `PSM`, `TCE`, `DCN` đều mô tả mình là "neural/mô hình học" nhưng weights
là hằng số viết tay trong file. Ví dụ `dcn.rs:149-163`:

```rust
let val = ((d as i32 * freq * 13) % 255) - 127;   // "depth embedding" = công thức giả ngẫu nhiên
```

* Không trainer nào chạm tới các bảng này (`train.rs`/`datagen.rs` chỉ sinh binpack cho NNUE).
* Spec (§24) yêu cầu ngưỡng phải **học/tune, không hard-code**. `spsa_tuner.py` chỉ tune 9
  tham số search cổ điển (`RFP_Margin`, `LMR_Base`, … — các UCI option này **có thật** trong
  `setoption.rs`, nên hạ tầng tune dùng được ngay).
* Rủi ro kèm theo: đây là các pruning *thêm* trên nền đã có RFP/LMP/history pruning. Chưa có
  bằng chứng (SPRT) rằng chúng dương Elo. Việc đã có toggle cho từng cái là điểm cộng lớn.

### 3.4. README ≠ code

* README quảng cáo **SCR (Sibling Cutoff Rate Pruning)** — không có module SCR trong `src/`;
  chỉ có 1 test tên `sibling_cutoff_rate_reduces_all_node_late_moves` (`lmr.rs:309`), logic
  thật là nhánh `!is_pv_node && !found_pv && move_count>=6 && momentum>=-100`.
* **PMT (Positional Momentum Tracking)** = biến `momentum` đã nêu. Không phải hệ thống riêng.
* Nên đồng bộ doc để tránh đánh giá sai khả năng engine (đặc biệt khi đối chiếu với spec 33 tiêu chí).

### 3.5. Không có MultiPV

`multipv` 0 hit ⇒ không thể debug personality kiểu "top candidates + lý do" (tiêu chí 28) và
không thể làm suite đo behavior dựa trên candidate ranking. Đây là **blocker kỹ thuật** cho
Phase 0 của roadmap.

### 3.6. Qsearch không sinh check

`quiescence.rs`: chỉ capture (SEE ≥ −74) + promoted-queen quiet, khi `in_check` thì chỉ evasion.
Tiêu chí 13 yêu cầu "expand forcing move search" — hiện chưa có nhánh check-generation.

---

## 4. Map 33 tiêu chí → 8 subsystem (kiến trúc triển khai đề xuất)

| # | Subsystem | Tiêu chí phủ | File đề xuất | Trạng thái hiện tại |
| :-: | :-------- | :----------- | :----------- | :------------------ |
| S1 | **State & Phase Controller** (state machine DEFEND→…→CONVERT) | 17, 22, 23, 25, 32 | `src/search/position_state.rs` | không có |
| S2 | **Counterplay Model** (CPI từ NodeThreats) | 3, 9, 20, 21 | `src/search/counterplay.rs` | không có (nguyên liệu đã có sẵn) |
| S3 | **Risk Envelope & Must-Try Gate** | 4, 10, 11, 12 | `src/search/risk.rs` | không có |
| S4 | **Opportunity / Concession Tracker** | 5, 6, 19, 26 | `src/search/concession.rs` | không có (cần feature ngoài NNUE) |
| S5 | **Pressure Planner** (freedom/plan restriction, pressure trajectory) | 7, 8, 21 | `src/search/pressure.rs` | chỉ có `momentum` |
| S6 | **Attack / Conversion Controller** | 13, 14, 16, 24 | `src/search/attack.rs` + `conversion.rs` | một phần (TCE/LQT/Syzygy) |
| S7 | **Recovery / Reset** | 16, 18 | gộp vào S1+S3 | không có |
| S8 | **Metrics & Verification Harness** (16 metric + Style Suite) | 24, 27, 28, 29, 30, 31 | `tools/aprm_suite.py`, `tests/aprm.rs`, MultiPV | không có |

Ghi chú thiết kế bắt buộc:

1. **TT-safety**: S1–S5 phải đi qua kênh "pruning/ordering/extension/time" *hoặc* state phải
   được hash vào TT key. Nếu không sẽ tái diễn lỗi mà SPS đã tránh được (per-thread optimism).
2. **Root-first**: must-try (S3) và candidate classification (28) nên làm ở root trước, dùng
   `MoveList` + `select_speculative_replies` (`smp_precompute.rs:14`) làm khuôn mẫu — đã có sẵn
   scoring TT-move/capture/promotion để mở rộng.
3. **Đo trước khi thêm** (Phase 0): spec nói ngưỡng phải tune ⇒ không thể tune khi chưa có thước
   đo. Dùng UCI toggle cho từng subsystem như `cfss_enabled`/`dad_enabled` để A/B bằng
   `spsa_tuner.py`/fastchess đã có.

---

## 5. 16 metric: nguồn dữ liệu đã có vs cần thêm

| Metric | Ý nghĩa | Dữ liệu cần | Hiện có gì |
| :----- | :------ | :---------- | :--------- |
| DSI | Defensive Stability Index | eval drop / error rate trong position bị ép | chưa có suite; có `search_state.score` để log |
| CDI | Counterplay Denial Index | CPI đối thủ sau nước đi | **NodeThreats** (cần thêm mobility/breaks) |
| ODI | Opportunity Detection Index | concession do đối thủ tạo, engine phản ứng trong N ply | cần tracker S4 + log |
| AUI | Attack Urgency Index | attack window (số ply còn tồn tại) | chưa có |
| MTR | Must-Try Recognition Rate | tỉ lệ window bị đóng mà engine không chọn attack | chưa có (cần S3) |
| RAE | Risk-Adjusted Aggression Efficiency | gain / risk của nước tấn công | chưa có (cần S3) |
| PPI | Pressure Persistence Index | số ply duy trì pressure | cần định nghĩa pressure + S5 |
| PEI | Pressure Efficiency Index | ΔCPI / Δrisk | cần S2+S3 |
| IOI | Initiative Ownership Index | tỉ lệ nước đối thủ là phản ứng | cần S2 + phân loại reply |
| MAI | Mistake Amplification Index | mức khai thác lỗi đối thủ | cần S6 |
| PCR | Premature Attack Rate | tấn công khi chưa đủ justification | chỉ đo được khi có S3 |
| OMR | Opportunity Miss Rate | bỏ lỡ cơ hội | như MTR |
| FRI | Freedom Reduction Index | số legal move/plan của đối thủ giảm | cần movegen trên board đối thủ (rẻ: `generate_moves` + đếm legal) |
| PCI | Plan Constraint Index | plan bị chặn (break, route, expansion) | cần S5 (feature ngoài NNUE) |
| CRI | Conversion Reliability Index | tỉ lệ thắng khi eval ≥ ngưỡng | có `wdl` + `search_state.score` + log game |
| RSI | Recovery / Reset Index | khả năng hồi phục sau khi eval xấu đi | cần S7 |

Kết luận: **FRI, CRI, CDI là 3 metric có thể làm ngay** bằng dữ liệu engine đã có (movegen,
NodeThreats, score/WDL) — nên làm trước để có baseline.

---

## 6. Roadmap đề xuất (mỗi phase có phép đo, không phase nào là "viết xong mới biết đúng/sai")

| Phase | Nội dung | Deliverable | Đo bằng |
| :---- | :------- | :---------- | :------ |
| **0** | Fix contempt; thêm MultiPV + `info string` diagnostics (state, CPI, momentum, freedom); đồng bộ README | patch nhỏ, rủi ro thấp | `cargo test`, unit test mới |
| **1** | S2 Counterplay Model: CPI node-local từ NodeThreats + mobility đối thủ; wire vào RFP margin / extension / quiet ordering (TT-safe) | `counterplay.rs` + toggle | SPRT vs baseline (`tournament.yml`), CDI |
| **2** | S1+S3 tối thiểu: `PositionState` + `RiskEnvelope` + must-try gate ở root (chỉ đổi depth/time/extension, không đổi eval) | `position_state.rs`, `risk.rs` + toggle | AUI/MTR/PCR |
| **3** | S4+S5: concession tracker + pressure/freedom; cần feature ngoài NNUE (pawn islands, defender count, break availability) | `concession.rs`, `pressure.rs` | ODI/PPI/PEI/FRI/PCI |
| **4** | S6 Attack & Conversion: window-closing check ở root, verification search với depth mở rộng, conversion threshold + Syzygy | `attack.rs`, `conversion.rs` | RAE/MAI/CRI, Elo |
| **5** | S7 Recovery/Reset + tune toàn bộ ngưỡng (SPSA mở rộng) + **train lại hoặc loại bỏ** các bảng hard-code §3.3 | tune config | Elo + toàn bộ suite hành vi |

Nguyên tắc bất di bất dịch khi triển khai:

* **Không thêm heuristic mà không có toggle + phép đo** (đúng pattern `cfss_enabled`… đang có).
* **Không để state ảnh hưởng leaf eval** trừ khi hash vào TT/eval-cache key (bài học SPS).
* Mỗi subsystem mới phải có test theo pattern `tests/tce_equiv.rs` (equivalence/reference test)
  để tránh regression âm thầm.

---

## 7. Kết luận trả lời câu hỏi

> **Engine có đáp ứng tiêu chí Adaptive Pressure không? → Không.**

* Whale hiện là **search engine mạnh, đúng chuẩn kỹ thuật** (NNUE, TT lock-free, SMP, pruning
  hiện đại, Syzygy), nhưng **chưa có tầng hành vi** (behavioral policy) mà spec yêu cầu:
  không state machine, không risk envelope, không must-try, không counterplay model, không
  concession/weakness tracking, không conversion threshold, không metric.
* 8/33 tiêu chí có "nguyên liệu" (momentum, optimism, DCN, TCE/LQT, DAD, NodeThreats, Syzygy,
  personas, UCI toggles) nhưng đều dùng **sai mục đích** so với spec (search-efficiency thay vì
  behavior policy).
* 25/33 tiêu chí trắng hoàn toàn, **bao gồm toàn bộ hệ metric** — nghĩa là hiện tại cũng
  **không thể đo** được engine có "adaptive pressure" hay không, kể cả bằng tay.
* Điểm tích cực: hạ tầng để làm đúng spec **đã có sẵn** (threat snapshot, UCI tier toggles,
  SPSA tuner, fastchess tournament, equivalence-test pattern) ⇒ chi phí triển khai thấp hơn
  nhiều so với việc phải viết lại search.

Thứ tự đúng: **đo trước (Phase 0) → CPI/counterplay (Phase 1) → state + risk + must-try
(Phase 2) → pressure/concession (Phase 3) → attack/conversion (Phase 4) → reset + tuning
(Phase 5).** Bỏ Phase 0 thì mọi phase sau chỉ là thêm heuristic không kiểm chứng được — đúng
kiểu "một đống heuristic chồng chéo" mà spec cảnh báo.



