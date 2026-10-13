# DSpark「按 confidence 裁切 draft token」在 4×H20 / Qwen3.8-27B 上的实现与实测

日期：2026-09-17。硬件与运行时同 `h20_qwen38_27b_dspark.md`（4×H20 96GB，TP=4，
bf16 target + bf16 KV，`/workspace/models/Qwen3.8-27B` + `/workspace/models/Qwen3.8-27B-DSpark`，
sglang `0.5.19.dev39+g91a20e1`）。本文件回答两个问题：

1. sglang 里到底有没有「依据 draft confidence 裁切 draft token / verify 长度」的实现？
2. 在当前仓库 + 这个模型上能不能开起来？开起来有没有收益？

## 0. 结论速览

1. **实现是有的**，就是 DSpark 的 **ragged-verify budget scheduler**
   （`sglang/srt/speculative/dspark_components/dspark_planner.py` +
   `kernels/ops/speculative/dspark/dspark_schedule.py`）：
   draft 的 **confidence head** 出每个位置的概率 → 算前缀存活率 `cumprod` → 每步在
   「预期接受 token / 步耗时」上取最优的全局 verify 预算 τ* → 把 τ* 个额外 verify
   名额按存活率从高到低分配给 (请求, 位置)，**低置信度的 draft 尾巴直接不 verify**。
2. **但默认不开**：默认 `SGLANG_RAGGED_VERIFY_MODE=static` = **verify 全窗口**，
   零裁切。官方 cookbook 的 DSPARK recipe 也是 static（见 `h20_qwen38_27b_dspark.md`
   §1 与官方 recipe 对照），**不含任何 confidence/预算相关参数**。
3. **开启需要的三样东西**：`SGLANG_RAGGED_VERIFY_MODE=compact`（完整预算调度）或
   `=cap-accept`（cap 式裁切）、`--speculative-dspark-sps-table-path`（离线画像的
   SPS 成本表，不给我会退化成 verify-all）、可选
   `--speculative-dspark-confidence-sts-path`（STS 校准）。
4. **本模型上实测结果：compact 直接崩、cap-accept 能跑但更慢**：
   - `compact` + 已画像 SPS 表 → 进程在第二次 forward 就死：
     `RuntimeError: shape '[13, -1]' is invalid for input of size 112`
     （`forward_batch_info.py:1102 compute_spec_mrope_positions`）；
     关掉 overlap scheduler 再试仍死（`'[9, -1]'` vs size 80）→ **不是并发/时序问题，
     是 ragged 布局与 mrope positions 不兼容**（Qwen3.8-27B 是 VL/mrope 模型）。
   - `cap-accept` + 同一张 SPS 表 → 能服务，但 rate 64 吞吐 **10.27 req/s
     （static 同 cap 12.98，-21%）**，accept 3.44 → **2.64**，TPOT 26.05 → 33.13 ms：
     每步耗时没降（≈87.5 vs 89.6 ms）却少了接受 token，属于纯亏。
     ※ 机制推断：cap-accept 用「按行 padding 到 batch 内最大 verify 长度」的
     dense 布局，只要有请求拿到满窗宽，整个 forward 宽度就还是 8，裁切只省了
     commit 不省算力（与实测的"步耗时不变 + accept 下降"一致，未逐层验证）。
5. **SPS 画像本身在 H20 上是通的**（本次产出 `docs/perf/results/dspark_sps_g7.json`，
   6 探针）：它要求服务器带 `SGLANG_DSPARK_ENABLE_SPS_RECORD=1` +
   `SGLANG_SIMULATE_ACC_LEN=1.0` 且跑 static 模式；off-diagonal 的 2D 加性表则要求
   compact 模式 —— 而 compact 在本模型上会崩，所以拿不到 2D 表。
6. STS 校准 JSON 本次**没采到数据**：默认 `SGLANG_DSPARK_FOLDED_PROPOSAL=1`（折叠
   proposal）时 `observe_verify_step` 里的 STS 采集分支被跳过（见 §4），需要额外设
   `SGLANG_DSPARK_FOLDED_PROPOSAL=0` 才能落 `*.pt` shard（基于代码阅读的结论，
   未单独复跑验证）。

一句话：**能力齐备、工具齐备，但这条路在 Qwen3.8-27B（mrope 模型）上还没打通；
H20 上唯一能跑的 cap-accept 形态是净负收益。** 要用就先修
`compute_spec_mrope_positions` 对 packed/ragged verify 布局的支持。

## 1. 机制（代码位置）

| 环节 | 位置 | 说明 |
|---|---|---|
| confidence head | `srt/models/dspark.py`（`confidence_head` / `VanillaMarkov`） | 出每个 draft 位置一个 [0,1] 置信度；`confidence_head_with_markov=True`；`_last_confidence_raw` 供观测/采集 |
| STS 校准 | `dspark_components/dspark_sts.py` | 每位置温度标定 `p^(1/T_k)`，改善存活率的校准（ECE）；可选 |
| 前缀存活率 | `kernels/ops/speculative/dspark/dspark_schedule.py::compute_sort_survival` | `cumprod(confidence)` = 「前 k 个位置全被接受」的概率 |
| 每步预算 τ* | `dspark_planner.py::compute_verify_token_budget` | 候选 =(请求×位置) 全部展开，按存活率降序；`τ* = argmax_τ (bs + Σtop-τ survival) × SPS(bs + τ)`，SPS 来自画像表 |
| 分配 verify 长度 | `dspark_schedule.py::schedule_verify_lens_topk` | 取前 τ* 个候选，每请求 verify 长度 ∈ [`min_verify_len`=1, γ+1]；没入选的尾部直接不 verify |
| 模式开关 | `srt/speculative/ragged_verify.py` | `static` / `cap-accept` / `compact`（`SGLANG_RAGGED_VERIFY_MODE`，默认 static） |
| 调试旋钮 | `POST /set_internal_state {"server_args":{"dspark_force_budget_frac":f}}` | 画像工具用它把预算钉在某个比例（off-diagonal 2D 表） |

三种模式的差别（`dspark_planner.py::schedule_layout`）：

| 模式 | 布局 | 是否裁切 | 依赖 |
|---|---|---|---|
| `static`（默认、官方 recipe） | 无 ragged 布局 | **不裁切**（verify γ+1 全窗） | 无 |
| `cap-accept` | 按行 padding 的 capped dense 布局（`RaggedVerifyLayout.cap`） | 按预算 cap 每请求 verify 长度；外层用 `block accept len`/`cap len` 记账 | SPS 表（否则退化为 verify-all） |
| `compact` | packed（`qo_indptr`）ragged 布局 | 完整 τ* 预算调度 + top-k 分配 | SPS 表（否则退化为 verify-all） |

相关的 server args / env（`server_args.py`）：

- `--speculative-dspark-sps-table-path <json>`：`dspark_sps_profiler` 产出；
  不带则表是 uninitialized/flat，**预算退化为 verify-all、零收益**（启动时会 warning）。
- `--speculative-dspark-confidence-sts-path <json>`：`dspark_sts_fit` 产出；无损性不受影响。
- `--speculative-dspark-align-verify-tokens-to-graph-tier`（仅 compact）：把 forward
  本来就付了的 CUDA-graph padding 变成真 token（按置信度补满到图档位），默认关。
- env：`SGLANG_DSPARK_ENABLE_SPS_RECORD=1`（画像必需）、`SGLANG_SIMULATE_ACC_LEN=1.0`
  （画像必需：让 spec 完全无效、KV 条件确定性）、`SGLANG_DSPARK_STS_COLLECT_PATH`、
  `SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS`（默认 2）、
  `SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS` / `SGLANG_DSPARK_DEBUG_DUMP`。

## 2. 开启步骤（可复现命令）

```bash
cd /workspace/projects/sglang-speedup-nvidia

# 1) 起一个「画像用」DSpark server：static 模式 + SPS 记录 + 模拟接受
SGLANG_DSPARK_ENABLE_SPS_RECORD=1 SGLANG_SIMULATE_ACC_LEN=1.0 \
CAP=48 TAGSUF=_spsprof48 RATES=16 PROFILE_SPS=docs/perf/results/dspark_sps_g7.json \
SPS_BS="8 16 24 32 40 48" \
  bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark     # 脚本会在 warmup 后调 profiler

# 等价的手工调用（对已在跑的 static server）：
.venv/bin/python3 -m sglang.benchmark.dspark_sps_profiler all \
    --base-url http://127.0.0.1:30000 --out docs/perf/results/dspark_sps_g7.json \
    --batch-size 8 16 24 32 40 48 --min-steady-seconds 5

# 2) 带表重启（二选一）
SGLANG_RAGGED_VERIFY_MODE=cap-accept DSPARK_SPS=docs/perf/results/dspark_sps_g7.json \
CAP=48 TAGSUF=_capaccept bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark
SGLANG_RAGGED_VERIFY_MODE=compact   DSPARK_SPS=docs/perf/results/dspark_sps_g7.json \
CAP=48 TAGSUF=_compact   bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark

# 3) 可选：STS 校准（先采数据再拟合）
#    采集需要关掉折叠 proposal，否则观测路径不落盘
SGLANG_DSPARK_FOLDED_PROPOSAL=0 SGLANG_DSPARK_STS_COLLECT_PATH=docs/perf/results/dspark_sts/shards \
CAP=48 TAGSUF=_sts bash docs/perf/scripts/run_qwen38_27b_h20.sh tp4_dspark
.venv/bin/python3 -m sglang.benchmark.dspark_sts_fit \
    --data-glob 'docs/perf/results/dspark_sts/*.pt' --out docs/perf/results/dspark_sts_g7.json
#    再把 DSPARK_STS=<json> 一起传进去

# 4) 需要 2D（off-diagonal）加性成本表时：服务器必须跑 compact，并给 profiler 传 --fracs
.venv/bin/python3 -m sglang.benchmark.dspark_sps_profiler all \
    --base-url http://127.0.0.1:30000 --out docs/perf/results/dspark_sps_g7_2d.json \
    --fracs 1.0 0.75 0.5 0.25
```

启动自检（server 日志）：`DSpark ragged-verify scheduler enabled (mode=…, lag=…,
relay_lag=…, sps_table=…, graph_tier=…)`；若同时出现
`SPS table is uninitialized (flat) …` 则说明表没生效。

## 3. H20 实测

### 3.1 SPS 成本表（成功）

`docs/perf/results/dspark_sps_g7.json`（6 探针，`conversion_formula: batch_tokens =
num_running_reqs_per_rank × verify_num_draft_tokens`）：

| batch_tokens | 64 | 128 | 192 | 256 | 320 | 384 |
|---|---|---|---|---|---|---|
| steps/s | 65.06 | 39.12 | 27.39 | 22.56 | 18.03 | 15.78 |
| 步耗时 (ms) | 15.4 | 25.6 | 36.5 | 44.3 | 55.5 | 63.4 |
| 边际 (ms/token) | — | 0.159 | 0.170 | 0.122 | 0.174 | 0.124 |

即 bs48/窗口 8 的 verify 步在「spec 完全无效」条件下 63.4 ms；实测真实服务下
γ7 静态步耗时 89.6 ms（差在真实接受/KV 历史/对齐），manifest 里也记了这条
`static_conditioning_caveat`。首次尝试 cap128 画像时 **OOM**（`cap128` 下
verify 中间态 18.14 GB，池子只留 1.79 GB 运行时余量），降到 cap48 才成功。

### 3.2 compact：崩溃（2/2）

| 配置 | 结果 | 报错 |
|---|---|---|
| `compact` + SPS 表，默认 overlap | 第 2 个 forward 后 scheduler 异常，4 卡同时死 | `RuntimeError: shape '[13, -1]' is invalid for input of size 112` @ `forward_batch_info.py:1102` |
| `compact` + SPS 表 + `--disable-overlap-schedule` | 同样死（`relay_lag=0`） | `shape '[9, -1]' is invalid for input of size 80` @ 同一行 |

两次都是 `compute_spec_mrope_positions` 里 `seq_positions.view(batch_size, -1)`：
ragged/padded 的 verify 位置张量长度（112 / 80）不是 `bs × 窗口` 的整数倍，
而该函数只支持均匀布局。关掉 overlap 后仍然复现 ⇒ 与 confidence relay 的时序
错位无关，是 **ragged 布局本身**不被 mrope positions 路径支持。

### 3.3 cap-accept：能跑，但净负收益

512 in / 256 out，TP=4，cap=48，同一张 SPS 表（对照列是同 cap、同模型的 static 全窗）：

| rate | 模式 | req/s | out tok/s | accept | TPOT (ms) | E2E50 (ms) | 每步 (ms) |
|---|---|---|---|---|---|---|---|
| 4 | static 全窗（γ7） | 4.08 | 550 | 3.36 | 5.06 | 627 | 17.0 |
| 4 | **cap-accept** | 4.07 | 549 | 3.39 | 5.60 | 673 | 19.0 |
| 16 | static 全窗 | 10.65 | 1 507 | 3.42 | 26.96 | 5 134 | 92.2 |
| 16 | **cap-accept** | 8.59 | 1 216 | **2.81** | 34.59 | 8 084 | 97.2 |
| 64 | static 全窗 | **12.98** | **1 700** | 3.44 | 26.05 | 11 545 | 89.6 |
| 64 | **cap-accept** | **10.27** | **1 345** | **2.64** | 33.13 | 16 020 | 87.5 |

- rate 4（低并发、带宽区）几乎无损；rate 16/64 分别 -19% / -21% 吞吐，
  accept 掉到 2.6–2.8，TPOT 反而变差。
- 步耗时（TPOT × accept）**没降**（87.5 vs 89.6 ms @ bs48），说明裁切没有换来更小的
  forward：与 dense capped 布局「宽度 = batch 内最大 verify 长度」的推断一致
  （详见 §0.4 的 ※ 与 §4）。
- 服务器侧记账佐证：`accept len: 2.56, cap len: 2.77, block accept len: 3.65`
  —— 即"本来能接受 ~3.6 个（block accept），实际只 commit 2.56"。

### 3.4 STS 采集：本次未落盘

三次带 `SGLANG_DSPARK_STS_COLLECT_PATH` 的运行目录均为空。代码路径：
`DSparkStepObservers.observe_verify_step()` 里

```python
if not proposal_folded:
    self._maybe_record_sts_collect(...)   # 只有非折叠路径才采集
```

而 `SGLANG_DSPARK_FOLDED_PROPOSAL` 默认 `True`（本工况全 greedy，必然折叠），
所以采集分支被跳过。要采数据需 `SGLANG_DSPARK_FOLDED_PROPOSAL=0`
（未单独复跑确认）。**STS 未参与本次任何测量。**

## 4. 根因与判断

1. **能力侧**：confidence 裁切在 fork 里实现得相当完整（预算最优解 + top-k 分配 +
   STS 标定 + 2D 成本表 + 观测指标 + 强制预算调试端点），并且本次已在 H20 上
   成功画像出 1D SPS 表。
2. **模型侧卡点**：ragged verify 布局要求 positions 是 packed/padded 的一维张量，
   而 Qwen3.8-27B 走的是 VL 的 mrope 分支
   （`model_config.model_is_mrope` → `compute_spec_mrope_positions`），该函数
   硬编码 `view(batch_size, -1)`，只接受均匀布局。**因此 compact 在本模型上必然崩**；
   cap-accept 之所以能跑，是因为它用 padded-capped（均匀宽度）布局绕过了这一点。
3. **收益侧**：即便绕过了布局问题（cap-accept），在 4×H20 + bf16 target 上也没有
   正收益：阈值裁切省不到步耗时（dense 宽度由 batch 内最大值决定），却把
   accept 从 3.4 砍到 2.6。**要让它有意义，需要 compact 的 packed 布局真正生效
   （即先修 mrope positions），或让 cap-accept 的行宽也随预算收缩。**
4. **对部署的结论**：维持 `h20_qwen38_27b_dspark.md` 的建议不变
   —— 4×H20 上 Qwen3.8-27B 用自带 MTP4；要用 DSpark draft 就压 γ（窗口 4）。
   本文件这条（confidence 裁切）**暂时不要在生产开**：compact 崩、cap-accept 亏。

## 5. 数据有效性

- SPS 画像运行：`Load weight begin. avail mem` 93.5–93.8 GB（GPU 干净）；
  6 个探针 `match_fraction=1.00`、`reached_target=true`、自检通过。
  （第一次 cap128 画像因运行时余量只有 1.79 GB 而 OOM，已弃用，日志保留。）
- cap-accept 服务运行：0 Traceback / 0 OOM；100/200/400 请求全部成功。
- compact 两次运行：进程崩溃（scheduler 异常 + 4 卡退出），**数据无效**，
  仅作为"该路径在本模型不可用"的证据留档。
- 所有测量都遵守项目约束：GPU 跑前跑后 0 MiB、日志 `tee` 进 `docs/perf/logs/`、
  H20 不带 `--cuda-graph-backend-prefill=disabled`。
- 未能验证的部分（如实标注）：① 2D 加性表没画像（需要 compact 服务）；
  ② STS 未采集；③ cap-accept 的 dense 宽度机制是推断而非逐层验证；
  ④ `--speculative-dspark-align-verify-tokens-to-graph-tier` 未测（同样需要 compact）。

## 6. 日志清单（docs/perf/logs/）

| 文件 | 内容 |
|---|---|
| `../results/dspark_sps_g7.json` / `../results/dspark_sps_g7.json.manifest.json` / `../logs/dspark_sps_g7.log` | H20 上画像出的 1D SPS 成本表（6 探针）、每轮 summary 与 profiler 日志（原始逐条记录 `../results/dspark_sps_g7.{records,rounds}.jsonl` 命中仓库 .gitignore 的 `*.jsonl`，仅本地留存） |
| `server_..._tp4_dspark_cap48_spsprof48.log` + `bench_..._spsprof48_rate16.log` + `runner_..._spsprof3.log` | 画像运行（static + SPS_RECORD + SIMULATE_ACC_LEN=1.0，cap48） |
| `server_..._tp4_dspark_cap48_capaccept.log` + `bench_..._capaccept_rate{4,16,64}.log` + `runner_..._capaccept.log` | cap-accept + SPS 表（cap48，可服务，吞吐 -21%） |
| `server_..._tp4_dspark_cap48_compact.log` + `runner_..._cap48_compact.log` | compact + SPS 表，崩（`[13,-1]` vs 112） |
| `server_..._tp4_dspark_cap48_compact_noovl.log` + `runner_..._compact_noovl.log` | compact + `--disable-overlap-schedule`，仍崩（`[9,-1]` vs 80） |
| `server_..._tp4_dspark_cap128_spsprof.log` + `runner_..._spsprof{,2}.log` | 画像的前两次失败（缺 SPS_RECORD / 缺 SIMULATE_ACC_LEN / cap128 OOM） |
| `server_..._tp4_dspark_cap128_static.log` + `bench_..._cap128_static_rate{4,16,64}.log` | 对照：γ7 static 全窗 cap128（12.99/10.69，复核 09-16 的 13.11/10.67） |
