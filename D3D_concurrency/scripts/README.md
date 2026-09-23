# 启动说明（SDAR-1.7B → Qwen3-8B）

从项目目录运行：

```bash
cd /inspire/ssd/project/sais-bio/public/linguancong/FUN_SPACE/T3D/D3D_concurrency
```

脚本使用已验证的 D3D 环境绝对路径，不需要先 conda activate。
默认两张 H800：CUDA_VISIBLE_DEVICES 中第一张运行 target Qwen3-8B，第二张运行 draft SDAR-1.7B。
默认 checkpoint 与已测 AIME 使用的模型一致。不同测试请串行运行，避免 GPU 争抢污染 TPS。

## 三个入口

```bash
# 先检查环境、HF greedy / 原版 SDAR 一致性和多并发补位：
bash scripts/smoke.sh

# 复现恢复进程后的 AIME：30 题，C=1，上限 2048，2 轮：
bash scripts/aime_c1.sh

# 正式三数据集表：全量，C=1/8/16/32，上限 2048，每档 3 轮：
bash scripts/table.sh
```

| 脚本 | 默认数据集 | 并发档 | 样本数/数据集 | 输出上限 | 重复 | 独立正确性验证 |
|---|---|---|---:|---:|---:|---|
| smoke.sh | Math500 | 1 8 16 32 | 40 | 32 | 1 | 开 |
| aime_c1.sh | AIME25 | 1 | 全量 30 | 2048 | 2 | 关 |
| table.sh | Math500 / HumanEval / MT-Bench | 1 8 16 32 | 全量 | 2048 | 3 | 关 |

正式表需要比此前 64 题、256-token 的试跑更长时间。想先得到快速表：

```bash
MAX_SAMPLES=64 MAX_NEW_TOKENS=256 REPEATS=1 bash scripts/table.sh
```

这会明确记录为子集/短输出实验，不能与全量 2048-token 的数字混称。
MT-Bench 目前仅第一轮问题，不是完整多轮对话评测。

## 覆盖参数

与 D3DV2 的脚本类似，通过环境变量覆盖。参数名见下表；不是任意 D3DV2 参数都可直接使用。

```bash
# 只跑 Math500，指定两张物理 GPU：
CUDA_VISIBLE_DEVICES=0,1 DATASETS=math500 CONCURRENCY='1 8 16 32' \
  MAX_SAMPLES=64 MAX_NEW_TOKENS=256 REPEATS=1 bash scripts/table.sh

# 改 draft 微批大小以控制显存，C 仍然是 16/32：
DRAFT_MICROBATCH=2 CONCURRENCY='16 32' bash scripts/table.sh

# 只显示展开后的命令，不加载模型，也不创建结果目录：
bash scripts/table.sh --dry-run
bash scripts/benchmark.sh --help
```

| 环境变量 | 含义 |
|---|---|
| DATASETS | 空格分隔的数据集名，如 `math500 humaneval mt-bench aime25` |
| CONCURRENCY | 空格分隔的并发档，如 `1 8 16 32`，各档顺序执行 |
| MAX_SAMPLES | 每数据集取前 N 题；0=全部 |
| MAX_NEW_TOKENS | 每请求最多生成的 token 数 |
| REPEATS | 每个数据集/并发档重复次数 |
| DRAFT_MICROBATCH | draft 每次前向处理多少个请求；默认 4 |
| ANCHOR_NUM | 正常预测每个接受位置的 anchor 数，默认 3 |
| FAST_MISS_ANCHORS | Fast miss 时首个接受位置的候选数；0=与 ANCHOR_NUM 相同 |
| PAGES | 每个模型的 KV 页数；默认 512，每页 256 个位置 |
| VALIDATE | 1=加载独立 HF/SDAR 参考进行短输出验证，0=不加载 |
| IGNORE_EOS | 默认 0，自然终止；设 1 会改变工作负载，不用于自然 EOS 对照 |
| CUDA_VISIBLE_DEVICES | 两张可见 GPU；第一张 target，第二张 draft |
| DRAFTER / VERIFIER | 覆盖 checkpoint 路径；默认使用已验证的 1.7B/8B，其他架构不保证支持 |
| D3D_PYTHON | 覆盖 Python 解释器绝对路径 |
| OUT_DIR | 指定一个尚不存在的结果目录；相对路径相对启动时的工作目录 |

block=16、greedy、BF16、denoising=1 目前固定；anchor 默认 3，可用 `ANCHOR_NUM=1` 调小。
其他算法参数没有开放任意修改。PAGES 是固定池容量，不是 GPU 百分比；长提示词或
更长输出可能耗尽池，需要结合显存调整页数/并发。draft 微批调小能减少临时激活，不能
替代足够的长期 KV 容量。没有抢占/自动 offload。

## 输出和后台运行

默认写到 `results/runs/<UTC时间>_<PID>/`。指定 OUT_DIR 时拒绝覆盖已有目录。

- `command.sh`：展开后的模型运行、审计、出表命令和关键环境变量；重放会写入相同输出路径。
- `run.log`：终端日志。`RESULT` 行表示完成一个数据集/并发档/重复轮。
- `benchmark.json`：实际 token、耗时、每请求状态、GPU 峰值、代码哈希、参数。
- `table.md` / `table.csv` / `table.tex`：审计通过后生成的吞吐表。

后台执行例子（`table_full_01` 必须是新目录）：

```bash
mkdir -p logs
OUT_DIR="$PWD/results/runs/table_full_01" \
  nohup bash scripts/table.sh > logs/table_full_01.log 2>&1 &
echo $!
tail -f logs/table_full_01.log
```

表格每格是实际输出总 token / 总生成时间，包含 prefill 和最后收尾，排除加载、预热、
tokenization、文件输出；多轮使用总 token / 总时间合并。当前没有 AR 性能基线，
所以不生成 `(x倍)`；`VALIDATE=1` 的 HF 对照只是正确性验证。

## 当前方案和文件

```text
bench.py：取数据/构造提示词 → 预热 → 按数据集、C、重复轮测量
  engine.py 主进程：请求队列、最多 C 个活动请求、SDAR draft（GPU1）
       │ CPU token、请求 ID、前缀长度、KV 回滚/释放命令
       └─ target_process.py 常驻进程：batched Qwen3 target（GPU0）
```

一轮中，target 将全部活动请求的当前 16-token block 一起验证；与此同时，draft
按微批（默认 4 请求）预测每个请求各种接受长度/anchor 的下一块。每请求有 46 个
候选分支，连同当前块是 752 个查询位置。两进程分别向两张 GPU 发射算子，规避此前
双线程重叠不佳的问题。收到验证结果后，每个请求独立提交接受前缀、选择命中的预测块，
未命中则补一次 recovery；然后回滚逻辑 KV。结束的请求释放页，下一调度轮补位。
这是按轮同步的动态补位，没有 vLLM 那种完整服务端/抢占调度，也不是开 C 份模型。

C=32 指最多 32 个在途请求，不是每个 draft 微批都必须有 32 个请求；每轮 target
看到全部活动请求，draft 以最多 4 个请求为一组顺序执行。到队尾没有待处理请求时，
实际并发自然下降。增加 C 不保证吞吐线性增长。

| 文件 | 职责 |
|---|---|
| scripts/benchmark.sh | 公共启动、参数展开、独立结果目录、日志、审计、出表 |
| scripts/smoke.sh / aime_c1.sh / table.sh | 三种默认实验配置 |
| run.sh | Python 环境、两 GPU 可见性、离线加载和 eager 执行配置 |
| bench.py | 数据、预热、计时、参数/代码哈希与结果保存 |
| engine.py | 活动请求管理、draft 打包、接受/命中/recovery、队列补位 |
| target_process.py | 常驻 target 进程、CPU IPC、worker 自有 KV 状态及显存统计 |
| backend.py | 模型 Runner、分页 KV 分配、prefill/forward、词表投影 |
| sdar_adapter.py | 原 SDAR 层接入分页 attention |
| vendor/ssd/layers/attention.py | target 的 paged FlashAttention，draft 的分页 gather + SDPA 自定义 mask |
| validation.py / tests/test_masks.py | 原 SDAR 回归、mask/页管理/argmax tie 测试 |
| audit_results.py / report.py | 审计 JSON，生成表格 |
| thread_engine_reference.py / profile*.py | 旧线程实现和性能诊断；正式启动不使用它们 |

当前是 SSD 派生的自定义 PyTorch 引擎。target 使用 SSD Qwen3 层和 FlashAttention；
draft 保留原 HuggingFace 风格 SDAR 实现并改造 attention/cache。不是 vLLM，也不是
调用 `transformers.generate()` 完成主实验；后者仅在独立正确性验证中使用。

## Fast / Neural Backup 对照

正式方案采用 `BACKUP=neural`，保持原来的同步补算。实验性 `BACKUP=fast` 在未命中时先返回
随机词表内候选，target 开始验证后，draft 才执行 masked anchor seeding 和下一轮
packed 预测；随机候选不会绕过 target 检查。初始请求仍使用共同的 neural bootstrap。
这是 SSD fast 策略的 SDAR 移植，未复现其 AR glue-decode。
新版 `BACKUP=fast` 对 miss 仅预测接受首 token 后的分支，减少无效打包计算；
旧实现保留为 `BACKUP=fast_legacy`。这属于 miss 专用的稀疏分配，并非 SSD
固定总预算的 fanout 重分配。
指定 `FAST_MISS_ANCHORS=16 ANCHOR_NUM=1` 可测试集中预算：正常分支共 16 个，
miss 时集中为首个接受位置的 16 个候选。所有候选仍需 target 验证。
仅支持现有 greedy 模式。

```bash
# 相同模型加载下，按 neural/fast、fast/neural 交替进行两轮：
bash scripts/compare_backup.sh --datasets math500 --max-samples 64 \
  --concurrency 1 8 16 32 --max-new-tokens 256 --repeats 2 \
  --output results/backup_comparison/raw.json
python report_backup.py results/backup_comparison/raw.json

# 单独运行某种策略（常规启动器自动保存日志和表格）：
BACKUP=fast MAX_SAMPLES=64 MAX_NEW_TOKENS=256 REPEATS=1 bash scripts/table.sh
```

专用对照入口默认写 `results/backup_comparison/raw.json`，重复运行会覆盖该文件，
保留实验请通过 `--output` 指定新路径。`report_backup.py` 审计后在相同目录写
`summary.json`、`table.md`。比较列 `Fast/Neural` 是 backup 策略比，不是 AR 加速比。
