"""Compare completed process benchmarks against the retained thread/original runs."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'results/aime_diagnosis'
old=json.loads((OUT/'original_rerun.summary.json').read_text())
thread=json.loads((OUT/'current_c1.json').read_text())['cells'][0]
data=json.loads((OUT/'process_c1.json').read_text())
cells=data['cells']
assert len(cells)==data['config']['repeats']
for c in cells:
    assert c['input_sha256']==thread['input_sha256']
    assert c['requests']==30 and c['concurrency']==1
    assert c['target_active_sequences']==0
    assert c['target_free_pages']==data['config']['pages']
    assert c['output_tokens']==thread['output_tokens']
    for before,after in zip(thread['outputs'],c['outputs']):
        assert before['index']==after['index']
        for key in ['tokens','accepts','hits','recoveries']:
            assert before[key]==after[key],(c['repeat'],after['index'],key)
aggregate=sum(c['output_tokens'] for c in cells)/sum(c['elapsed_s'] for c in cells)
summary={
    'original_overall_tps':old['timing']['elapsed_tokens_per_s'],
    'original_decode_tps':old['timing']['aggregate_decode_tps'],
    'thread_tps':thread['tps'],
    'process_repetitions':[{'repeat':c['repeat'],'tps':c['tps'],'elapsed_s':c['elapsed_s'],
                            'output_tokens':c['output_tokens']} for c in cells],
    'process_aggregate_tps':aggregate,
    'process_vs_original_percent':100*(aggregate/old['timing']['elapsed_tokens_per_s']-1),
    'process_vs_thread_speedup':aggregate/thread['tps'],
    'all_30_outputs_and_acceptance_traces_equal_thread_each_repeat':True,
    'all_target_pages_released_each_repeat':True,
}
(OUT/'process_comparison.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
rows='\n'.join(f"| 恢复进程，第 {c['repeat']+1} 轮 | {c['tps']:.2f} | {c['elapsed_s']:.2f} | {c['output_tokens']} |" for c in cells)
report=f'''# 独立进程恢复后的 AIME25 对照

当前正式引擎已恢复 target 常驻独立进程；C=1/8/16/32 共用同一套多并发引擎。
恢复后 {len(cells)} 轮合并吞吐 **{aggregate:.2f} tok/s**，相比线程版
**{summary['process_vs_thread_speedup']:.3f}×**，相比原版同机整体吞吐
**{summary['process_vs_original_percent']:+.2f}%**。

## 完整 AIME25 测量

同一组 30 题、相同顺序，输出上限 2048，自然 EOS，BF16 greedy，
SDAR-1.7B → Qwen3-8B，block=16，anchor=3，全部接受位置，2×H800。
新引擎两轮连续在同一次模型加载后运行，测量排除加载、预热、tokenization 和结果保存，包含 prefill。

| 版本 | 整体 TPS（tok/s） | 整体耗时（s） | 输出 token |
|---|---:|---:|---:|
| 原版历史结果 | 191.77 | 270.42 | 51858 |
| 原版同机复测 | {summary['original_overall_tps']:.2f} | {51858/summary['original_overall_tps']:.2f} | 51858 |
| 恢复前线程版 | {thread['tps']:.2f} | {thread['elapsed_s']:.2f} | {thread['output_tokens']} |
{rows}

合并 TPS = 两轮总输出 token / 两轮总时间，不是单请求 TPS 的平均值。
原版的整体计时还包含 tokenization、结果处理和 JSON 写入，因此跨原版的比例是
近似端到端对照，不是严格同计时边界的内核速度比较；原版同机 decode-only
为 {summary['original_decode_tps']:.2f} tok/s。恢复前后的新引擎计时边界一致。

## 正确性与并发验证

- 4 个 CPU 单元测试通过。
- 4 个 Math500 提示词、每题 32 tokens，与独立 HF target greedy 一致。
- SDAR recovery 一致，46 个 packed 候选分支共 736/736 tokens 与原版一致。
- C=1/8/16/32，各完成 40 个 Math500 请求，覆盖请求完成和队列补位。
  C=8/16/32 各 40/40 个输出与保留的线程版同档位结果一致；C=1 的旧有 4 个样本也一致。
- AIME 每轮的 30/30 个完整输出、accepts、hits、recoveries 均与恢复前线程版完全一致。
  因此恢复前后的吞吐提升没有伴随生成内容或接受过程变化。
- 每轮结束 target 活动请求数为 0，全部 target KV 页释放；引擎同时断言 draft 页全部释放。
- JSON 结果通过吞吐公式、请求数量、并发上限、EOS 和工作负载哈希审计。

与原版 D3DV2 的长序列输出并非逐 token 相同（这一差异在恢复进程前已存在），
本次未重新评测 AIME 解题准确率，也不将跨并发的浮点差异宣称为逐位一致。

## 实现变化

`target_process.py` 使用 spawn 启动常驻 target worker，独占其模型和分页缓存。
主进程仅发送 CPU token 列表、请求 ID 和逻辑前缀长度；draft 运行期间 target
可独立发射 GPU 算子。KV trim/release 更新随下一次请求批量传递，轮末显式冲刷。
worker 异常通过管道返回，关闭引擎时清理 worker。

模型、SDAR 适配层、attention 和词表投影代码未改，torch.compile 仍关闭。
`thread_engine_reference.py` 保留改动前引擎，文件哈希与旧测量记录一致，
供旧 profiling 复现；它不是按 C=1 特殊切换的生产路径。
GPU 峰值分配量现在由实际持有 target 模型的 worker 上报。

## 文件与复现

- `process_c1.json` / `.log`：完整 AIME 两轮与源代码哈希。
- `process_smoke.json` / `.log`：HF、SDAR 和多并发验证。
- `process_comparison.json`：机器可读汇总和一致性检查。
- `current_c1.json`、`original_rerun.summary.json`：保留的线程版和原版基线。
- 原始性能诊断见 [README.md](README.md)。

从 T3D 工作目录执行（会覆盖同名输出）：

```bash
bash D3D_concurrency/run.sh --datasets aime25 --max-samples 0 --concurrency 1 --max-new-tokens 2048 --repeats 2 --output results/aime_diagnosis/process_c1.json
python D3D_concurrency/audit_results.py D3D_concurrency/results/aime_diagnosis/process_c1.json
python D3D_concurrency/compare_process_results.py
```

本次只重跑了上述 AIME 性能对照及短输出并发验证；此前 Math500/HumanEval/MT-Bench
完整表仍是旧线程版结果，尚未用恢复后的引擎重测。
'''
(OUT/'PROCESS_RESTORED.md').write_text(report)
