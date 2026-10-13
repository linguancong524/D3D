# 发布与分支策略（Release & Branching）

本仓库基于 SGLang v0.5.17（上游 sgl-project/sglang tag v0.5.17, commit b6a09f38）建立，
用于 NVIDIA H800 上的 Qwen 2B/4B/9B 部署性能优化（prefill/decode 效率、dflash/dspark 加速实验）。

## 分支模型

| 分支 | 用途 |
|------|------|
| `main` | 日常开发：基准测试脚本、性能优化实验 |
| `release` | 稳定版本：优化通过 H800 回归基准后从 `main` 合并，并在其上打 tag 发布 |

## 发布流程

1. 在 `main` 上完成优化，并在远程 H800 服务器上跑回归基准（对比 v0.5.17 基线）；
2. 将验证过的改动同步到 `release` 并打 tag：
   
   ```bash
   git checkout release
   git merge --no-ff main
   git tag -a v0.5.17.1 -m "H800 perf: <改动摘要>"
   git push origin release --tags
   ```
3. 远程 H800 服务器拉取稳定版部署：
   
   ```bash
   git checkout release && git pull origin release
   pip install -e "python[all,huggingface]"   # 安装方式见 README
   ```

## 基线记录

- 首次提交 `891978f`：SGLang v0.5.17 原始代码（已去除上游 git 历史，由本仓库统一管理）；
- 基线已更新为 SGLang v0.5.18（上游 tag v0.5.18, commit ff4c6e64），含上游提交的 `rust/Cargo.lock`；
- 性能数据、测试配置与结论统一按用途归档在 `docs/perf/` 的子目录，分类与入口见 `docs/perf/README.md`。
