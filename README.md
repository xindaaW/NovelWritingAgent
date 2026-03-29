# NovelWritingAgent

`NovelWritingAgent` 是一个独立的多 Agent 小说创作框架，当前已经可以跑通以下流程：

- 前期 `canon` 生成与收敛
- 章节写作、审核、修订、章节收敛
- long-term memory 持久化与渐进式 `retrieve_memory`
- prompt 级 memory budget 控制与风险感知审核
- 本地语义检索 fallback 与 long-horizon memory 压缩
- 结构化 `canon_patch`，支持章节修订反向更新 canon
- 项目状态落盘与按 `project_id` 恢复续写
- 无模型骨架调试和接入真实 LLM 的两种运行模式

## 目录结构

- `novel_writing_agent/`: 核心包
- `examples/`: 运行入口
- `tests/`: 测试
- `config/`: 配置模板
- `workspace/novel_projects/`: 默认输出目录

## 环境准备

推荐 Python 3.10 及以上。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 配置

真实模型运行前需要准备配置文件：

```bash
cp config/config-example.yaml config/config.yaml
```

然后至少填写这些字段：

- `api_key`
- `api_base`
- `model`
- `provider`

`run_skeleton.py` 不依赖真实模型，不需要 `config.yaml`。

## 如何运行

### 1. 骨架模式

只验证框架、状态流转和产物落盘：

```bash
python -m examples.run_skeleton
```

### 2. 真实 LLM 模式

使用 `config/config.yaml` 中的模型配置：

```bash
python -m examples.run_with_llm
```

新建项目时可以传入题材：

```bash
python -m examples.run_with_llm \
  --title "未来都市孤独感" \
  --brief "写一个关于未来都市中孤独感的短篇小说。"
```

恢复续写：

```bash
python -m examples.run_with_llm --resume-project-id <project_id>
```

## 输出位置

所有运行产物默认写到：

```text
workspace/novel_projects/<project_id>/
```

常见文件包括：

- `outputs/ideation_result.md`
- `outputs/chapters/chapter_001_draft.md`
- `outputs/chapters/chapter_001_review.md`
- `outputs/chapters/chapter_001_revision.md`
- `state/state_snapshot.json`
- `state/run_manifest.json`
- `memory/memory_snapshot.json`

更详细的运行说明见 `STANDALONE_RUNBOOK.md`。后续优化路线见 `OPTIMIZATION_CHECKLIST.md`。
