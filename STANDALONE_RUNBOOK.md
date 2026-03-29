# NovelWritingAgent 独立运行说明

这份文档基于当前仓库代码，说明如何准备环境、如何启动两个入口，以及产物会落到哪里。

## 1. 当前项目实际实现了什么

这个仓库不是单一写手脚本，而是一个分阶段的多 Agent 小说创作系统：

- `ideation` 阶段：生成并收敛 `story_outline`、`character_profiles`、`chapter_outline`
- `drafting` 阶段：按章节生成正文
- `review` 阶段：角色、连续性、风格三个 reviewer 并行审核
- `revision` 阶段：根据审核意见修订，并由 `chapter_convergence` 判断是否冻结当前章
- `memory`：持续维护 narrative、review、event 等长期记忆，并按任务裁切上下文
- `resume`：按 `project_id` 恢复项目，扫描坏章节并从断点继续

## 2. 环境准备

推荐 Python 3.10 及以上。

### 2.1 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2.2 安装依赖

```bash
python -m pip install -r requirements.txt
```

## 3. 配置文件

真实模型模式需要：

```bash
cp config/config-example.yaml config/config.yaml
```

然后修改 `config/config.yaml`，至少填这些字段：

- `api_key`
- `api_base`
- `model`
- `provider`

`run_skeleton.py` 不需要真实模型配置。

## 4. 两个入口各自做什么

### 4.1 `examples/run_skeleton.py`

命令：

```bash
python -m examples.run_skeleton
```

作用：

- 不依赖真实 LLM
- 会创建一个短篇项目
- 跑一轮 `ideation`
- 把中间状态、日志和产物写入 `workspace/novel_projects/<project_id>/`

适合用来检查：

- 框架是否能启动
- 主 Agent 与子 Agent 的调度是否正常
- `canon` 是否能正常落盘
- UI 与日志路径是否正常

### 4.2 `examples/run_with_llm.py`

命令：

```bash
python -m examples.run_with_llm
```

作用：

- 加载 `config/config.yaml`
- 初始化真实 LLM client
- 先跑 `ideation`
- 再循环跑 `drafting -> review -> revision`
- 每个阶段都会持久化 state、memory、章节文件和恢复报告

## 5. 运行时参数

### 5.1 新建项目

```bash
python -m examples.run_with_llm \
  --title "未来都市孤独感" \
  --brief "写一个关于未来都市中孤独感的短篇小说。"
```

### 5.2 恢复续写

```bash
python -m examples.run_with_llm --resume-project-id <project_id>
```

恢复时会先做这些事：

- 扫描已有章节文件
- 标出缺失或无效的章节产物
- 生成 `recovery_report.md`
- 从需要补跑的位置继续

## 6. 当前代码里的默认行为

如果你不传 `--brief`，`run_with_llm.py` 会使用内置中文题材作为默认主题。

当前代码里默认模式是：

- `NovelMode.SHORT`

也就是说，默认走的是“短篇模式”，不是长篇模式。

## 7. 输出路径

所有产物默认落在：

```text
workspace/novel_projects/<project_id>/
```

主要文件：

- `canon/current/story_outline.md`
- `canon/current/character_profiles.md`
- `canon/current/chapter_outline.md`
- `canon/history/*.md`
- `reviews/*.md`
- `outputs/ideation_result.md`
- `outputs/chapters/chapter_001_draft.md`
- `outputs/chapters/chapter_001_review.md`
- `outputs/chapters/chapter_001_revision.md`
- `outputs/chapters/chapter_001_working_notes.md`
- `outputs/memory/*.json`
- `memory/memory_snapshot.json`
- `state/state_snapshot.json`

## 8. 如何验证安装是否完整

安装完依赖后，可以先跑测试：

```bash
python -m pytest -q tests/test_novel_writing.py
```

然后跑骨架：

```bash
python -m examples.run_skeleton
```

如果这两步都能通过，再去跑真实 LLM 模式。
