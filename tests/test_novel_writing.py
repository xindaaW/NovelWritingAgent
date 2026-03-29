"""Tests for the NovelWritingAgent framework skeleton."""

import json
from pathlib import Path

import pytest

from novel_writing_agent import NovelMainAgent, NovelMode, NovelProjectState, NovelStage
from novel_writing_agent.agents import WritingSubAgent
from novel_writing_agent.ideation import IdeationCoordinator
from novel_writing_agent.memory import MemoryOrchestrator, WorkingMemoryBundle
from novel_writing_agent.models import AgentExecutionRequest, AgentRole, AgentTaskKind, CanonAsset
from novel_writing_agent.storage import ProjectArtifactStore
from novel_writing_agent.tools import RetrieveMemoryTool
from novel_writing_agent.retry import RetryConfig as RuntimeRetryConfig, coerce_retry_config


def test_short_story_policy_sets_a_safety_cap_for_ideation(tmp_path: Path):
    """Short story mode should use a safety cap, not a fixed workflow loop."""
    state = NovelProjectState(
        project_id="short-1",
        title="Short",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    policy = agent.policy()

    assert policy.ideation_max_rounds == 6
    assert policy.allow_direct_publish_after_revision is True


def test_long_novel_policy_prefers_longer_drafting_loop(tmp_path: Path):
    """Long novel mode should revisit drafting after revision."""
    state = NovelProjectState(
        project_id="long-1",
        title="Long",
        user_brief="brief",
        mode=NovelMode.LONG,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    state.stage = NovelStage.REVISION
    next_stage = agent.advance_stage()

    assert next_stage == NovelStage.DRAFTING


def test_main_agent_snapshot_stays_compact(tmp_path: Path):
    """The main agent should operate on a compact state snapshot."""
    state = NovelProjectState(
        project_id="short-2",
        title="Compact",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    snapshot = agent.clean_context()

    assert snapshot["mode"] == NovelMode.SHORT.value
    assert snapshot["stage"] == NovelStage.IDEATION.value
    assert "policy" in snapshot


@pytest.mark.asyncio
async def test_ideation_stage_converges_and_freezes_canon(tmp_path: Path):
    """Ideation should iterate canon assets and freeze them for drafting."""
    state = NovelProjectState(
        project_id="short-3",
        title="Canon",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    outputs = await agent.run_stage()

    assert len(outputs) == 3
    assert state.canon.story_outline.frozen is True
    assert state.canon.character_profiles.frozen is True
    assert state.canon.chapter_outline.frozen is True
    assert state.canon_text(CanonAsset.STORY_OUTLINE)
    assert state.canon_text(CanonAsset.CHARACTER_PROFILES)
    assert any("converged" in note.lower() for note in state.stage_notes)
    project_root = tmp_path / state.project_id
    assert (project_root / "canon/current/story_outline.md").exists()
    assert (project_root / "canon/current/character_profiles.md").exists()
    assert (project_root / "canon/current/chapter_outline.md").exists()
    assert (project_root / "outputs/ideation_result.md").exists()


@pytest.mark.asyncio
async def test_drafting_review_revision_updates_memory_and_completes_chapter(tmp_path: Path):
    """The drafting loop should persist draft, feedback, revision, and completion state."""
    state = NovelProjectState(
        project_id="short-4",
        title="Loop",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    await agent.run_stage()
    state.stage = NovelStage.DRAFTING
    drafting_outputs = await agent.run_stage()
    assert drafting_outputs
    assert 1 in state.draft.chapter_drafts
    assert state.draft.chapter_summaries[1]
    assert state.draft.chapter_status[1] == "drafted"
    assert (tmp_path / state.project_id / "outputs/chapters/chapter_001_draft.md").exists()
    assert (tmp_path / state.project_id / "outputs/chapters/chapter_001_working_notes.md").exists()
    assert (tmp_path / state.project_id / "outputs/memory/writer_chapter_001.json").exists()

    state.stage = NovelStage.REVIEW
    review_outputs = await agent.run_stage()
    assert len(review_outputs) == 5
    assert state.draft.latest_feedback
    assert state.draft.review_feedback_by_chapter[1]
    assert (tmp_path / state.project_id / "outputs/chapters/chapter_001_review.md").exists()
    assert (tmp_path / state.project_id / "outputs/memory/character_reviewer_chapter_001.json").exists()

    state.stage = NovelStage.REVISION
    revision_outputs = await agent.run_stage()
    assert revision_outputs
    assert 1 in state.draft.completed_chapters
    assert state.draft.chapter_status[1] == "revised"
    assert state.memory.review_memory
    assert state.memory.chapter_memory[1]
    assert (tmp_path / state.project_id / "outputs/chapters/chapter_001_revision.md").exists()
    assert (tmp_path / state.project_id / "outputs/memory/memory_overview.md").exists()
    assert len(revision_outputs) == 7


@pytest.mark.asyncio
async def test_ideation_history_and_reviews_are_persisted(tmp_path: Path):
    """Ideation should write canon history plus review/decision records."""
    state = NovelProjectState(
        project_id="history-demo",
        title="History Demo",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    await agent.run_stage()

    project_root = tmp_path / state.project_id
    assert any((project_root / "canon/history").iterdir())
    assert any((project_root / "reviews").iterdir())


@pytest.mark.asyncio
async def test_revision_persists_chapter_convergence_decision(tmp_path: Path):
    """Revision should write a chapter-level convergence decision file."""
    state = NovelProjectState(
        project_id="chapter-convergence",
        title="Chapter Convergence",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)

    await agent.run_stage()
    state.stage = NovelStage.DRAFTING
    await agent.run_stage()
    state.stage = NovelStage.REVIEW
    await agent.run_stage()
    state.stage = NovelStage.REVISION
    await agent.run_stage()

    review_dir = tmp_path / state.project_id / "reviews"
    assert any(path.name.startswith("chapter_001_round_01_decision") for path in review_dir.iterdir())


class FakeLLM:
    """Minimal fake LLM for structured-output parsing tests."""

    def __init__(self, content: str):
        self.content = content

    async def generate(self, messages):
        class Response:
            def __init__(self, content: str):
                self.content = content
                self.thinking = None

        return Response(self.content)


@pytest.mark.asyncio
async def test_llm_backed_agent_uses_freeform_output_with_lightweight_parsing():
    """LLM-backed sub-agents should remain skill-driven and tolerate freeform output."""
    fake_response = """Chapter 1 prose

- Keep the secondary character sharper.
- Suggest tightening the city-introduction paragraph.
This is a stronger direction and we may want to update canon in the next round.
"""
    agent = WritingSubAgent(
        role=AgentRole.WRITER,
        system_prompt="Writer system prompt",
        llm_client=FakeLLM(fake_response),
    )

    result = await agent.execute(
        request=AgentExecutionRequest(
            role=agent.role,
            task_kind=AgentTaskKind.GENERATE,
            objective="Write chapter 1",
        ),
        working_memory=WorkingMemoryBundle(),
    )

    assert result.output.startswith("Chapter 1 prose")
    assert "Keep the secondary character sharper." in result.feedback
    assert result.should_update_canon is True
    assert result.metadata["summary"].startswith("Chapter 1 prose")


def test_ideation_normalizer_keeps_clean_canon(tmp_path: Path):
    """Revision chatter should not leak into canon memory or final canon files."""
    state = NovelProjectState(
        project_id="clean-canon",
        title="Clean Canon",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)
    coordinator = IdeationCoordinator(agent)

    raw_outline = """# Chapter Outline

## Summary of Changes from Round 1
1. 强化了主角动机
2. 提前了关键反转

## Chapter 1: 起点
主角在雨夜进入旧城。

## Chapter 2: 代价
他第一次失去同伴。

## Round 2 Status
Chapter outline revised.
"""
    cleaned = coordinator._normalize_canon_text(CanonAsset.CHAPTER_OUTLINE, raw_outline)

    assert "Summary of Changes" not in cleaned
    assert "Round 2 Status" not in cleaned
    assert "## Chapter 1: 起点" in cleaned
    assert "## Chapter 2: 代价" in cleaned


def test_working_notes_are_separate_from_final_canon():
    """Process notes should stay outside the final canon text."""
    state = NovelProjectState(
        project_id="notes",
        title="Notes",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    state.update_canon_asset(CanonAsset.STORY_OUTLINE, "# 正式大纲\n\n主角踏上旅程。")
    state.add_asset_working_note(CanonAsset.STORY_OUTLINE, "round 2 feedback: 强化主角动机")

    assert "强化主角动机" in state.asset_working_notes(CanonAsset.STORY_OUTLINE)[0]
    assert "强化主角动机" not in state.canon_text(CanonAsset.STORY_OUTLINE)


@pytest.mark.asyncio
async def test_writer_memory_bundle_includes_relationships_and_scene_cast():
    """Writer memory should include relationship anchors and a scene cast for the current chapter."""
    state = NovelProjectState(
        project_id="relations",
        title="Relations",
        user_brief="写家族复仇故事",
        mode=NovelMode.SHORT,
    )
    state.update_canon_asset(
        CanonAsset.CHARACTER_PROFILES,
        """# 人物小传

## 林渊
主角。林家长房独子。
与林崇是父子，与林崇山是叔侄。

## 林崇
林渊的父亲，林家长房家主。
与林崇山是兄弟，始终护着林渊。

## 林崇山
林渊的二叔，林家二房掌权者。
与林渊对立，与林崇是兄弟反目。
""",
    )
    state.update_canon_asset(
        CanonAsset.CHAPTER_OUTLINE,
        """第1章：林渊在宗堂与父亲林崇、二叔林崇山正面冲突。""",
    )
    state.store_chapter_draft(
        1,
        "林渊抬头看着林崇山，余光里却看见父亲林崇的手在发抖。",
    )

    bundle = await MemoryOrchestrator().build_for_role(state, AgentRole.WRITER, chapter_index=1)

    assert any("林渊" in item and "父子" in item for item in bundle.relation_context)
    assert any("Scene Cast" in item and "林崇山" in item for item in bundle.scene_cast_context)


@pytest.mark.asyncio
async def test_writer_memory_bundle_includes_prior_narrative_history():
    """Writer retrieval should bring forward prior chapter summaries for long-form continuity."""
    state = NovelProjectState(
        project_id="narrative-memory",
        title="Narrative Memory",
        user_brief="写一个长期成长故事",
        mode=NovelMode.LONG,
    )
    state.update_canon_asset(CanonAsset.CHAPTER_OUTLINE, "第1章：家族覆灭。\n第2章：流亡求生。\n第3章：旧敌重逢。")
    state.store_chapter_summary(1, "主角在家族覆灭之夜失去一切。")
    state.store_chapter_summary(2, "主角在流亡途中第一次怀疑二叔并发现旧令牌。")

    bundle = await MemoryOrchestrator().build_for_role(state, AgentRole.WRITER, chapter_index=3)

    assert any("主角在家族覆灭之夜失去一切" in item for item in bundle.narrative_context)
    assert any("第一次怀疑二叔并发现旧令牌" in item for item in bundle.narrative_context)


@pytest.mark.asyncio
async def test_writer_memory_bundle_exposes_progressive_retrieval_layers():
    """Writer should receive a layered retrieval index for progressive memory disclosure."""
    state = NovelProjectState(
        project_id="progressive-memory",
        title="Progressive Memory",
        user_brief="写一个长期成长故事",
        mode=NovelMode.LONG,
    )
    state.update_canon_asset(CanonAsset.CHARACTER_PROFILES, "# 人物小传\n\n## 林渊\n主角，性格隐忍。")
    state.update_canon_asset(CanonAsset.CHAPTER_OUTLINE, "第1章：家族覆灭。\n第2章：流亡求生。")
    state.store_chapter_summary(1, "主角在家族覆灭之夜失去一切。")

    bundle = await MemoryOrchestrator().build_for_role(state, AgentRole.WRITER, chapter_index=2)

    assert "immediate" in bundle.retrieval_index
    assert "contextual" in bundle.retrieval_index
    assert "deep" in bundle.retrieval_index
    tool = RetrieveMemoryTool(bundle.retrieval_index)
    result = await tool.execute(level="contextual", memory_type="narrative", focus="家族覆灭", limit=3)
    assert result.success is True
    assert "家族覆灭" in result.content
    detail = await tool.execute(level="contextual", memory_type="narrative", focus="家族覆灭", limit=1, reveal="detail")
    assert "正文：" in detail.content


@pytest.mark.asyncio
async def test_reviewer_memory_bundle_exposes_progressive_retrieval_layers():
    """Reviewer should also receive the retrieval tool index for progressive checking."""
    state = NovelProjectState(
        project_id="reviewer-progressive-memory",
        title="Reviewer Progressive Memory",
        user_brief="写一个长期成长故事",
        mode=NovelMode.LONG,
    )
    state.update_canon_asset(CanonAsset.CHARACTER_PROFILES, "# 人物小传\n\n## 林渊\n主角，性格隐忍。")
    state.update_canon_asset(CanonAsset.CHAPTER_OUTLINE, "第1章：家族覆灭。\n第2章：流亡求生。")
    state.store_chapter_summary(1, "林渊在家族覆灭之夜失去一切。")
    state.add_review_feedback(2, ["人物情绪承接不足"])

    bundle = await MemoryOrchestrator().build_for_role(state, AgentRole.CONTINUITY_REVIEWER, chapter_index=2)

    assert "contextual" in bundle.retrieval_index
    tool = RetrieveMemoryTool(bundle.retrieval_index)
    result = await tool.execute(level="immediate", memory_type="review", focus="情绪", limit=2)
    assert result.success is True
    assert "open_question" in result.content
    assert "人物情绪承接不足" in result.content


def test_structured_event_memory_is_recorded_for_summary_and_review():
    """Long-term memory should retain typed events for retrieval and continuation."""
    state = NovelProjectState(
        project_id="event-memory",
        title="Event Memory",
        user_brief="写一个长期成长故事",
        mode=NovelMode.LONG,
    )

    state.store_chapter_summary(1, "林渊在家族覆灭之夜失去一切，并发现父亲留下的旧令牌。")
    state.add_review_feedback(1, ["旧令牌的伏笔和父亲遗言衔接还不够清晰"])

    assert any(event.event_type == "plot_event" and "旧令牌" in event.summary for event in state.memory.event_memory)
    assert any(
        event.event_type == "open_question" and "衔接还不够清晰" in event.summary
        for event in state.memory.event_memory
    )
    assert any(event.event_type == "foreshadowing" for event in state.memory.event_memory)
    assert any(event.event_type == "reveal" for event in state.memory.event_memory)


def test_summary_derives_relationship_and_character_state_events():
    """Chapter summaries should also derive relationship and character-state changes."""
    state = NovelProjectState(
        project_id="derived-events",
        title="Derived Events",
        user_brief="写一个长期成长故事",
        mode=NovelMode.LONG,
    )

    state.store_chapter_summary(2, "林渊开始怀疑二叔，并与旧友彻底决裂。")

    assert any(event.event_type == "relationship_change" for event in state.memory.event_memory)
    assert any(event.event_type == "character_state_change" for event in state.memory.event_memory)


def test_review_memory_is_stateful_after_resolution():
    """Resolved review issues should remain auditable but leave the active working set."""
    state = NovelProjectState(
        project_id="review-state",
        title="Review State",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    state.add_review_feedback(1, ["人物动机不够清晰", "二叔称谓偶有混乱"])
    assert len(state.active_review_items(chapter_index=1)) == 2

    state.resolve_review_feedback(1, "第1章修订已吸收反馈", status="absorbed")

    assert state.active_review_items(chapter_index=1) == []
    assert len(state.recent_review_items()) == 2
    assert all(item.status == "absorbed" for item in state.recent_review_items())


def test_project_state_and_memory_are_persisted_and_loadable(tmp_path: Path):
    """Long-term memory and state should survive process boundaries."""
    state = NovelProjectState(
        project_id="resume-demo",
        title="Resume Demo",
        user_brief="写一个复仇成长长篇",
        mode=NovelMode.LONG,
    )
    state.update_canon_asset(CanonAsset.STORY_OUTLINE, "# 故事大纲\n\n## 第一幕\n少年家破人亡。")
    state.update_canon_asset(CanonAsset.CHARACTER_PROFILES, "# 人物小传\n\n## 林渊\n主角，性格隐忍。")
    state.update_canon_asset(CanonAsset.CHAPTER_OUTLINE, "第1章：家族覆灭。\n第2章：踏上流亡。")
    state.memory.outline_memory.append("第一幕：家族覆灭")
    state.memory.character_memory.append("林渊：隐忍成长")
    state.store_chapter_draft(1, "林渊在血夜中醒来。")
    state.store_chapter_summary(1, "第一章写了家族覆灭。")
    state.add_review_feedback(1, ["第一章节奏偏快"])

    store = ProjectArtifactStore(project_id=state.project_id, workspace_root=tmp_path)
    store.persist_state(state)
    store.persist_memory(state)
    store.persist_run_manifest(state)

    assert (tmp_path / state.project_id / "state" / "state_snapshot.json").exists()
    assert (tmp_path / state.project_id / "memory" / "memory_snapshot.json").exists()
    assert (tmp_path / state.project_id / "state" / "run_manifest.json").exists()

    restored = ProjectArtifactStore.load_project_state(state.project_id, workspace_root=tmp_path)

    assert restored.project_id == state.project_id
    assert restored.mode == NovelMode.LONG
    assert restored.canon.story_outline.current == state.canon.story_outline.current
    assert restored.memory.outline_memory[-1] == "第一幕：家族覆灭"
    assert restored.memory.chapter_memory[1]
    assert restored.recent_review_items()[0].text == "第一章节奏偏快"
    assert any(event.event_type == "plot_event" for event in restored.memory.event_memory)
    assert any(event.event_type == "open_question" for event in restored.memory.event_memory)


def test_old_project_can_be_reconstructed_without_state_snapshot(tmp_path: Path):
    """Older projects should still be resumable from canon and chapter artifacts."""
    project_root = tmp_path / "legacy-project"
    (project_root / "canon/current").mkdir(parents=True)
    (project_root / "outputs/chapters").mkdir(parents=True)
    (project_root / "outputs").mkdir(exist_ok=True)
    (project_root / "canon/current/story_outline.md").write_text("# 故事大纲\n\n## 第一幕\n家族覆灭。", encoding="utf-8")
    (project_root / "canon/current/character_profiles.md").write_text("# 人物小传\n\n## 林渊\n主角。", encoding="utf-8")
    (project_root / "canon/current/chapter_outline.md").write_text("第1章：家族覆灭。\n第2章：踏上流亡。", encoding="utf-8")
    (project_root / "outputs/ideation_result.md").write_text(
        "# Ideation Result: Legacy Demo\n\nproject_id: legacy-project\nmode: long_novel\nstage: revision\n\n## User Brief\n写一个复仇成长故事。\n",
        encoding="utf-8",
    )
    (project_root / "outputs/chapters/chapter_001_revision.md").write_text("林渊在血夜中醒来。", encoding="utf-8")
    (project_root / "outputs/chapters/chapter_001_working_notes.md").write_text(
        "# Chapter 1 Working Notes\n\n- revision completed\n",
        encoding="utf-8",
    )

    restored = ProjectArtifactStore.load_project_state("legacy-project", workspace_root=tmp_path)

    assert restored.project_id == "legacy-project"
    assert restored.user_brief == "写一个复仇成长故事。"
    assert restored.canon.story_outline.current
    assert 1 in restored.draft.completed_chapters
    assert restored.draft.chapter_drafts[1] == "林渊在血夜中醒来。"


def test_run_manifest_supports_recovery_when_state_snapshot_is_missing(tmp_path: Path):
    """Run manifests should preserve lightweight resume metadata even without a state snapshot."""
    project_root = tmp_path / "manifest-only"
    (project_root / "state").mkdir(parents=True)
    (project_root / "canon/current").mkdir(parents=True)
    (project_root / "outputs/chapters").mkdir(parents=True)
    (project_root / "canon/current/story_outline.md").write_text("# 故事大纲\n\n## 第一幕\n家族覆灭。", encoding="utf-8")
    (project_root / "canon/current/character_profiles.md").write_text("# 人物小传\n\n## 林渊\n主角。", encoding="utf-8")
    (project_root / "canon/current/chapter_outline.md").write_text("第1章：家族覆灭。", encoding="utf-8")
    (project_root / "outputs/chapters/chapter_001_draft.md").write_text("林渊在血夜中醒来。", encoding="utf-8")
    (project_root / "state/run_manifest.json").write_text(
        """
{
  "project_id": "manifest-only",
  "title": "Manifest Recovery",
  "user_brief": "从 manifest 恢复项目。",
  "mode": "short_story",
  "stage": "review"
}
""".strip(),
        encoding="utf-8",
    )

    restored = ProjectArtifactStore.load_project_state("manifest-only", workspace_root=tmp_path)

    assert restored.title == "Manifest Recovery"
    assert restored.user_brief == "从 manifest 恢复项目。"
    assert restored.mode == NovelMode.SHORT
    assert restored.stage == NovelStage.REVIEW
    assert restored.draft.chapter_status[1] == "drafted"


def test_recovery_scan_reports_invalid_and_missing_chapter_files(tmp_path: Path):
    """Recovery scan should identify invalid chapters that need reruns."""
    store = ProjectArtifactStore(project_id="scan-demo", workspace_root=tmp_path)
    store.persist_chapter_draft(1, "正常正文", version="draft")
    store.persist_chapter_review(1, ["有反馈"])
    store.persist_chapter_draft(1, "修订后正文", version="revision")
    store.persist_chapter_draft(2, "<minimax:tool_call>", version="draft")

    report = store.scan_chapter_artifacts()

    assert 1 in report["valid_completed"]
    assert 2 in report["rerun_chapters"]
    assert any(item["chapter"] == 2 and "invalid_draft" in item["issues"] for item in report["issues"])


def test_legacy_recovery_does_not_mark_invalid_revision_as_completed(tmp_path: Path):
    """Invalid legacy revision outputs should not be treated as completed chapters."""
    project_root = tmp_path / "legacy-invalid"
    (project_root / "canon/current").mkdir(parents=True)
    (project_root / "outputs/chapters").mkdir(parents=True)
    (project_root / "outputs").mkdir(exist_ok=True)
    (project_root / "canon/current/story_outline.md").write_text("# 故事大纲\n\n## 第一幕\n家族覆灭。", encoding="utf-8")
    (project_root / "canon/current/character_profiles.md").write_text("# 人物小传\n\n## 林渊\n主角。", encoding="utf-8")
    (project_root / "canon/current/chapter_outline.md").write_text("第1章：家族覆灭。", encoding="utf-8")
    (project_root / "outputs/ideation_result.md").write_text(
        "# Ideation Result: Legacy Invalid\n\nproject_id: legacy-invalid\nmode: long_novel\nstage: revision\n\n## User Brief\n写一个复仇成长故事。\n",
        encoding="utf-8",
    )
    (project_root / "outputs/chapters/chapter_001_revision.md").write_text("<minimax:tool_call>", encoding="utf-8")

    restored = ProjectArtifactStore.load_project_state("legacy-invalid", workspace_root=tmp_path)

    assert restored.draft.completed_chapters == []


@pytest.mark.asyncio
async def test_invalid_writer_output_does_not_overwrite_chapter(tmp_path: Path):
    """Tool-call style invalid outputs should leave the chapter open for resume."""
    state = NovelProjectState(
        project_id="invalid-output",
        title="Invalid Output",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(state, artifact_workspace_root=tmp_path)
    state.update_canon_asset(CanonAsset.CHAPTER_OUTLINE, "第1章：起点。")
    state.freeze_asset(CanonAsset.STORY_OUTLINE)
    state.freeze_asset(CanonAsset.CHARACTER_PROFILES)
    state.freeze_asset(CanonAsset.CHAPTER_OUTLINE)

    class FakeBadResultAgent:
        async def execute(self, request, working_memory):
            from novel_writing_agent.models import AgentExecutionResult

            return AgentExecutionResult(
                role=request.role,
                task_kind=request.task_kind,
                output="<minimax:tool_call>\n<invoke name=\"bash\">",
            )

    agent.registry._agents[AgentRole.WRITER] = FakeBadResultAgent()
    state.stage = NovelStage.DRAFTING
    outputs = await agent.run_stage()

    assert "draft_path=<invalid_output_skipped>" in outputs
    assert 1 not in state.draft.chapter_drafts


def test_retry_config_from_settings_is_coerced_to_runtime_shape():
    """Config-like retry objects should be converted to the runtime retry config."""
    class ConfigLikeRetry:
        enabled = True
        max_retries = 5
        initial_delay = 0.5
        max_delay = 10.0
        exponential_base = 2.0

    runtime_retry = coerce_retry_config(ConfigLikeRetry())

    assert isinstance(runtime_retry, RuntimeRetryConfig)
    assert runtime_retry.max_retries == 5
    assert runtime_retry.retryable_exceptions == (Exception,)


def test_story_outline_partial_update_preserves_existing_sections():
    """Updating one outline section should not drop previously good sections."""
    state = NovelProjectState(
        project_id="outline-sections",
        title="Outline Sections",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    state.update_canon_asset(
        CanonAsset.STORY_OUTLINE,
        """# 故事大纲

## 一、核心设定
主角在家族覆灭后被迫流亡。

## 二、第一幕
主角在谷底忍辱求生。
""",
    )
    state.update_canon_asset(
        CanonAsset.STORY_OUTLINE,
        """## 二、第一幕
主角在谷底忍辱求生，并第一次发现自身血脉异变。
""",
    )

    final_outline = state.canon_text(CanonAsset.STORY_OUTLINE)
    assert "## 一、核心设定" in final_outline
    assert "家族覆灭后被迫流亡" in final_outline
    assert "血脉异变" in final_outline


def test_character_profiles_partial_update_preserves_other_characters():
    """Updating one character card should keep the rest of the character bible intact."""
    state = NovelProjectState(
        project_id="character-sections",
        title="Character Sections",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    state.update_canon_asset(
        CanonAsset.CHARACTER_PROFILES,
        """# 人物小传

## 林渊
主角，性格隐忍。

## 苏晚
女主，聪明冷静。
""",
    )
    state.update_canon_asset(
        CanonAsset.CHARACTER_PROFILES,
        """## 林渊
主角，性格隐忍，善于在羞辱中隐藏锋芒。
""",
    )

    final_profiles = state.canon_text(CanonAsset.CHARACTER_PROFILES)
    assert "## 苏晚" in final_profiles
    assert "聪明冷静" in final_profiles
    assert "隐藏锋芒" in final_profiles


def test_chapter_partial_update_preserves_existing_scenes_and_keeps_notes_separate():
    """Chapter revision should preserve existing scene sections while keeping notes outside final prose."""
    state = NovelProjectState(
        project_id="chapter-sections",
        title="Chapter Sections",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    state.store_chapter_draft(
        1,
        """## 场景一
林渊在雨夜逃亡。

## 场景二
他第一次看见苏晚。
""",
    )
    state.add_chapter_working_note(1, "上一轮指出场景二情绪不够。")
    state.store_chapter_draft(
        1,
        """## 场景二
他第一次看见苏晚时，忽然生出一种近乎荒谬的安宁。

## 修订说明
强化了情绪波动。
""",
    )

    final_draft = state.draft.chapter_drafts[1]
    assert "## 场景一" in final_draft
    assert "林渊在雨夜逃亡" in final_draft
    assert "近乎荒谬的安宁" in final_draft
    assert "修订说明" not in final_draft
    assert "上一轮指出场景二情绪不够。" in state.chapter_working_notes(1)[0]


@pytest.mark.asyncio
async def test_llm_backed_reviewer_prefers_structured_json_appendix():
    """JSON appendices should override brittle freeform heuristics when present."""
    fake_response = """整体方向正确。

```json
{
  "feedback": [
    "补强主角在本章开头的即时目标。",
    "让对手的反制更具体。"
  ],
  "should_update_canon": true,
  "canon_update_reason": "当前章节暴露出更强的对立线，可以同步更新 chapter outline。"
}
```"""
    agent = WritingSubAgent(
        role=AgentRole.CONTINUITY_REVIEWER,
        system_prompt="Reviewer system prompt",
        llm_client=FakeLLM(fake_response),
    )

    result = await agent.execute(
        request=AgentExecutionRequest(
            role=agent.role,
            task_kind=AgentTaskKind.REVIEW,
            objective="Review chapter 2",
        ),
        working_memory=WorkingMemoryBundle(),
    )

    assert result.feedback == [
        "补强主角在本章开头的即时目标。",
        "让对手的反制更具体。",
    ]
    assert result.should_update_canon is True
    assert "更强的对立线" in result.canon_update_reason


@pytest.mark.asyncio
async def test_llm_backed_convergence_prefers_structured_json_appendix():
    """Convergence metadata should be parsed from JSON when the model provides it."""
    fake_response = """建议冻结当前版本。

```json
{
  "decision": "freeze",
  "reason": "本章主冲突已经闭合，剩余问题只属于微调层面。"
}
```"""
    agent = WritingSubAgent(
        role=AgentRole.CHAPTER_CONVERGENCE,
        system_prompt="Convergence system prompt",
        llm_client=FakeLLM(fake_response),
    )

    result = await agent.execute(
        request=AgentExecutionRequest(
            role=agent.role,
            task_kind=AgentTaskKind.DECIDE,
            objective="Decide whether chapter 2 should freeze.",
            context={"current_round": 1, "max_rounds": 3},
        ),
        working_memory=WorkingMemoryBundle(),
    )

    assert result.metadata["decision"] == "freeze"
    assert "主冲突已经闭合" in result.metadata["reason"]


@pytest.mark.asyncio
async def test_writer_memory_bundle_applies_prompt_budget_caps():
    """Extreme context growth should be trimmed before it reaches the writer prompt."""
    state = NovelProjectState(
        project_id="budgeted-memory",
        title="Budgeted Memory",
        user_brief="写一个很长的复仇故事",
        mode=NovelMode.LONG,
    )
    state.update_canon_asset(
        CanonAsset.STORY_OUTLINE,
        "# 故事大纲\n\n" + ("主角在废墟中忍辱求生，并反复回想家族覆灭之夜。 " * 240),
    )
    state.update_canon_asset(
        CanonAsset.CHARACTER_PROFILES,
        "# 人物小传\n\n## 林渊\n" + ("林渊与父亲、宿敌、盟友之间的关系复杂。 " * 240),
    )
    state.update_canon_asset(
        CanonAsset.CHAPTER_OUTLINE,
        "第1章：家族覆灭。\n第2章：流亡求生。\n第3章：旧敌重逢。",
    )
    state.store_chapter_summary(1, "主角在家族覆灭之夜失去一切。" * 120)
    state.store_chapter_summary(2, "主角在流亡途中第一次意识到宿敌其实一直在监视自己。" * 120)

    bundle = await MemoryOrchestrator().build_for_role(state, AgentRole.WRITER, chapter_index=3)
    canon_budget = MemoryOrchestrator.SECTION_BUDGETS["canon_context"]
    narrative_budget = MemoryOrchestrator.SECTION_BUDGETS["narrative_context"]

    assert sum(len(item) for item in bundle.canon_context) <= canon_budget.max_total_chars
    assert sum(len(item) for item in bundle.narrative_context) <= narrative_budget.max_total_chars
    assert any(item.endswith("...[truncated]") for item in bundle.canon_context + bundle.narrative_context)


def test_artifact_store_atomic_writes_leave_no_temp_files(tmp_path: Path):
    """Artifact persistence should not leave temporary files behind after a successful write."""
    state = NovelProjectState(
        project_id="atomic-store",
        title="Atomic Store",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    store = ProjectArtifactStore(project_id=state.project_id, workspace_root=tmp_path)

    store.persist_state(state)
    store.persist_memory(state)
    store.persist_run_manifest(state)
    store.persist_chapter_draft(1, "正文")

    assert not list(tmp_path.rglob("*.tmp"))


def test_run_manifest_records_compact_operational_snapshot(tmp_path: Path):
    """Run manifests should expose enough state to inspect a run without loading full snapshots."""
    state = NovelProjectState(
        project_id="manifest-snapshot",
        title="Manifest Snapshot",
        user_brief="brief",
        mode=NovelMode.SHORT,
    )
    state.stage = NovelStage.REVISION
    state.stage_notes.extend(["完成 ideation", "完成 draft", "进入修订"])
    state.draft.chapter_status[1] = "revised"
    state.draft.completed_chapters.append(1)
    state.memory.outline_memory.append("第一幕")
    store = ProjectArtifactStore(project_id=state.project_id, workspace_root=tmp_path)

    manifest_path = store.persist_run_manifest(state)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["stage"] == "revision"
    assert manifest["draft"]["completed_chapters"] == [1]
    assert manifest["memory"]["outline_items"] == 1
    assert manifest["artifact_paths"]["state"].endswith("/state")
    assert manifest["recent_stage_notes"][-1] == "进入修订"
