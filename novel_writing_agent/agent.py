"""Main orchestration agent for NovelWritingAgent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .schema import Message

from .agents import IdeationSubAgent, WritingSubAgent
from .ideation import IdeationCoordinator
from .memory import MemoryOrchestrator
from .models import (
    AgentExecutionRequest,
    AgentExecutionResult,
    AgentRole,
    AgentTaskKind,
    CanonAsset,
    DeviationAction,
    NovelMode,
    NovelStage,
)
from .registry import AgentRegistry
from .state import NovelProjectState
from .storage import ProjectArtifactStore
from .ui import Colors, NovelUI


@dataclass(slots=True)
class StagePolicy:
    """Mode-specific stage control policy."""

    ideation_max_rounds: int
    drafting_review_frequency: int
    chapter_revision_max_rounds: int
    allow_direct_publish_after_revision: bool


class NovelMainAgent:
    """Main agent that keeps orchestration context clean.

    It decides which stage to run, which sub-agent to invoke, and what memory
    slices should be injected. Concrete LLM-backed execution can be layered on
    top of this framework incrementally.
    """

    def __init__(
        self,
        state: NovelProjectState,
        registry: AgentRegistry | None = None,
        memory: MemoryOrchestrator | None = None,
        llm_client: object | None = None,
        verbose: bool = False,
        log_path: str | Path | None = None,
        artifact_workspace_root: str | Path = "workspace/novel_projects",
    ) -> None:
        self.state = state
        self.registry = registry or AgentRegistry()
        self.memory = memory or MemoryOrchestrator(llm_client=llm_client)
        self.llm_client = llm_client
        self.verbose = verbose
        self.log_path = Path(log_path).resolve() if log_path else None
        self.ui = NovelUI()
        self.artifact_store = ProjectArtifactStore(
            project_id=state.project_id,
            workspace_root=artifact_workspace_root,
        )
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._register_default_agents()

    def policy(self) -> StagePolicy:
        """Return the active policy for the current mode."""
        if self.state.mode == NovelMode.SHORT:
            return StagePolicy(
                ideation_max_rounds=6,
                drafting_review_frequency=1,
                chapter_revision_max_rounds=3,
                allow_direct_publish_after_revision=True,
            )
        return StagePolicy(
            ideation_max_rounds=8,
            drafting_review_frequency=3,
            chapter_revision_max_rounds=4,
            allow_direct_publish_after_revision=False,
        )

    def clean_context(self) -> dict[str, object]:
        """Expose only the minimal state snapshot required for orchestration."""
        snapshot = self.state.snapshot_for_main_agent()
        snapshot["policy"] = asdict(self.policy())
        snapshot["artifact_paths"] = self.artifact_store.artifact_paths()
        return snapshot

    async def run_stage(self) -> list[str]:
        """Run one orchestration pass for the current stage."""
        self.log_progress(f"Starting stage: {self.state.stage.value}")
        if self.state.stage == NovelStage.IDEATION:
            result = await self._run_ideation_stage()
            self.persist_project_state()
            self.log_progress("Finished ideation stage.")
            return result
        if self.state.stage == NovelStage.DRAFTING:
            result = await self._run_drafting_stage()
            self.persist_project_state()
            self.log_progress("Finished drafting stage.")
            return result
        if self.state.stage == NovelStage.REVIEW:
            result = await self._run_review_stage()
            self.persist_project_state()
            self.log_progress("Finished review stage.")
            return result
        if self.state.stage == NovelStage.REVISION:
            result = await self._run_revision_stage()
            self.persist_project_state()
            self.log_progress("Finished revision stage.")
            return result
        if self.state.stage == NovelStage.PUBLISHING:
            result = await self._run_publishing_stage()
            self.persist_project_state()
            self.log_progress("Finished publishing stage.")
            return result
        return ["No further stage execution required."]

    async def _run_ideation_stage(self) -> list[str]:
        logs = await IdeationCoordinator(self).run()
        outputs = []
        for asset, log in logs.items():
            outputs.append(
                "\n".join(
                    [
                        f"asset={asset.value}",
                        f"proposal_rounds={len(log.proposals)}",
                        f"review_rounds={len(log.reviews)}",
                        f"frozen={self.state.canon_ready() if asset == log.asset else False}",
                    ]
                )
            )
        return outputs

    async def _run_drafting_stage(self) -> list[str]:
        next_chapter = len(self.state.draft.completed_chapters) + 1
        chapter_goal = self._chapter_goal(next_chapter)
        self.log_progress(f"Drafting chapter {next_chapter}.")
        self.state.start_chapter(next_chapter)
        request = AgentExecutionRequest(
            role=AgentRole.WRITER,
            task_kind=AgentTaskKind.GENERATE,
            objective=f"Write chapter {next_chapter} while staying aligned with canon.",
            context={
                "chapter_index": next_chapter,
                "main_snapshot": self.clean_context(),
                "chapter_goal": chapter_goal,
            },
            constraints=[
                "Follow the canon unless a substantially better idea emerges.",
                "Preserve voice consistency across all written chapters.",
            ],
            expected_output="A Chinese chapter draft that fulfills the chapter goal or a justified proposal to update canon.",
        )
        result = await self.dispatch_result(AgentRole.WRITER, request, chapter_index=next_chapter)
        if self._looks_like_invalid_generation(result.output):
            self.state.add_chapter_working_note(
                next_chapter,
                "draft generation returned invalid model output; chapter was not advanced",
            )
            notes_path = self.artifact_store.persist_chapter_working_notes(
                next_chapter,
                self.state.chapter_working_notes(next_chapter),
            )
            self.log_progress(
                f"Chapter {next_chapter} draft was invalid and has been left pending for resume.",
                label="Fallback",
                color=Colors.BRIGHT_YELLOW,
            )
            return [
                f"chapter={next_chapter}",
                f"goal={chapter_goal or '未找到章节目标'}",
                "draft_path=<invalid_output_skipped>",
                f"notes_path={notes_path}",
            ]
        if result.output.strip():
            self.state.add_chapter_working_note(
                next_chapter,
                f"draft round generated for chapter {next_chapter}",
            )
        integrated_draft = await self.integrate_chapter_draft(next_chapter, result.output)
        draft_path = self.artifact_store.persist_chapter_draft(next_chapter, integrated_draft, version="draft")
        notes_path = self.artifact_store.persist_chapter_working_notes(
            next_chapter,
            self.state.chapter_working_notes(next_chapter),
        )
        self.state.store_chapter_summary(
            next_chapter,
            result.metadata.get("summary", f"Chapter {next_chapter} draft completed."),
        )
        self.state.stage_notes.append(f"Drafted chapter {next_chapter}.")
        self.log_preview(f"chapter {next_chapter} draft preview", integrated_draft, max_lines=8)
        self.log_progress(f"Chapter {next_chapter} draft stored.")
        return [
            f"chapter={next_chapter}",
            f"goal={chapter_goal or '未找到章节目标'}",
            f"draft_path={draft_path}",
            f"notes_path={notes_path}",
        ]

    async def _run_review_stage(self) -> list[str]:
        outputs: list[str] = []
        current_chapter = self.state.latest_chapter_index()
        chapter_goal = self._chapter_goal(current_chapter)
        risk_profile = self.state.chapter_risk_profile(current_chapter)
        self.log_progress(f"Reviewing chapter {current_chapter}.")
        aggregated_feedback: list[str] = []
        canon_change_votes = 0
        for role in (
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
        ):
            request = AgentExecutionRequest(
                role=role,
                task_kind=AgentTaskKind.REVIEW,
                objective=f"Review chapter {current_chapter} against canon and project constraints.",
                context={
                    "chapter_index": current_chapter,
                    "main_snapshot": self.clean_context(),
                    "chapter_goal": chapter_goal,
                    "chapter_risk": risk_profile,
                },
                constraints=[
                    "Flag concrete issues, not vague critiques.",
                    "If deviation is beneficial, explain why canon should change.",
                    "If the chapter risk is medium or high, be extra strict about continuity and unresolved issues.",
                ],
                expected_output="Review findings in Chinese with clear actionable feedback.",
            )
            result = await self.dispatch_result(role, request, chapter_index=current_chapter)
            aggregated_feedback.extend(result.feedback)
            if result.should_update_canon:
                canon_change_votes += 1
            outputs.append(result.output)
        if not aggregated_feedback:
            aggregated_feedback = ["审核未返回有效反馈；保持当前版本并在下一轮继续复核。"]
        self.state.add_review_feedback(current_chapter, aggregated_feedback)
        self.state.add_chapter_working_note(
            current_chapter,
            "review feedback applied to chapter pipeline:\n- " + "\n- ".join(aggregated_feedback[:8]),
        )
        review_path = self.artifact_store.persist_chapter_review(current_chapter, aggregated_feedback)
        notes_path = self.artifact_store.persist_chapter_working_notes(
            current_chapter,
            self.state.chapter_working_notes(current_chapter),
        )
        action = self._resolve_deviation_action(canon_change_votes, aggregated_feedback)
        self.state.stage_notes.append(
            f"Review completed for chapter {current_chapter}; next action: {action.value}."
        )
        self.log_preview(
            f"chapter {current_chapter} review summary",
            "\n".join(f"- {item}" for item in aggregated_feedback),
            max_lines=8,
        )
        self.log_progress(
            f"Review completed for chapter {current_chapter}. Suggested action: {action.value}."
        )
        return [
            f"chapter={current_chapter}",
            f"goal={chapter_goal or '未找到章节目标'}",
            f"risk_level={risk_profile['level']}",
            f"risk_score={risk_profile['score']}",
            f"review_path={review_path}",
            f"notes_path={notes_path}",
            f"next_action={action.value}",
        ]

    async def _run_revision_stage(self) -> list[str]:
        current_chapter = self.state.latest_chapter_index()
        chapter_goal = self._chapter_goal(current_chapter)
        risk_profile = self.state.chapter_risk_profile(current_chapter)
        self.log_progress(f"Revising chapter {current_chapter}.")
        deviation_action = self._resolve_deviation_action(
            1 if any("canon" in feedback.lower() for feedback in self.state.draft.latest_feedback) else 0,
            self.state.draft.latest_feedback,
        )
        request = AgentExecutionRequest(
            role=AgentRole.WRITER,
            task_kind=AgentTaskKind.REVISE,
            objective=f"Revise chapter {current_chapter} using the latest review feedback.",
            context={
                "chapter_index": current_chapter,
                "main_snapshot": self.clean_context(),
                "chapter_goal": chapter_goal,
                "chapter_risk": risk_profile,
            },
            constraints=[
                "Prefer revising the draft over modifying canon unless there is a superior creative deviation.",
                "Keep revisions consistent with the existing authorial voice.",
                "If chapter risk is medium or high, resolve continuity and open-question issues before polishing style.",
            ],
            expected_output="A revised Chinese chapter draft with any canon-change recommendation called out explicitly.",
            metadata={"deviation_action": deviation_action.value},
        )
        result = await self.dispatch_result(AgentRole.WRITER, request, chapter_index=current_chapter)
        if self._looks_like_invalid_generation(result.output):
            self.state.add_chapter_working_note(
                current_chapter,
                "revision returned invalid model output; keeping previous official draft for resume",
            )
            notes_path = self.artifact_store.persist_chapter_working_notes(
                current_chapter,
                self.state.chapter_working_notes(current_chapter),
            )
            self.log_progress(
                f"Chapter {current_chapter} revision was invalid; previous official draft remains active.",
                label="Fallback",
                color=Colors.BRIGHT_YELLOW,
            )
            return [
                f"chapter={current_chapter}",
                f"goal={chapter_goal or '未找到章节目标'}",
                "revision_path=<invalid_output_skipped>",
                f"notes_path={notes_path}",
                f"deviation_action={deviation_action.value}",
                "chapter_decision=continue",
                "chapter_decision_reason=invalid model output; keep chapter open for resume",
            ]
        self.state.add_chapter_working_note(
            current_chapter,
            f"revision completed with action={deviation_action.value}",
        )
        integrated_revision = await self.integrate_chapter_draft(current_chapter, result.output)
        revision_path = self.artifact_store.persist_chapter_draft(
            current_chapter,
            integrated_revision,
            version="revision",
        )
        notes_path = self.artifact_store.persist_chapter_working_notes(
            current_chapter,
            self.state.chapter_working_notes(current_chapter),
        )
        convergence_decision, convergence_reason = await self._decide_chapter_convergence(
            current_chapter,
            chapter_goal,
            integrated_revision,
        )
        if deviation_action == DeviationAction.UPDATE_CANON:
            self._apply_canon_patch_from_revision(current_chapter, result)
            self.state.resolve_review_feedback(
                current_chapter,
                resolution_note=f"chapter {current_chapter} revision absorbed into canon",
                status="canonized",
            )
        else:
            self.state.resolve_review_feedback(
                current_chapter,
                resolution_note=f"chapter {current_chapter} revised draft absorbed reviewer feedback",
                status="absorbed",
            )
        if convergence_decision == "freeze":
            self.state.mark_revision_complete(current_chapter)
            self.state.stage_notes.append(f"Revision completed for chapter {current_chapter}.")
        else:
            self.state.mark_chapter_needs_another_revision(current_chapter)
            self.state.stage_notes.append(
                f"Chapter {current_chapter} requires another review-revision cycle."
            )
        self.log_preview(f"chapter {current_chapter} revision preview", integrated_revision, max_lines=8)
        self.log_progress(
            f"Revision completed for chapter {current_chapter}. Deviation action: {deviation_action.value}. Chapter decision: {convergence_decision}."
        )
        return [
            f"chapter={current_chapter}",
            f"goal={chapter_goal or '未找到章节目标'}",
            f"risk_level={risk_profile['level']}",
            f"risk_score={risk_profile['score']}",
            f"revision_path={revision_path}",
            f"notes_path={notes_path}",
            f"deviation_action={deviation_action.value}",
            f"chapter_decision={convergence_decision}",
            f"chapter_decision_reason={convergence_reason}",
        ]

    async def _run_publishing_stage(self) -> list[str]:
        self.log_progress("Packaging manuscript for publishing.")
        request = AgentExecutionRequest(
            role=AgentRole.PUBLISHING,
            task_kind=AgentTaskKind.PACKAGE,
            objective="Package the current manuscript for publishing channels such as Xiaohongshu.",
            context={"main_snapshot": self.clean_context()},
            constraints=[
                "Generate packaging materials without losing the story's voice.",
                "Treat publishing as a downstream packaging task, not a rewrite task.",
            ],
            expected_output="A publishing package with title ideas, intro copy, tags, and serialized hooks.",
        )
        result = await self.dispatch_result(AgentRole.PUBLISHING, request)
        return [result.output]

    async def dispatch_result(
        self,
        role: AgentRole,
        request: AgentExecutionRequest,
        chapter_index: int | None = None,
    ) -> AgentExecutionResult:
        agent = self.registry.get(role)
        memory_bundle = await self.memory.build_for_role(self.state, role, chapter_index=chapter_index)
        self.artifact_store.persist_memory_bundle(role.value, memory_bundle, chapter_index=chapter_index)
        task_desc = request.task_kind.value
        if request.target_asset:
            task_desc = f"{task_desc} {request.target_asset.value}"
        elif chapter_index is not None:
            task_desc = f"{task_desc} chapter {chapter_index}"
        self.log_progress(f"Dispatching {role.value}: {task_desc}")
        result = await agent.execute(request, memory_bundle)
        output_size = len(result.output.strip())
        self.log_progress(f"Completed {role.value}: {task_desc} ({output_size} chars)")
        return result

    def _resolve_deviation_action(
        self,
        canon_change_votes: int,
        feedback: list[str],
    ) -> DeviationAction:
        """Choose whether to revise prose or adapt canon."""
        beneficial_signal = any("stronger direction" in item.lower() or "update canon" in item.lower() for item in feedback)
        if canon_change_votes > 0 and beneficial_signal:
            return DeviationAction.UPDATE_CANON
        return DeviationAction.UPDATE_WRITER

    def _apply_canon_patch_from_revision(
        self,
        chapter_index: int,
        result: AgentExecutionResult,
    ) -> None:
        """Record canon evolution caused by a beneficial deviation."""
        patch = result.metadata.get("canon_patch")
        if isinstance(patch, (dict, list)):
            applied = self.state.apply_structured_canon_patch(patch, chapter_index=chapter_index)
            if applied:
                return
        if not patch:
            patch = f"Canon adjusted after chapter {chapter_index} revision."
        patch_text = str(patch).strip()
        if not patch_text:
            patch_text = f"Canon adjusted after chapter {chapter_index} revision."
        self.state.canon.world_rules.append(patch_text)
        self.state.log_canon_change(patch_text)

    async def _decide_chapter_convergence(
        self,
        chapter_index: int,
        chapter_goal: str,
        current_draft: str,
    ) -> tuple[str, str]:
        """Ask a dedicated agent whether the current chapter should continue iterating."""
        current_round = self.state.draft.revision_rounds_by_chapter.get(chapter_index, 0) + 1
        max_rounds = self.policy().chapter_revision_max_rounds
        request = AgentExecutionRequest(
            role=AgentRole.CHAPTER_CONVERGENCE,
            task_kind=AgentTaskKind.DECIDE,
            objective=f"Judge whether chapter {chapter_index} is ready to freeze or should continue revising.",
            context={
                "chapter_index": chapter_index,
                "chapter_goal": chapter_goal,
                "current_draft": current_draft,
                "latest_feedback": self.state.draft.latest_feedback,
                "current_round": current_round,
                "max_rounds": max_rounds,
                "main_snapshot": self.clean_context(),
            },
            constraints=[
                "Do not rewrite the chapter.",
                "If major issues remain, request another review-revision cycle.",
                "If the chapter is stable enough, freeze it and allow the story to move on.",
            ],
            expected_output="A decision to freeze, continue, or escalate with a short rationale.",
        )
        result = await self.dispatch_result(AgentRole.CHAPTER_CONVERGENCE, request, chapter_index=chapter_index)
        decision = str(result.metadata.get("decision", "continue")).lower()
        reason = str(result.metadata.get("reason", result.output.strip()))
        if current_round >= max_rounds and decision == "continue":
            decision = "freeze"
            reason = f"Reached chapter revision safety cap ({max_rounds}); freeze current best draft."
        self.state.add_chapter_working_note(
            chapter_index,
            f"chapter convergence round {current_round}: {decision}\n{reason}",
        )
        self.artifact_store.persist_chapter_decision(chapter_index, current_round, decision, reason)
        self.log_preview(f"chapter {chapter_index} convergence decision", reason, max_lines=4)
        return decision, reason

    def advance_stage(self) -> NovelStage:
        """Advance to the next stage using mode-aware defaults."""
        transitions = {
            NovelStage.IDEATION: NovelStage.DRAFTING,
            NovelStage.DRAFTING: NovelStage.REVIEW,
            NovelStage.REVIEW: NovelStage.REVISION,
            NovelStage.REVISION: NovelStage.PUBLISHING if self.policy().allow_direct_publish_after_revision else NovelStage.DRAFTING,
            NovelStage.PUBLISHING: NovelStage.COMPLETE,
        }
        self.state.stage = transitions.get(self.state.stage, NovelStage.COMPLETE)
        self.log_progress(f"Advanced stage to: {self.state.stage.value}")
        return self.state.stage

    def log_progress(self, message: str, label: str = "Progress", color: str = Colors.BRIGHT_CYAN) -> None:
        """Emit progress updates to terminal and optional log file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[NovelWritingAgent {timestamp}] {message}"
        if self.verbose:
            print(self.ui.event(label, message, color=color), flush=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def log_section(self, title: str) -> None:
        """Print a section header for the terminal and log."""
        section = self.ui.section(title)
        if self.verbose:
            print(section, flush=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n=== {title} ===\n")

    def log_preview(self, title: str, text: str, max_lines: int = 6) -> None:
        """Print and persist a short preview of intermediate outputs."""
        preview = self.ui.preview(title, text, max_lines=max_lines)
        if self.verbose:
            print(preview, flush=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"[Preview] {title}\n{text.strip()}\n\n")

    def persist_project_state(self) -> None:
        """Persist project state and memory for inspection."""
        self.artifact_store.persist_current_canon_bundle(self.state)
        self.artifact_store.persist_memory_overview(self.state)
        self.artifact_store.persist_state(self.state)
        self.artifact_store.persist_memory(self.state)
        self.artifact_store.persist_run_manifest(self.state)
        if self.state.stage == NovelStage.IDEATION or self.state.canon_ready():
            self.artifact_store.persist_ideation_result(self.state)

    @staticmethod
    def _looks_like_invalid_generation(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        invalid_markers = (
            "<minimax:tool_call>",
            "<invoke name=",
            "<parameter name=",
            "tool_call",
        )
        return any(marker in stripped for marker in invalid_markers)

    async def integrate_canon_asset(
        self,
        asset: CanonAsset,
        incoming_content: str,
        working_notes: list[str],
    ) -> str:
        """Integrate canon revisions into a full official version."""
        cleaned_incoming = incoming_content.strip()
        if not cleaned_incoming:
            return self.state.canon_text(asset)

        current_text = self.state.canon_text(asset).strip()
        if self.llm_client is None or not current_text:
            self.state.update_canon_asset(asset, cleaned_incoming)
            return self.state.canon_text(asset)

        prompt = "\n\n".join(
            [
                f"目标资产：{asset.value}",
                "任务：把当前正式版与本轮新修订整合成新的完整正式版。",
                "整合原则：保留旧版中仍然成立的内容，吸收新版里真正更优的修正。",
                "不要输出修订说明、变化总结、轮次说明。",
                "即便本轮只修改局部，也必须输出完整正式版。",
                "当前正式版：\n" + current_text,
                "本轮新修订：\n" + cleaned_incoming,
                "最近 working notes：\n" + ("\n".join(f"- {item}" for item in working_notes[-8:]) or "- 无"),
                "请只输出新的完整正式版。",
            ]
        )
        response = await self.llm_client.generate(
            [
                Message(
                    role="system",
                    content="你是小说系统内部的 canon integrator。你的职责是把多轮修订整合成一份新的完整正式版。",
                ),
                Message(role="user", content=prompt),
            ]
        )
        integrated = (response.content or "").strip() or cleaned_incoming
        self.state.update_canon_asset(asset, integrated)
        return self.state.canon_text(asset)

    async def integrate_chapter_draft(
        self,
        chapter_index: int,
        incoming_content: str,
    ) -> str:
        """Integrate a chapter revision into a full official chapter."""
        cleaned_incoming = incoming_content.strip()
        if not cleaned_incoming:
            return self.state.draft.chapter_drafts.get(chapter_index, "")

        current_text = self.state.draft.chapter_drafts.get(chapter_index, "").strip()
        if self.llm_client is None or not current_text:
            self.state.store_chapter_draft(chapter_index, cleaned_incoming)
            return self.state.draft.chapter_drafts.get(chapter_index, cleaned_incoming)

        working_notes = self.state.chapter_working_notes(chapter_index)
        prompt = "\n\n".join(
            [
                f"目标章节：第 {chapter_index} 章",
                "任务：把当前正式章节和本轮新稿/修订稿整合成新的完整正式章节。",
                "整合原则：保留旧版中仍然成立的内容，吸收新版里真正更优的修正。",
                "不要输出修订说明、变化总结、轮次说明。",
                "即便本轮只修改局部，也必须输出完整章节。",
                "当前正式章节：\n" + current_text,
                "本轮新稿/修订稿：\n" + cleaned_incoming,
                "最近 working notes：\n" + ("\n".join(f"- {item}" for item in working_notes[-8:]) or "- 无"),
                "请只输出新的完整正式章节。",
            ]
        )
        response = await self.llm_client.generate(
            [
                Message(
                    role="system",
                    content="你是小说系统内部的 chapter integrator。你的职责是把已有正式章节与本轮新稿整合成新的完整正式章节。",
                ),
                Message(role="user", content=prompt),
            ]
        )
        integrated = (response.content or "").strip() or cleaned_incoming
        self.state.store_chapter_draft(chapter_index, integrated)
        return self.state.draft.chapter_drafts.get(chapter_index, integrated)

    def _chapter_goal(self, chapter_index: int) -> str:
        if chapter_index <= 0 or chapter_index > len(self.state.canon.chapter_outline.chapters):
            return ""
        return self.state.canon.chapter_outline.chapters[chapter_index - 1].strip()

    def _register_default_agents(self) -> None:
        """Register placeholder agents so the framework is runnable from day one."""
        skills_dir = Path(__file__).parent / "skills"
        self.registry.register(
            AgentRole.PREMISE,
            IdeationSubAgent(
                AgentRole.PREMISE,
                "Premise ideation agent",
                skill_dir=skills_dir / "premise-agent",
                llm_client=self.llm_client,
            ),
            "ideation",
        )
        self.registry.register(
            AgentRole.OUTLINE,
            IdeationSubAgent(
                AgentRole.OUTLINE,
                "Outline ideation agent",
                skill_dir=skills_dir / "outline-agent",
                llm_client=self.llm_client,
            ),
            "ideation",
        )
        self.registry.register(
            AgentRole.CHARACTER,
            IdeationSubAgent(
                AgentRole.CHARACTER,
                "Character ideation agent",
                skill_dir=skills_dir / "character-agent",
                llm_client=self.llm_client,
            ),
            "ideation",
        )
        self.registry.register(
            AgentRole.CHAPTER_PLANNER,
            IdeationSubAgent(
                AgentRole.CHAPTER_PLANNER,
                "Chapter planning agent",
                skill_dir=skills_dir / "chapter-planner-agent",
                llm_client=self.llm_client,
            ),
            "ideation",
        )
        self.registry.register(
            AgentRole.CREATIVE_REVIEWER,
            IdeationSubAgent(
                AgentRole.CREATIVE_REVIEWER,
                "Creative review agent",
                skill_dir=skills_dir / "reviewer-agent",
                llm_client=self.llm_client,
            ),
            "ideation",
        )
        self.registry.register(
            AgentRole.CONVERGENCE,
            IdeationSubAgent(
                AgentRole.CONVERGENCE,
                "Convergence decision agent",
                skill_dir=skills_dir / "convergence-agent",
                llm_client=self.llm_client,
            ),
            "ideation",
        )
        self.registry.register(
            AgentRole.WRITER,
            WritingSubAgent(
                AgentRole.WRITER,
                "Primary writer agent",
                skill_dir=skills_dir / "writer-agent",
                llm_client=self.llm_client,
            ),
            "writing",
        )
        self.registry.register(
            AgentRole.CHARACTER_REVIEWER,
            WritingSubAgent(
                AgentRole.CHARACTER_REVIEWER,
                "Character review agent",
                skill_dir=skills_dir / "reviewer-agent",
                llm_client=self.llm_client,
            ),
            "writing",
        )
        self.registry.register(
            AgentRole.CONTINUITY_REVIEWER,
            WritingSubAgent(
                AgentRole.CONTINUITY_REVIEWER,
                "Continuity review agent",
                skill_dir=skills_dir / "reviewer-agent",
                llm_client=self.llm_client,
            ),
            "writing",
        )
        self.registry.register(
            AgentRole.STYLE_REVIEWER,
            WritingSubAgent(
                AgentRole.STYLE_REVIEWER,
                "Style review agent",
                skill_dir=skills_dir / "reviewer-agent",
                llm_client=self.llm_client,
            ),
            "writing",
        )
        self.registry.register(
            AgentRole.CHAPTER_CONVERGENCE,
            WritingSubAgent(
                AgentRole.CHAPTER_CONVERGENCE,
                "Chapter convergence agent",
                skill_dir=skills_dir / "convergence-agent",
                llm_client=self.llm_client,
            ),
            "writing",
        )
        self.registry.register(
            AgentRole.PUBLISHING,
            WritingSubAgent(
                AgentRole.PUBLISHING,
                "Publishing packaging agent",
                skill_dir=skills_dir / "reviewer-agent",
                llm_client=self.llm_client,
            ),
            "writing",
        )
