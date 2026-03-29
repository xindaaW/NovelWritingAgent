"""Ideation-stage convergence loop for NovelWritingAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .models import AgentExecutionRequest, AgentRole, AgentTaskKind, CanonAsset
from .ui import Colors


@dataclass(slots=True)
class AssetIterationLog:
    """Tracks one asset's ideation cycle."""

    asset: CanonAsset
    proposals: list[str] = field(default_factory=list)
    reviews: list[list[str]] = field(default_factory=list)


class IdeationCoordinator:
    """Runs iterative proposal-review-revision loops for canon assets."""

    ASSET_GENERATORS = {
        CanonAsset.STORY_OUTLINE: AgentRole.OUTLINE,
        CanonAsset.CHARACTER_PROFILES: AgentRole.CHARACTER,
        CanonAsset.CHAPTER_OUTLINE: AgentRole.CHAPTER_PLANNER,
    }

    def __init__(self, main_agent: "NovelMainAgent") -> None:
        self.main_agent = main_agent

    async def run(self) -> dict[CanonAsset, AssetIterationLog]:
        """Converge all canon assets for the ideation stage."""
        logs: dict[CanonAsset, AssetIterationLog] = {}
        max_rounds = self.main_agent.policy().ideation_max_rounds
        for asset in (
            CanonAsset.STORY_OUTLINE,
            CanonAsset.CHARACTER_PROFILES,
            CanonAsset.CHAPTER_OUTLINE,
        ):
            self.main_agent.log_section(f"Ideation: {asset.value}")
            self.main_agent.log_progress(
                f"Starting ideation for {asset.value}. Convergence agent will decide when to stop; safety cap is {max_rounds} rounds.",
                label="Stage",
                color=Colors.BRIGHT_YELLOW,
            )
            logs[asset] = await self._iterate_asset(asset, max_rounds)
            self.main_agent.state.freeze_asset(asset)
            self.main_agent.log_progress(f"Frozen canon asset: {asset.value}")

        if self.main_agent.state.canon_ready():
            self.main_agent.state.stage_notes.append("Ideation canon assets converged and frozen.")
            self.main_agent.log_progress("All canon assets converged and frozen.")
        return logs

    async def _iterate_asset(self, asset: CanonAsset, max_rounds: int) -> AssetIterationLog:
        log = AssetIterationLog(asset=asset)
        generator_role = self.ASSET_GENERATORS[asset]
        current_text = self.main_agent.state.canon_text(asset)
        round_index = 1

        while round_index <= max_rounds:
            self.main_agent.log_progress(f"{asset.value}: proposal round {round_index}", label="Round")
            proposal_request = AgentExecutionRequest(
                role=generator_role,
                task_kind=AgentTaskKind.REVISE if current_text else AgentTaskKind.GENERATE,
                objective=f"Produce round {round_index} for {asset.value}.",
                context={
                    "main_snapshot": self.main_agent.clean_context(),
                    "current_draft": current_text,
                    "known_feedback": self._feedback_for_asset(asset),
                    "working_notes": self._working_notes_for_asset(asset),
                },
                constraints=[
                    "Treat this as a canon asset that should become stable before drafting begins.",
                    "Preserve strong ideas from previous rounds while fixing concrete weaknesses.",
                ],
                expected_output=f"A stronger {asset.value} draft.",
                target_asset=asset,
            )
            proposal = await self.main_agent.dispatch_result(generator_role, proposal_request)
            raw_proposed_text = proposal.output.strip() or current_text
            proposed_text = self._normalize_canon_text(asset, raw_proposed_text)
            if raw_proposed_text and raw_proposed_text != proposed_text:
                self.main_agent.state.add_asset_working_note(
                    asset,
                    f"round {round_index} proposal notes preserved outside canon",
                )
            log.proposals.append(raw_proposed_text)
            integrated_text = await self.main_agent.integrate_canon_asset(
                asset,
                incoming_content=proposed_text,
                working_notes=self._working_notes_for_asset(asset),
            )
            self._update_planning_memory(asset, integrated_text)
            self.main_agent.artifact_store.persist_canon_asset(asset, integrated_text, round_index)
            self.main_agent.persist_project_state()
            current_text = integrated_text
            self.main_agent.log_preview(f"{asset.value} draft preview", integrated_text)

            self.main_agent.log_progress(f"{asset.value}: creative review round {round_index}", label="Review")
            review_request = AgentExecutionRequest(
                role=AgentRole.CREATIVE_REVIEWER,
                task_kind=AgentTaskKind.REVIEW,
                objective=f"Critique the current {asset.value} draft and point out concrete weaknesses.",
                context={
                    "main_snapshot": self.main_agent.clean_context(),
                    "candidate": proposed_text,
                    "round_index": round_index,
                    "working_notes": self._working_notes_for_asset(asset),
                },
                constraints=[
                    "Feedback must be concrete, not generic.",
                    "Only suggest canon changes when they materially improve the concept.",
                ],
                expected_output="A focused review with actionable bullets.",
                target_asset=asset,
            )
            review = await self.main_agent.dispatch_result(AgentRole.CREATIVE_REVIEWER, review_request)
            feedback = review.feedback or [review.output.strip()]
            self.main_agent.state.add_asset_feedback(asset, feedback)
            self.main_agent.state.add_asset_working_note(
                asset,
                f"round {round_index} feedback:\n- " + "\n- ".join(feedback[:8]),
            )
            self.main_agent.artifact_store.persist_review(asset, feedback, round_index)
            self.main_agent.persist_project_state()
            log.reviews.append(feedback)
            self.main_agent.log_progress(
                f"{asset.value}: collected {len(feedback)} review note(s) in round {round_index}"
            )
            self.main_agent.log_preview(
                f"{asset.value} review notes",
                "\n".join(f"- {item}" for item in feedback),
                max_lines=5,
            )

            decision = await self._decide_convergence(
                asset=asset,
                current_text=current_text,
                feedback=feedback,
                round_index=round_index,
                max_rounds=max_rounds,
            )
            if decision == "freeze":
                self.main_agent.log_progress(
                    f"{asset.value}: convergence-agent decided to freeze after round {round_index}.",
                    label="Decision",
                    color=Colors.BRIGHT_GREEN,
                )
                break
            if decision == "escalate":
                self.main_agent.log_progress(
                    f"{asset.value}: convergence-agent requested escalation; freezing current best version for now.",
                    label="Decision",
                    color=Colors.BRIGHT_YELLOW,
                )
                break
            self.main_agent.log_progress(
                f"{asset.value}: convergence-agent requested another iteration.",
                label="Decision",
                color=Colors.BRIGHT_MAGENTA,
            )
            round_index += 1

        return log

    async def _decide_convergence(
        self,
        asset: CanonAsset,
        current_text: str,
        feedback: list[str],
        round_index: int,
        max_rounds: int,
    ) -> str:
        """Ask the convergence agent whether this asset should keep iterating."""
        decision_request = AgentExecutionRequest(
            role=AgentRole.CONVERGENCE,
            task_kind=AgentTaskKind.DECIDE,
            objective=f"Decide whether {asset.value} should continue iterating or be frozen.",
            context={
                "asset": asset.value,
                "round_index": round_index,
                "max_rounds": max_rounds,
                "current_draft": current_text,
                "latest_feedback": feedback,
                "working_notes": self._working_notes_for_asset(asset),
            },
            constraints=[
                "Do not rewrite the asset.",
                "Decide whether to continue, freeze, or escalate based on quality and diminishing returns.",
            ],
            expected_output="A convergence decision with rationale.",
            target_asset=asset,
        )
        result = await self.main_agent.dispatch_result(AgentRole.CONVERGENCE, decision_request)
        decision = str(result.metadata.get("decision", "continue")).lower()
        rationale = str(result.metadata.get("reason", result.output.strip()))
        self.main_agent.state.add_asset_working_note(
            asset,
            f"round {round_index} convergence: {decision}\n{rationale}",
        )
        self.main_agent.artifact_store.persist_decision(asset, decision, rationale, round_index)
        self.main_agent.persist_project_state()
        self.main_agent.log_preview(f"{asset.value} convergence decision", rationale, max_lines=4)
        return decision

    def _feedback_for_asset(self, asset: CanonAsset) -> list[str]:
        if asset == CanonAsset.STORY_OUTLINE:
            return list(self.main_agent.state.canon.story_outline.review_feedback[-5:])
        if asset == CanonAsset.CHARACTER_PROFILES:
            return list(self.main_agent.state.canon.character_profiles.review_feedback[-5:])
        if asset == CanonAsset.CHAPTER_OUTLINE:
            return list(self.main_agent.state.canon.chapter_outline.review_feedback[-5:])
        return []

    def _working_notes_for_asset(self, asset: CanonAsset) -> list[str]:
        return self.main_agent.state.asset_working_notes(asset)[-6:]

    def _update_planning_memory(self, asset: CanonAsset, content: str) -> None:
        """Refresh ideation-stage memory for the canon-writing loop."""
        summary = content.strip()[:500]
        if asset == CanonAsset.STORY_OUTLINE:
            self.main_agent.state.memory.outline_memory.append(summary)
        elif asset == CanonAsset.CHARACTER_PROFILES:
            self.main_agent.state.memory.character_memory.append(summary)
        elif asset == CanonAsset.CHAPTER_OUTLINE:
            self.main_agent.state.memory.world_memory.append(summary)

    def _normalize_canon_text(self, asset: CanonAsset, content: str) -> str:
        """Strip revision chatter so canon stays clean for downstream memory."""
        text = content.strip()
        if not text:
            return text

        lines = text.splitlines()
        cleaned_lines: list[str] = []
        skip_meta_section = False

        meta_section_patterns = (
            r"^##\s+summary of changes",
            r"^##\s+round \d+\s+status",
            r"^##\s+revision notes",
            r"^##\s+changes from round",
            r"^##\s+本轮修改",
            r"^##\s+修订说明",
            r"^##\s+修改说明",
            r"^##\s+本轮状态",
        )
        real_content_resume_patterns = (
            r"^##\s+\d+",
            r"^##\s+chapter\b",
            r"^##\s+第.+章",
            r"^##\s+[^\n]+$",
        )

        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower()

            if any(re.match(pattern, lowered) for pattern in meta_section_patterns):
                skip_meta_section = True
                continue

            if skip_meta_section:
                if any(re.match(pattern, lowered) for pattern in real_content_resume_patterns):
                    skip_meta_section = False
                else:
                    continue

            if self._is_revision_chatter_line(stripped):
                continue

            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()
        return cleaned_text or text

    @staticmethod
    def _is_revision_chatter_line(line: str) -> bool:
        lowered = line.lower()
        chatter_signals = (
            "summary of changes",
            "round status",
            "recommended next steps",
            "changes from round",
            "本轮修改",
            "修订说明",
            "修改说明",
            "本轮状态",
            "改动如下",
            "以下修改",
            "针对上一轮反馈",
        )
        return any(signal in lowered for signal in chatter_signals)
