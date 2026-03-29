"""Agent abstractions for the NovelWritingAgent framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
from textwrap import dedent

from .schema import Message

from .memory import WorkingMemoryBundle
from .models import AgentExecutionRequest, AgentExecutionResult, AgentRole
from .prompting import load_skill_prompt
from .tools import RetrieveMemoryTool


class BaseNovelSubAgent(ABC):
    """Base class for all novel-writing sub-agents."""

    group = "generic"

    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        skill_dir: str | Path | None = None,
        llm_client: object | None = None,
    ):
        self.role = role
        self.skill_dir = Path(skill_dir) if skill_dir else None
        self.llm_client = llm_client
        self.prompt_bundle = load_skill_prompt(self.skill_dir, system_prompt)
        self.system_prompt = self.prompt_bundle.system_prompt

    @abstractmethod
    async def execute(
        self,
        request: AgentExecutionRequest,
        working_memory: WorkingMemoryBundle,
    ) -> AgentExecutionResult:
        """Execute a task with role-specific context."""


class StubSubAgent(BaseNovelSubAgent):
    """Safe placeholder implementation while the framework is under construction."""

    async def execute(
        self,
        request: AgentExecutionRequest,
        working_memory: WorkingMemoryBundle,
    ) -> AgentExecutionResult:
        if self.llm_client is not None:
            messages = self._build_messages(request, working_memory)
            tools = self._tools_for_request(request, working_memory)
            if tools:
                try:
                    response = await self.llm_client.generate(messages, tools=tools)
                except TypeError:
                    response = await self.llm_client.generate(messages)
                    tools = []
            else:
                response = await self.llm_client.generate(messages)
            if tools and response.tool_calls:
                for _ in range(4):
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = tool_call.function.arguments
                        tool = next((item for item in tools if item.name == tool_name), None)
                        if tool is None:
                            tool_content = f"Unknown tool: {tool_name}"
                        else:
                            result = await tool.execute(**arguments)
                            tool_content = result.content if result.success else result.error or "Tool failed"
                        messages.append(
                            Message(
                                role="assistant",
                                content=response.content.strip() or "",
                                thinking=response.thinking,
                                tool_calls=response.tool_calls,
                            )
                        )
                        messages.append(
                            Message(
                                role="tool",
                                content=tool_content,
                                tool_call_id=tool_call.id,
                                name=tool_name,
                            )
                        )
                    response = await self.llm_client.generate(messages, tools=tools)
                    if not response.tool_calls:
                        break
            return AgentExecutionResult(
                role=self.role,
                task_kind=request.task_kind,
                output=response.content.strip(),
                structured_output={
                    "thinking": response.thinking,
                    "target_asset": request.target_asset.value if request.target_asset else None,
                    "raw_response": response.content,
                },
                feedback=self._extract_feedback(response.content),
                confidence=0.7,
                should_update_canon=self._should_update_canon(response.content),
                canon_update_reason=self._canon_update_reason(response.content),
                metadata=self._extract_metadata(request, response.content),
            )

        summary = [
            f"role={self.role.value}",
            f"task_kind={request.task_kind.value}",
            f"objective={request.objective}",
            f"target_asset={request.target_asset.value if request.target_asset else 'none'}",
            f"canon_items={len(working_memory.canon_context)}",
            f"narrative_items={len(working_memory.narrative_context)}",
            f"review_items={len(working_memory.review_context)}",
        ]
        if request.task_kind.value == "review":
            feedback = [
                f"Review focus for {request.target_asset.value if request.target_asset else 'task'}",
                "Tighten motivation and make conflict escalation more explicit.",
            ]
            should_update_canon = False
            metadata = {"deviation_signal": "minor"}
            if self.role == AgentRole.CONTINUITY_REVIEWER:
                feedback = [
                    "Chapter drift detected: the scene introduces a new direction that is not in the current chapter plan.",
                    "Decide whether to fold this stronger direction back into canon or rewrite the chapter to match the planned line.",
                ]
                should_update_canon = True
                metadata = {"deviation_signal": "beneficial"}
            return AgentExecutionResult(
                role=self.role,
                task_kind=request.task_kind,
                output="\n".join(summary),
                structured_output={
                    "objective": request.objective,
                    "expected_output": request.expected_output,
                },
                feedback=feedback,
                confidence=0.3,
                should_update_canon=should_update_canon,
                canon_update_reason=feedback[0],
                metadata=metadata,
            )
        if self.role in {AgentRole.CONVERGENCE, AgentRole.CHAPTER_CONVERGENCE} and request.task_kind.value == "decide":
            round_index = int(request.context.get("round_index", 1))
            if self.role == AgentRole.CHAPTER_CONVERGENCE:
                round_index = int(request.context.get("current_round", 1))
            known_feedback = request.context.get("latest_feedback", [])
            decision = "continue"
            rationale = "Major issues remain and another revision round is justified."
            if self.role == AgentRole.CHAPTER_CONVERGENCE:
                decision = "freeze"
                rationale = "The chapter is coherent enough to freeze for now and move to the next chapter."
                if round_index >= int(request.context.get("max_rounds", 3)):
                    decision = "freeze"
                    rationale = "Reached the chapter revision ceiling; freeze the current best version."
                elif any(
                    keyword in " ".join(known_feedback).lower()
                    for keyword in ("major issue", "still broken", "hard contradiction", "structural failure")
                ):
                    decision = "continue"
                    rationale = "Structural issues still remain, so the chapter needs another review-revision cycle."
                return AgentExecutionResult(
                    role=self.role,
                    task_kind=request.task_kind,
                    output=f"Decision: {decision}\nReason: {rationale}",
                    feedback=[rationale],
                    confidence=0.45,
                    metadata={"decision": decision, "reason": rationale},
                )
            if round_index >= 2 and len(known_feedback) <= 2:
                decision = "freeze"
                rationale = "The remaining issues look minor and the asset is stable enough to freeze."
            if round_index >= int(request.context.get("max_rounds", 4)):
                decision = "freeze"
                rationale = "Reached the iteration ceiling; freeze the best current version and move on."
            return AgentExecutionResult(
                role=self.role,
                task_kind=request.task_kind,
                output=f"Decision: {decision}\nReason: {rationale}",
                feedback=[rationale],
                confidence=0.45,
                metadata={"decision": decision, "reason": rationale},
            )
        if request.target_asset is not None and request.task_kind.value in {"generate", "revise"}:
            mock_content = self._mock_ideation_content(request)
            return AgentExecutionResult(
                role=self.role,
                task_kind=request.task_kind,
                output=mock_content,
                structured_output={
                    "objective": request.objective,
                    "expected_output": request.expected_output,
                },
                confidence=0.42,
                metadata={"summary": mock_content.split("\n\n", 1)[0][:240]},
            )
        if request.task_kind.value == "revise":
            metadata = {"revision_applied": True}
            if request.metadata.get("deviation_action") == "update_canon":
                metadata["canon_patch"] = f"Canon updated based on chapter {request.context.get('chapter_index', 'unknown')} creative deviation."
            return AgentExecutionResult(
                role=self.role,
                task_kind=request.task_kind,
                output="\n".join(summary + ["revision=applied"]),
                structured_output={
                    "objective": request.objective,
                    "expected_output": request.expected_output,
                },
                confidence=0.35,
                metadata=metadata,
            )
        if self.role == AgentRole.WRITER and request.task_kind.value == "generate":
            metadata = {
                "summary": f"Chapter {request.context.get('chapter_index', 1)} advances the core conflict and preserves the current voice.",
            }
            return AgentExecutionResult(
                role=self.role,
                task_kind=request.task_kind,
                output="\n".join(summary + ["draft=chapter text placeholder"]),
                structured_output={
                    "objective": request.objective,
                    "expected_output": request.expected_output,
                },
                confidence=0.4,
                metadata=metadata,
            )
        return AgentExecutionResult(
            role=self.role,
            task_kind=request.task_kind,
            output="\n".join(summary),
            structured_output={
                "objective": request.objective,
                "expected_output": request.expected_output,
            },
            confidence=0.25,
        )

    def _mock_ideation_content(self, request: AgentExecutionRequest) -> str:
        """Generate readable placeholder canon content for skeleton runs."""
        brief = str(request.context.get("main_snapshot", {}).get("user_brief", "")).strip()
        user_hint = request.context.get("current_draft") or brief or request.objective
        if request.target_asset.value == "story_outline":
            return dedent(
                f"""
                # Story Outline

                In a near-future metropolis, an isolated protagonist discovers that a commercial memory service can edit away loneliness for a price. What begins as a tempting escape becomes a moral trap when the protagonist realizes the city is quietly reshaping citizens by standardizing what they remember and what they forget.

                The first act establishes the protagonist's emotional emptiness, the seductive promise of memory editing, and the first small success that makes the technology feel irresistible.

                The middle of the story escalates through consequences: relationships become smoother but less real, grief loses texture, and the protagonist starts noticing cracks between lived experience and curated recall.

                The ending forces a choice between a painless artificial self and a painful but authentic identity, landing on a bittersweet recovery of agency.
                """
            ).strip()
        if request.target_asset.value == "character_profiles":
            return dedent(
                """
                # Character Profiles

                ## Protagonist
                Quiet, observant, and emotionally self-protective. Speaks in restrained, literal sentences and often avoids naming feelings directly. Wants relief from loneliness but fears becoming someone unreal.

                ## Memory Clinic Sales Lead
                Warm, elegant, and persuasive. Speaks with polished reassurance and frames every moral compromise as self-care. Embodies the system's seductive logic.

                ## Old Friend
                Messy, direct, and emotionally transparent. Speaks fast, interrupts, and says uncomfortable truths. Represents an older, more painful but more authentic version of connection.
                """
            ).strip()
        if request.target_asset.value == "chapter_outline":
            return dedent(
                """
                Chapter 1: Introduce the protagonist's urban isolation and first encounter with the memory service.
                Chapter 2: Show the emotional improvement after the first memory edit and why the service feels useful.
                Chapter 3: Reveal subtle distortions in relationships and memory continuity.
                Chapter 4: Force the protagonist to compare artificial relief with real unresolved pain.
                Chapter 5: Drive the final choice and its emotional aftermath.
                """
            ).strip()
        return user_hint if isinstance(user_hint, str) else str(user_hint)

    def _build_messages(
        self,
        request: AgentExecutionRequest,
        working_memory: WorkingMemoryBundle,
    ) -> list[Message]:
        """Build a minimal prompt for LLM-backed sub-agents."""
        user_sections = [
            f"Role: {self.role.value}",
            f"Task kind: {request.task_kind.value}",
            f"Objective: {request.objective}",
            "Output language: 简体中文",
        ]
        main_snapshot = request.context.get("main_snapshot", {})
        if isinstance(main_snapshot, dict):
            user_brief = str(main_snapshot.get("user_brief", "")).strip()
            if user_brief:
                user_sections.append(f"User brief: {user_brief}")
            project_title = str(main_snapshot.get("title", "")).strip()
            if project_title:
                user_sections.append(f"Project title: {project_title}")
        if request.target_asset:
            user_sections.append(f"Target asset: {request.target_asset.value}")
        if request.constraints:
            user_sections.append("Constraints:\n- " + "\n- ".join(request.constraints))
        if request.expected_output:
            user_sections.append(f"Expected output: {request.expected_output}")
        format_hint = self._response_format_hint(request)
        if format_hint:
            user_sections.append(format_hint)
        if self.role == AgentRole.WRITER and working_memory.retrieval_index:
            user_sections.append(
                "Memory strategy:\n你先基于当前最小上下文写作；如果需要补人物、伏笔、前情、审核问题，请主动调用 retrieve_memory 工具逐步取回。默认先取 summary；只有摘要不够时再取 detail。不要一开始假设自己已经看过全部记忆。"
            )
        if self.role in {
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
        } and working_memory.retrieval_index:
            user_sections.append(
                "Memory strategy:\n你先基于当前审核目标使用最小上下文；如果需要核对前情、人物关系、伏笔或未解决问题，请主动调用 retrieve_memory。默认先看 summary，不够再看 detail。"
            )
        if request.context:
            user_sections.append(f"Request context: {request.context}")
        if working_memory.task_context:
            user_sections.append("Task context:\n" + "\n\n".join(working_memory.task_context))
        if working_memory.canon_context:
            user_sections.append("Canon context:\n" + "\n\n".join(working_memory.canon_context))
        if working_memory.relation_context:
            user_sections.append("Relationship context:\n" + "\n\n".join(working_memory.relation_context))
        if working_memory.scene_cast_context:
            user_sections.append("Scene cast context:\n" + "\n\n".join(working_memory.scene_cast_context))
        if working_memory.planning_context:
            user_sections.append("Planning memory:\n" + "\n\n".join(working_memory.planning_context))
        if working_memory.narrative_context:
            user_sections.append("Narrative context:\n" + "\n\n".join(working_memory.narrative_context))
        if working_memory.review_context:
            user_sections.append("Review context:\n" + "\n\n".join(working_memory.review_context))

        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content="\n\n".join(user_sections)),
        ]

    def _tools_for_request(
        self,
        request: AgentExecutionRequest,
        working_memory: WorkingMemoryBundle,
    ) -> list:
        if self.role == AgentRole.WRITER and request.task_kind.value in {"generate", "revise"}:
            if working_memory.retrieval_index:
                return [RetrieveMemoryTool(working_memory.retrieval_index)]
        if self.role in {
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
        } and request.task_kind.value == "review":
            if working_memory.retrieval_index:
                return [RetrieveMemoryTool(working_memory.retrieval_index)]
        return []

    def _response_format_hint(self, request: AgentExecutionRequest) -> str:
        """Encourage structured appendices without breaking freeform skill-driven writing."""
        if request.task_kind.value == "review":
            return (
                "Response format:\n"
                "先给自然语言审核意见；最后追加一个 ```json``` 代码块，包含 keys: "
                "`feedback` (string list), `should_update_canon` (bool), `canon_update_reason` (string)."
            )
        if request.task_kind.value == "decide":
            return (
                "Response format:\n"
                "先给简短结论；最后追加一个 ```json``` 代码块，包含 keys: "
                "`decision` (`freeze` | `continue` | `escalate`), `reason` (string)."
            )
        if self.role == AgentRole.WRITER and request.task_kind.value in {"generate", "revise"}:
            return (
                "Response format:\n"
                "正文保持自然语言输出；如果方便，请在末尾追加一个 ```json``` 代码块，"
                "至少包含 `summary`，如果建议改 canon 再补 `should_update_canon` 和 `canon_update_reason`。"
            )
        return ""

    def _extract_json_payload(self, text: str) -> dict[str, object] | None:
        """Parse an optional structured appendix from model output."""
        fenced_patterns = (
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
        )
        for pattern in fenced_patterns:
            match = re.search(pattern, text, flags=re.S | re.I)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_feedback(self, text: str) -> list[str]:
        """Extract lightweight review-style feedback from natural language output."""
        payload = self._extract_json_payload(text)
        if payload:
            feedback = payload.get("feedback") or payload.get("review_feedback")
            if isinstance(feedback, list):
                return [str(item).strip() for item in feedback if str(item).strip()][:8]
        feedback: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                feedback.append(stripped[2:])
                continue
            if any(stripped.startswith(f"{idx}. ") for idx in range(1, 10)):
                feedback.append(stripped[3:])
                continue
            lowered = stripped.lower()
            if any(keyword in lowered for keyword in ("issue", "problem", "suggest", "feedback", "revise", "tighten")):
                feedback.append(stripped)
        return feedback[:8]

    def _should_update_canon(self, text: str) -> bool:
        """Heuristic for beneficial deviation suggestions."""
        payload = self._extract_json_payload(text)
        if payload and isinstance(payload.get("should_update_canon"), bool):
            return bool(payload["should_update_canon"])
        lowered = text.lower()
        return "update canon" in lowered or "canon should change" in lowered or "stronger direction" in lowered

    def _extract_decision(self, text: str) -> str:
        """Extract convergence decisions from free-form output."""
        payload = self._extract_json_payload(text)
        if payload and isinstance(payload.get("decision"), str):
            decision = payload["decision"].strip().lower()
            if decision in {"freeze", "continue", "escalate"}:
                return decision
        lowered = text.lower()
        if "decision: freeze" in lowered or "\nfreeze" in lowered or " can be frozen" in lowered:
            return "freeze"
        if "decision: escalate" in lowered or "escalate" in lowered:
            return "escalate"
        return "continue"

    def _canon_update_reason(self, text: str) -> str:
        """Extract a lightweight canon update reason."""
        payload = self._extract_json_payload(text)
        if payload:
            reason = payload.get("canon_update_reason") or payload.get("reason")
            if isinstance(reason, str) and reason.strip():
                return reason.strip()
        if not self._should_update_canon(text):
            return ""
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if "update canon" in lowered or "canon should change" in lowered or "stronger direction" in lowered:
                return stripped
        return "Model suggested that canon should be updated."

    def _extract_metadata(self, request: AgentExecutionRequest, text: str) -> dict[str, object]:
        """Build lightweight metadata from free-form responses."""
        metadata: dict[str, object] = {}
        payload = self._extract_json_payload(text)
        if self.role in {AgentRole.CONVERGENCE, AgentRole.CHAPTER_CONVERGENCE} and request.task_kind.value == "decide":
            metadata["decision"] = self._extract_decision(text)
            if payload and isinstance(payload.get("reason"), str) and payload.get("reason", "").strip():
                metadata["reason"] = payload["reason"].strip()
            else:
                metadata["reason"] = self._canon_update_reason(text) or text.strip().splitlines()[0]
        if self.role == AgentRole.WRITER and request.task_kind.value == "generate":
            if payload and isinstance(payload.get("summary"), str) and payload.get("summary", "").strip():
                metadata["summary"] = payload["summary"].strip()[:240]
                return metadata
            first_paragraph = text.strip().split("\n\n")[0].strip() if text.strip() else ""
            metadata["summary"] = first_paragraph[:240] if first_paragraph else request.objective
        if request.task_kind.value == "revise":
            metadata["revision_applied"] = True
            if payload and isinstance(payload.get("canon_patch"), str) and payload.get("canon_patch", "").strip():
                metadata["canon_patch"] = payload["canon_patch"].strip()
            elif request.metadata.get("deviation_action") == "update_canon":
                metadata["canon_patch"] = f"Canon updated based on chapter {request.context.get('chapter_index', 'unknown')} creative deviation."
        return metadata


class IdeationSubAgent(StubSubAgent):
    """Base placeholder for ideation-focused agents."""

    group = "ideation"


class WritingSubAgent(StubSubAgent):
    """Base placeholder for writing-focused agents."""

    group = "writing"
