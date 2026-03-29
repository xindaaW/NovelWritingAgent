"""Memory orchestration for the NovelWritingAgent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

from .schema import Message

from .models import AgentRole
from .state import MemoryEvent, NovelProjectState, ReviewIssue


@dataclass(slots=True)
class WorkingMemoryBundle:
    """Task-scoped memory injected into a sub-agent."""

    task_context: list[str] = field(default_factory=list)
    canon_context: list[str] = field(default_factory=list)
    relation_context: list[str] = field(default_factory=list)
    scene_cast_context: list[str] = field(default_factory=list)
    narrative_context: list[str] = field(default_factory=list)
    review_context: list[str] = field(default_factory=list)
    planning_context: list[str] = field(default_factory=list)
    retrieval_index: dict[str, dict[str, list[dict[str, str]]]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Simple text budget for one working-memory section."""

    max_items: int
    max_chars_per_item: int
    max_total_chars: int
    max_tokens_per_item: int
    max_total_tokens: int


class MemoryOrchestrator:
    """Build task-scoped memory bundles from project state.

    The key design is:
    - long-lived memory is explicit in state
    - each role only receives a narrow working set
    - relevance selection should be semantic when a model is available
    """

    SECTION_BUDGETS = {
        "task_context": ContextBudget(max_items=8, max_chars_per_item=700, max_total_chars=2400, max_tokens_per_item=220, max_total_tokens=760),
        "canon_context": ContextBudget(max_items=10, max_chars_per_item=1400, max_total_chars=5200, max_tokens_per_item=420, max_total_tokens=1600),
        "relation_context": ContextBudget(max_items=6, max_chars_per_item=500, max_total_chars=1800, max_tokens_per_item=160, max_total_tokens=560),
        "scene_cast_context": ContextBudget(max_items=3, max_chars_per_item=700, max_total_chars=1200, max_tokens_per_item=220, max_total_tokens=420),
        "narrative_context": ContextBudget(max_items=6, max_chars_per_item=800, max_total_chars=2400, max_tokens_per_item=260, max_total_tokens=760),
        "review_context": ContextBudget(max_items=6, max_chars_per_item=600, max_total_chars=1800, max_tokens_per_item=180, max_total_tokens=520),
        "planning_context": ContextBudget(max_items=5, max_chars_per_item=700, max_total_chars=1600, max_tokens_per_item=220, max_total_tokens=500),
    }
    RETRIEVAL_BUCKET_LIMITS = {
        "immediate": 4,
        "contextual": 6,
        "deep": 8,
    }
    RETRIEVAL_ENTRY_SUMMARY_MAX = 180
    RETRIEVAL_ENTRY_BODY_MAX = 1200
    RETRIEVAL_ENTRY_SUMMARY_TOKENS = 80
    RETRIEVAL_ENTRY_BODY_TOKENS = 320
    SEMANTIC_EMBED_DIM = 96

    def __init__(self, llm_client: object | None = None) -> None:
        self.llm_client = llm_client

    async def build_for_role(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None = None,
    ) -> WorkingMemoryBundle:
        """Select the most relevant memory slices for a given role."""
        task_context = self._task_context(state, role, chapter_index)
        canon_context = await self._canon_context(state, role, chapter_index)
        relation_context = self._relation_context(state, role, chapter_index)
        scene_cast_context = self._scene_cast_context(state, role, chapter_index)
        narrative_context = await self._narrative_context(state, role, chapter_index)
        review_context = await self._review_context(state, role, chapter_index)
        planning_context = self._planning_context(state, role)
        retrieval_index = self._retrieval_index(state, role, chapter_index)

        if role == AgentRole.WRITER:
            bundle = WorkingMemoryBundle(
                task_context=task_context,
                canon_context=canon_context,
                relation_context=relation_context,
                scene_cast_context=scene_cast_context,
                narrative_context=narrative_context,
                review_context=review_context,
                planning_context=planning_context,
                retrieval_index=retrieval_index,
            )
            return self._apply_bundle_budgets(bundle)

        if role in {
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CREATIVE_REVIEWER,
        }:
            bundle = WorkingMemoryBundle(
                task_context=task_context,
                canon_context=canon_context,
                relation_context=relation_context,
                scene_cast_context=scene_cast_context,
                narrative_context=narrative_context[-3:],
                review_context=review_context,
                planning_context=planning_context,
                retrieval_index=retrieval_index,
            )
            return self._apply_bundle_budgets(bundle)

        bundle = WorkingMemoryBundle(
            task_context=task_context,
            canon_context=canon_context,
            relation_context=relation_context,
            scene_cast_context=scene_cast_context,
            narrative_context=narrative_context[-2:],
            review_context=review_context[-5:],
            planning_context=planning_context,
            retrieval_index=retrieval_index,
        )
        return self._apply_bundle_budgets(bundle)

    def _apply_bundle_budgets(self, bundle: WorkingMemoryBundle) -> WorkingMemoryBundle:
        """Trim each context lane so prompt size stays bounded as projects grow."""
        return WorkingMemoryBundle(
            task_context=self._cap_context_list(bundle.task_context, self.SECTION_BUDGETS["task_context"]),
            canon_context=self._cap_context_list(bundle.canon_context, self.SECTION_BUDGETS["canon_context"]),
            relation_context=self._cap_context_list(bundle.relation_context, self.SECTION_BUDGETS["relation_context"]),
            scene_cast_context=self._cap_context_list(bundle.scene_cast_context, self.SECTION_BUDGETS["scene_cast_context"]),
            narrative_context=self._cap_context_list(bundle.narrative_context, self.SECTION_BUDGETS["narrative_context"]),
            review_context=self._cap_context_list(bundle.review_context, self.SECTION_BUDGETS["review_context"]),
            planning_context=self._cap_context_list(bundle.planning_context, self.SECTION_BUDGETS["planning_context"]),
            retrieval_index=self._cap_retrieval_index(bundle.retrieval_index),
        )

    def _cap_context_list(self, items: list[str], budget: ContextBudget) -> list[str]:
        """Apply both character and estimated-token budgets to a context lane."""
        trimmed: list[str] = []
        total_chars = 0
        total_tokens = 0
        for item in items[: budget.max_items]:
            normalized = self._truncate_text_to_budget(
                item.strip(),
                max_chars=budget.max_chars_per_item,
                max_tokens=budget.max_tokens_per_item,
            )
            if not normalized:
                continue
            item_tokens = self.estimate_text_tokens(normalized)
            projected_chars = total_chars + len(normalized)
            projected_tokens = total_tokens + item_tokens
            if projected_chars > budget.max_total_chars or projected_tokens > budget.max_total_tokens:
                remaining_chars = budget.max_total_chars - total_chars
                remaining_tokens = budget.max_total_tokens - total_tokens
                if remaining_chars <= 64 or remaining_tokens <= 24:
                    break
                normalized = self._truncate_text_to_budget(
                    normalized,
                    max_chars=remaining_chars,
                    max_tokens=remaining_tokens,
                )
                item_tokens = self.estimate_text_tokens(normalized)
            trimmed.append(normalized)
            total_chars += len(normalized)
            total_tokens += item_tokens
            if total_chars >= budget.max_total_chars or total_tokens >= budget.max_total_tokens:
                break
        return trimmed

    def _cap_retrieval_index(
        self,
        retrieval_index: dict[str, dict[str, list[dict[str, str]]]],
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        """Keep progressive retrieval useful without letting deep memory blow up prompts."""
        if not retrieval_index:
            return {}

        trimmed: dict[str, dict[str, list[dict[str, str]]]] = {}
        for level, buckets in retrieval_index.items():
            bucket_limit = self.RETRIEVAL_BUCKET_LIMITS.get(level, 6)
            trimmed[level] = {}
            for memory_type, entries in buckets.items():
                normalized_entries: list[dict[str, str]] = []
                for entry in entries[:bucket_limit]:
                    normalized_entries.append(
                        {
                            "id": entry.get("id", ""),
                            "summary": self._truncate_text(
                                entry.get("summary", ""),
                                self.RETRIEVAL_ENTRY_SUMMARY_MAX,
                                self.RETRIEVAL_ENTRY_SUMMARY_TOKENS,
                            ),
                            "body": self._truncate_text(
                                entry.get("body", ""),
                                self.RETRIEVAL_ENTRY_BODY_MAX,
                                self.RETRIEVAL_ENTRY_BODY_TOKENS,
                            ),
                            "source": entry.get("source", ""),
                        }
                    )
                trimmed[level][memory_type] = normalized_entries
        return trimmed

    @staticmethod
    def estimate_text_tokens(text: str) -> int:
        """Estimate prompt tokens without binding the framework to one tokenizer."""
        cleaned = text.strip()
        if not cleaned:
            return 0
        cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
        latin_words = len(re.findall(r"[A-Za-z0-9_]+", cleaned))
        ascii_nonspace = len(re.findall(r"[^\s\u4e00-\u9fff]", cleaned))
        estimated = cjk_chars + math.ceil(latin_words * 1.3) + math.ceil(ascii_nonspace / 4)
        return max(1, estimated)

    @classmethod
    def estimate_bundle_tokens(cls, bundle: WorkingMemoryBundle) -> dict[str, int]:
        """Estimate token usage by memory lane for observability and tests."""
        sections = {
            "task_context": bundle.task_context,
            "canon_context": bundle.canon_context,
            "relation_context": bundle.relation_context,
            "scene_cast_context": bundle.scene_cast_context,
            "narrative_context": bundle.narrative_context,
            "review_context": bundle.review_context,
            "planning_context": bundle.planning_context,
        }
        return {
            name: sum(cls.estimate_text_tokens(item) for item in items)
            for name, items in sections.items()
        }

    @classmethod
    def _truncate_text(cls, text: str, max_chars: int, max_tokens: int) -> str:
        """Trim long sections with an explicit marker instead of silently dropping context."""
        return cls._truncate_text_to_budget(text, max_chars=max_chars, max_tokens=max_tokens)

    @classmethod
    def _truncate_text_to_budget(cls, text: str, max_chars: int, max_tokens: int) -> str:
        """Apply character and token budgets together with a visible truncation marker."""
        cleaned = text.strip()
        if len(cleaned) <= max_chars and cls.estimate_text_tokens(cleaned) <= max_tokens:
            return cleaned
        if max_chars <= 16:
            truncated = cleaned[:max_chars]
            while truncated and cls.estimate_text_tokens(truncated) > max_tokens:
                truncated = truncated[:-1]
            return truncated

        marker = "\n...[truncated]"
        candidate = cleaned[: max_chars - len(marker)].rstrip()
        while candidate and cls.estimate_text_tokens(candidate) > max_tokens:
            overflow = cls.estimate_text_tokens(candidate) - max_tokens
            shrink_by = max(1, overflow * 2)
            candidate = candidate[:-shrink_by].rstrip()
        return (candidate + marker).strip() if candidate else cleaned[: max(1, min(max_chars, 16))]

    def _task_context(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> list[str]:
        context = [
            f"用户主题：{state.user_brief.strip()}",
            f"创作模式：{state.mode.value}",
            f"当前阶段：{state.stage.value}",
            f"输出语言：{state.output_language}",
        ]
        if chapter_index is not None:
            context.append(f"当前章节：第 {chapter_index} 章")
            chapter_goal = self._chapter_goal(state, chapter_index)
            if chapter_goal:
                context.append("章节任务：\n" + chapter_goal)
            current_draft = state.draft.chapter_drafts.get(chapter_index, "").strip()
            if current_draft:
                context.append("当前章节草稿（待续写或修订）：\n" + current_draft)
        if role in {
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
        } and chapter_index is not None:
            latest_feedback = state.draft.review_feedback_by_chapter.get(chapter_index, [])
            if latest_feedback:
                context.append("本章已有审核记录：\n- " + "\n- ".join(latest_feedback[-5:]))
        return context

    async def _canon_context(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> list[str]:
        context: list[str] = []
        if chapter_index is not None and role in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CHAPTER_CONVERGENCE,
        }:
            risk_profile = state.chapter_risk_profile(chapter_index)
            outline_limit = 5 if risk_profile["deep_review"] else 3
            character_limit = 6 if risk_profile["deep_review"] else 4
            query = self._section_query_text(state, chapter_index)
            outline_sections = await self._select_story_outline_sections(state, query, max_items=outline_limit)
            character_sections = await self._select_character_sections(state, query, chapter_index, max_items=character_limit)
            if outline_sections:
                context.extend(outline_sections)
            elif state.canon.story_outline.current:
                context.append(state.canon.story_outline.current)
            if character_sections:
                context.extend(character_sections)
            elif state.canon.character_profiles.current:
                context.append(state.canon.character_profiles.current)
        else:
            if state.canon.story_outline.current:
                context.append(state.canon.story_outline.current)
            if state.canon.character_profiles.current:
                context.append(state.canon.character_profiles.current)

        if state.canon.chapter_outline.chapters:
            if chapter_index is not None:
                chapter_goal = self._chapter_goal(state, chapter_index)
                if chapter_goal:
                    context.append(chapter_goal)
                nearby_indexes = {
                    index
                    for index in (chapter_index - 1, chapter_index, chapter_index + 1)
                    if index > 0 and index <= len(state.canon.chapter_outline.chapters)
                }
                for idx in sorted(nearby_indexes):
                    context.append(state.canon.chapter_outline.chapters[idx - 1])
            else:
                context.extend(state.canon.chapter_outline.chapters[:8])

        if state.canon.world_rules:
            context.extend(state.canon.world_rules[-5:])

        if role in {
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
        } and chapter_index is not None:
            current_draft = state.draft.chapter_drafts.get(chapter_index, "").strip()
            if current_draft:
                context.append("待审核正文：\n" + current_draft)
        return context

    def _relation_context(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> list[str]:
        character_cards = self._character_cards(state)
        if not character_cards:
            return []

        if role not in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CHAPTER_CONVERGENCE,
            AgentRole.CHARACTER,
            AgentRole.CREATIVE_REVIEWER,
        }:
            return []

        selected_names = self._relevant_character_names(state, chapter_index)
        if not selected_names:
            selected_names = list(character_cards.keys())[:3]

        context: list[str] = []
        for name in selected_names:
            card = character_cards.get(name, "")
            if not card:
                continue
            relation_lines = [
                line.strip()
                for line in card.splitlines()
                if self._looks_like_relation_line(line)
            ]
            if relation_lines:
                context.append(f"{name} 的人物关系线索：\n- " + "\n- ".join(relation_lines[:6]))
            else:
                preview = "\n".join(card.splitlines()[:6]).strip()
                if preview:
                    context.append(f"{name} 的人物卡摘要：\n{preview}")
        return context[:6]

    def _scene_cast_context(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> list[str]:
        if chapter_index is None:
            return []
        if role not in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CHAPTER_CONVERGENCE,
        }:
            return []

        character_cards = self._character_cards(state)
        names = self._relevant_character_names(state, chapter_index)
        if not names:
            names = list(character_cards.keys())[:3]
        if not names:
            return []

        entries: list[str] = []
        for name in names[:5]:
            card = character_cards.get(name, "")
            if not card:
                entries.append(f"- {name}")
                continue
            preview_lines = [
                line.strip()
                for line in card.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ][:3]
            if preview_lines:
                entries.append(f"- {name}: " + " / ".join(preview_lines))
            else:
                entries.append(f"- {name}")
        return [f"本章角色表（Scene Cast）：\n" + "\n".join(entries)]

    async def _narrative_context(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> list[str]:
        if chapter_index is None:
            return list(state.memory.narrative_history[-5:])

        context: list[str] = []
        if state.memory.compressed_narrative and chapter_index > 4:
            context.extend(state.memory.compressed_narrative[-2:])
        prior_indexes = [
            idx
            for idx in range(max(1, chapter_index - 2), chapter_index)
            if idx < chapter_index
        ]
        for idx in prior_indexes[:-1]:
            prior_summary = state.draft.chapter_summaries.get(idx, "").strip()
            if prior_summary:
                context.append(f"更早前情摘要（第{idx}章）：{prior_summary}")
        risk_profile = state.chapter_risk_profile(chapter_index)
        if chapter_index > 1:
            context.extend(state.memory.chapter_memory.get(chapter_index - 1, [])[-3:])
            previous_summary = state.draft.chapter_summaries.get(chapter_index - 1, "").strip()
            if previous_summary:
                context.append(f"上一章摘要：{previous_summary}")
        context.extend(state.memory.chapter_memory.get(chapter_index, [])[-5:])
        candidate_map = self._narrative_candidates(state, role, chapter_index)
        if chapter_index is not None and role in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CHAPTER_CONVERGENCE,
        }:
            query = self._section_query_text(state, chapter_index)
            selected_ids = await self._select_candidate_ids_with_llm(
                selector_name="narrative_selector",
                instruction="请从历史章节事件中选出当前章节写作或审核最需要参考的事件、伏笔、关系变化和最近进展。",
                query=query,
                candidates=candidate_map,
                max_items=8 if risk_profile["deep_review"] else 6,
            )
            if selected_ids:
                merged: list[str] = []
                for item in context:
                    if item and item not in merged:
                        merged.append(item)
                for item_id in selected_ids:
                    value = candidate_map.get(item_id)
                    if value and value not in merged:
                        merged.append(value)
                fallback_limit = 8 if risk_profile["deep_review"] else 6
                return merged[:fallback_limit]
            if candidate_map:
                fallback_limit = 8 if risk_profile["deep_review"] else 6
                merged = []
                for item in context + list(candidate_map.values()):
                    if item and item not in merged:
                        merged.append(item)
                return merged[:fallback_limit]
        if context:
            fallback_limit = 8 if risk_profile["deep_review"] else 6
            return context[:fallback_limit]
        return list(state.memory.narrative_history[-5:])

    async def _review_context(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> list[str]:
        issues = state.active_review_items(chapter_index=chapter_index, limit=12)
        risk_profile = state.chapter_risk_profile(chapter_index) if chapter_index is not None else None
        if chapter_index is not None and role in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CHAPTER_CONVERGENCE,
        }:
            query = self._section_query_text(state, chapter_index)
            issues = await self._select_review_issues(
                issues,
                query,
                max_items=8 if risk_profile and risk_profile["deep_review"] else 5,
            )
        elif not issues:
            issues = state.recent_review_items(limit=8)

        context = [self._format_review_issue(issue) for issue in issues]
        if chapter_index is not None:
            context.extend(state.memory.chapter_memory.get(chapter_index, [])[-2:])
        if role == AgentRole.CHARACTER_REVIEWER:
            context.extend(state.memory.character_memory[-3:])
        return context[-12:] if risk_profile and risk_profile["deep_review"] else context[-10:]

    def _chapter_goal(self, state: NovelProjectState, chapter_index: int) -> str:
        if chapter_index <= 0 or chapter_index > len(state.canon.chapter_outline.chapters):
            return ""
        return state.canon.chapter_outline.chapters[chapter_index - 1].strip()

    def _narrative_candidates(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int,
    ) -> dict[str, str]:
        candidates: dict[str, str] = {}
        for event in state.active_memory_events(
            chapter_index=chapter_index,
            event_types=self._preferred_event_types(role, "narrative"),
            limit=24,
        ):
            candidates[f"event_{event.event_id}"] = f"[{event.event_type}] {event.summary}"
        for idx, summary in sorted(state.draft.chapter_summaries.items()):
            if idx >= chapter_index:
                continue
            summary = summary.strip()
            if summary:
                candidates[f"summary_ch_{idx:03d}"] = f"第{idx}章摘要：{summary}"
        for idx, memories in sorted(state.memory.chapter_memory.items()):
            if idx >= chapter_index:
                continue
            if not memories:
                continue
            latest = memories[-1].strip()
            if latest:
                candidates[f"memory_ch_{idx:03d}"] = f"第{idx}章记忆：{latest[:400]}"
        for offset, item in enumerate(state.memory.narrative_history[-8:], start=1):
            text = item.strip()
            if text:
                candidates[f"history_{offset:02d}"] = text[:400]
        for offset, item in enumerate(state.memory.compressed_narrative[-6:], start=1):
            text = item.strip()
            if text:
                candidates[f"compressed_{offset:02d}"] = text[:400]
        return candidates

    def _planning_context(self, state: NovelProjectState, role: AgentRole) -> list[str]:
        """Expose front-stage canon-writing memory to ideation agents."""
        if role not in {
            AgentRole.PREMISE,
            AgentRole.OUTLINE,
            AgentRole.CHARACTER,
            AgentRole.CHAPTER_PLANNER,
            AgentRole.CREATIVE_REVIEWER,
            AgentRole.CONVERGENCE,
        }:
            return []

        context: list[str] = []
        context.extend(state.memory.outline_memory[-4:])
        context.extend(state.memory.character_memory[-4:])
        context.extend(state.memory.world_memory[-4:])
        return context[-10:]

    def _retrieval_index(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int | None,
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        """Build layered memory pools for progressive disclosure via tools."""
        if chapter_index is None or role not in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
        }:
            return {}

        immediate = {
            "canon": [],
            "narrative": [],
            "character": [],
            "review": [],
        }
        contextual = {
            "canon": [],
            "narrative": [],
            "character": [],
            "review": [],
        }
        deep = {
            "canon": [],
            "narrative": [],
            "character": [],
            "review": [],
        }

        chapter_goal = self._chapter_goal(state, chapter_index)
        if chapter_goal:
            immediate["canon"].append(
                self._memory_entry(
                    entry_id=f"canon_chapter_goal_{chapter_index:03d}",
                    summary=f"第{chapter_index}章任务摘要",
                    body=chapter_goal,
                    source=f"chapter_outline.chapter_{chapter_index:03d}",
                )
            )

        if state.canon.chapter_outline.chapters:
            nearby_indexes = {
                index
                for index in (chapter_index - 1, chapter_index, chapter_index + 1)
                if index > 0 and index <= len(state.canon.chapter_outline.chapters)
            }
            contextual["canon"].extend(
                self._memory_entry(
                    entry_id=f"canon_chapter_{idx:03d}",
                    summary=f"第{idx}章分章摘要",
                    body=state.canon.chapter_outline.chapters[idx - 1],
                    source=f"chapter_outline.chapter_{idx:03d}",
                )
                for idx in sorted(nearby_indexes)
            )
            deep["canon"].extend(
                self._memory_entry(
                    entry_id=f"canon_chapter_{idx:03d}",
                    summary=f"第{idx}章分章摘要",
                    body=text,
                    source=f"chapter_outline.chapter_{idx:03d}",
                )
                for idx, text in enumerate(state.canon.chapter_outline.chapters, start=1)
            )

        immediate["character"].extend(
            self._context_list_to_entries(
                self._scene_cast_context(state, role, chapter_index),
                prefix="scene_cast",
                summary_prefix="本章角色表",
            )
        )
        contextual["character"].extend(
            self._context_list_to_entries(
                self._relation_context(state, role, chapter_index),
                prefix="relation",
                summary_prefix="人物关系摘要",
            )
        )
        deep["character"].extend(
            self._memory_entry(
                entry_id=f"character_card_{idx:03d}",
                summary=self._first_meaningful_line(card) or f"人物卡 {idx}",
                body=card,
                source=f"character_profiles.card_{idx:03d}",
            )
            for idx, card in enumerate(list(self._character_cards(state).values())[:12], start=1)
        )

        previous_summary = state.draft.chapter_summaries.get(chapter_index - 1, "").strip() if chapter_index > 1 else ""
        if previous_summary:
            immediate["narrative"].append(
                self._memory_entry(
                    entry_id=f"narrative_previous_summary_{chapter_index - 1:03d}",
                    summary=f"第{chapter_index - 1}章摘要",
                    body=previous_summary,
                    source=f"chapter_summary.{chapter_index - 1:03d}",
                )
            )
        contextual["narrative"].extend(self._narrative_candidate_entries(state, role, chapter_index))
        deep["narrative"].extend(
            self._memory_entry(
                entry_id=f"compressed_narrative_{idx:03d}",
                summary=f"长程摘要{idx}",
                body=text,
                source=f"compressed_narrative.{idx:03d}",
            )
            for idx, text in enumerate(state.memory.compressed_narrative[-6:], start=1)
        )
        deep["narrative"].extend(
            self._memory_event_to_entry(event)
            for event in state.recent_memory_events(limit=24)
            if event.event_type in self._preferred_event_types(role, "narrative")
        )

        immediate["review"].extend(
            self._memory_event_to_entry(event)
            for event in state.active_memory_events(
                chapter_index=chapter_index,
                event_types={"open_question"},
                limit=3,
            )
        )
        contextual["review"].extend(
            self._memory_event_to_entry(event)
            for event in state.active_memory_events(
                chapter_index=chapter_index,
                event_types={"open_question"},
                limit=8,
            )
        )
        deep["review"].extend(
            self._memory_event_to_entry(event)
            for event in state.recent_memory_events(limit=20)
            if event.event_type == "open_question"
        )

        if state.canon.story_outline.current:
            contextual["canon"].append(
                self._memory_entry(
                    entry_id="story_outline_current",
                    summary="当前故事大纲总览",
                    body=state.canon.story_outline.current,
                    source="story_outline.current",
                )
            )
        if state.canon.character_profiles.current:
            contextual["canon"].append(
                self._memory_entry(
                    entry_id="character_profiles_current",
                    summary="当前人物小传总览",
                    body=state.canon.character_profiles.current,
                    source="character_profiles.current",
                )
            )
        if state.canon.story_outline.sections:
            deep["canon"].extend(
                self._memory_entry(
                    entry_id=f"story_outline_section_{section_id}",
                    summary=self._first_meaningful_line(text) or section_id,
                    body=text,
                    source=f"story_outline.section.{section_id}",
                )
                for section_id, text in list(state.canon.story_outline.sections.items())[:12]
            )
        if state.canon.character_profiles.sections:
            deep["canon"].extend(
                self._memory_entry(
                    entry_id=f"character_profile_section_{section_id}",
                    summary=self._first_meaningful_line(text) or section_id,
                    body=text,
                    source=f"character_profiles.section.{section_id}",
                )
                for section_id, text in list(state.canon.character_profiles.sections.items())[:12]
            )

        return {
            "immediate": {key: [item for item in values if item] for key, values in immediate.items()},
            "contextual": {key: [item for item in values if item] for key, values in contextual.items()},
            "deep": {key: [item for item in values if item] for key, values in deep.items()},
        }

    def _narrative_candidate_entries(
        self,
        state: NovelProjectState,
        role: AgentRole,
        chapter_index: int,
    ) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        event_lookup = {
            f"event_{event.event_id}": event
            for event in state.active_memory_events(
                chapter_index=chapter_index,
                event_types=self._preferred_event_types(role, "narrative"),
                limit=40,
            )
        }
        for item_id, text in self._narrative_candidates(state, role, chapter_index).items():
            if item_id in event_lookup:
                entries.append(self._memory_event_to_entry(event_lookup[item_id]))
                continue
            entries.append(
                self._memory_entry(
                    entry_id=item_id,
                    summary=self._summarize_memory_item(text),
                    body=text,
                    source=f"narrative_candidate.{item_id}",
                )
            )
        return entries

    def _preferred_event_types(self, role: AgentRole, memory_type: str) -> set[str]:
        if memory_type == "review":
            return {"open_question"}
        if role in {
            AgentRole.WRITER,
            AgentRole.CHARACTER_REVIEWER,
            AgentRole.CONTINUITY_REVIEWER,
            AgentRole.STYLE_REVIEWER,
            AgentRole.CHAPTER_CONVERGENCE,
        }:
            return {"plot_event", "foreshadowing", "reveal", "relationship_change", "character_state_change"}
        return {"plot_event", "reveal"}

    def _memory_event_to_entry(self, event: MemoryEvent) -> dict[str, str]:
        chapter_text = f"第{event.chapter_index}章" if event.chapter_index is not None else "全局"
        source = event.source or event.event_type
        summary = f"{chapter_text}[{event.event_type}] {event.summary}"
        if event.status not in {"active", "open"}:
            summary += f" ({event.status})"
        return self._memory_entry(
            entry_id=event.event_id,
            summary=summary,
            body=event.detail or event.summary,
            source=source,
        )

    def _context_list_to_entries(
        self,
        items: list[str],
        prefix: str,
        summary_prefix: str,
    ) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for idx, item in enumerate(items, start=1):
            if not item:
                continue
            entries.append(
                self._memory_entry(
                    entry_id=f"{prefix}_{idx:03d}",
                    summary=f"{summary_prefix}{idx}",
                    body=item,
                    source=f"{prefix}.{idx:03d}",
                )
            )
        return entries

    def _review_issue_to_entry(self, issue: ReviewIssue) -> dict[str, str]:
        chapter_text = f"第{issue.chapter_index}章" if issue.chapter_index is not None else "全局"
        return self._memory_entry(
            entry_id=issue.issue_id,
            summary=f"{chapter_text}审核问题：{issue.text[:48]}",
            body=self._format_review_issue(issue),
            source=f"review_issue.{issue.status}",
        )

    def _memory_entry(
        self,
        entry_id: str,
        summary: str,
        body: str,
        source: str,
    ) -> dict[str, str]:
        clean_summary = (summary or "").strip()
        clean_body = (body or "").strip()
        return {
            "id": entry_id,
            "summary": clean_summary[:240] or self._summarize_memory_item(clean_body),
            "body": clean_body,
            "source": source,
        }

    def _first_meaningful_line(self, text: str) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped[:120]
        return text.strip().splitlines()[0][:120] if text.strip() else ""

    def _summarize_memory_item(self, text: str) -> str:
        cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return cleaned[:120]

    def _character_cards(self, state: NovelProjectState) -> dict[str, str]:
        text = state.canon.character_profiles.current.strip()
        if not text:
            return {}

        cards: dict[str, str] = {}
        matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.M))
        for idx, match in enumerate(matches):
            raw_name = match.group(1).strip()
            if not raw_name or raw_name.lower() in {"character profiles", "人物小传"}:
                continue
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            cards[raw_name] = text[start:end].strip()
        return cards

    def _relevant_character_names(self, state: NovelProjectState, chapter_index: int | None) -> list[str]:
        character_cards = self._character_cards(state)
        if not character_cards:
            return []
        source_parts: list[str] = []
        if chapter_index is not None:
            source_parts.append(self._chapter_goal(state, chapter_index))
            source_parts.append(state.draft.chapter_drafts.get(chapter_index, ""))
            source_parts.append(state.draft.chapter_summaries.get(chapter_index, ""))
            if chapter_index > 1:
                source_parts.append(state.draft.chapter_summaries.get(chapter_index - 1, ""))
        source_text = "\n".join(part for part in source_parts if part).strip()

        matched_names = [name for name in character_cards if name and name in source_text]
        if matched_names:
            return matched_names[:5]
        return list(character_cards.keys())[:3]

    async def _select_story_outline_sections(
        self,
        state: NovelProjectState,
        query: str,
        max_items: int,
    ) -> list[str]:
        sections = state.canon.story_outline.sections
        if not sections:
            return []
        selected_ids = await self._select_candidate_ids_with_llm(
            selector_name="story_outline_selector",
            instruction="请从故事大纲 section 中选出当前章节写作最相关的部分，优先选与当前章节目标、人物冲突、阶段推进直接相关的 section。",
            query=query,
            candidates=sections,
            max_items=max_items,
        )
        if selected_ids:
            return [sections[item_id] for item_id in selected_ids if item_id in sections]
        return list(sections.values())[:max_items]

    async def _select_character_sections(
        self,
        state: NovelProjectState,
        query: str,
        chapter_index: int | None,
        max_items: int,
    ) -> list[str]:
        sections = state.canon.character_profiles.sections
        if not sections:
            return []
        candidate_map = sections.copy()
        selected_names = self._relevant_character_names(state, chapter_index)
        if selected_names:
            narrowed = {
                key: value
                for key, value in sections.items()
                if any(name in self._normalize_section_key(key) or self._normalize_section_key(key) in name for name in selected_names)
            }
            if narrowed:
                candidate_map = narrowed
        selected_ids = await self._select_candidate_ids_with_llm(
            selector_name="character_selector",
            instruction="请从人物小传 section 中选出当前章节最需要读取的人物卡，优先保留当前出场人物、核心对立人物和对本章关系判断关键的人物。",
            query=query,
            candidates=candidate_map,
            max_items=max_items,
        )
        if selected_ids:
            return [candidate_map[item_id] for item_id in selected_ids if item_id in candidate_map]
        return list(candidate_map.values())[:max_items]

    async def _select_review_issues(
        self,
        issues: list[ReviewIssue],
        query: str,
        max_items: int,
    ) -> list[ReviewIssue]:
        if not issues:
            return []
        candidates = {
            issue.issue_id: self._format_review_issue(issue)
            for issue in issues
        }
        selected_ids = await self._select_candidate_ids_with_llm(
            selector_name="review_issue_selector",
            instruction="请从审核问题中选出当前章节修订最需要优先处理的问题，只保留仍然会直接影响本章质量的问题。",
            query=query,
            candidates=candidates,
            max_items=max_items,
        )
        if selected_ids:
            selected = [issue for issue in issues if issue.issue_id in set(selected_ids)]
            if selected:
                return selected[:max_items]
        return issues[-max_items:]

    async def _select_candidate_ids_with_llm(
        self,
        selector_name: str,
        instruction: str,
        query: str,
        candidates: dict[str, str],
        max_items: int,
    ) -> list[str]:
        if not candidates:
            return []
        semantic_ids = self._select_candidate_ids_by_embedding(query, candidates, max_items=max_items)
        if self.llm_client is None:
            return semantic_ids

        candidate_lines = []
        for key, value in candidates.items():
            preview = value.strip().replace("\n", " ")
            candidate_lines.append(f"- {key}: {preview[:240]}")

        prompt = "\n\n".join(
            [
                f"Selector: {selector_name}",
                instruction,
                f"当前任务查询：\n{query or '无'}",
                f"最多选择 {max_items} 项。",
                "候选列表：\n" + "\n".join(candidate_lines),
                "请只输出候选 id，每行一个，不要解释。",
            ]
        )
        response = await self.llm_client.generate(
            [
                Message(
                    role="system",
                    content="你是一个小说写作系统内部的 memory selector。你的职责是根据当前任务语义判断哪些候选上下文最相关。",
                ),
                Message(role="user", content=prompt),
            ]
        )
        text = (response.content or "").strip()
        if not text:
            return semantic_ids
        selected_ids: list[str] = []
        valid_ids = set(candidates.keys())
        for line in text.splitlines():
            item = line.strip().lstrip("-").strip()
            if item in valid_ids and item not in selected_ids:
                selected_ids.append(item)
            if len(selected_ids) >= max_items:
                break
        for item in semantic_ids:
            if item in valid_ids and item not in selected_ids:
                selected_ids.append(item)
            if len(selected_ids) >= max_items:
                break
        return selected_ids[:max_items]

    def _select_candidate_ids_by_embedding(
        self,
        query: str,
        candidates: dict[str, str],
        max_items: int,
    ) -> list[str]:
        """Rank candidates locally with a lightweight hashed embedding model."""
        if not candidates:
            return []
        if not query.strip():
            return list(candidates.keys())[:max_items]

        scored = [
            (key, self._semantic_score(query, value))
            for key, value in candidates.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        ranked = [key for key, score in scored if score > 0]
        if ranked:
            return ranked[:max_items]
        return list(candidates.keys())[:max_items]

    def _semantic_score(self, query: str, candidate: str) -> float:
        query_tokens = set(self._semantic_tokens(query))
        candidate_tokens = set(self._semantic_tokens(candidate))
        if not query_tokens or not candidate_tokens:
            return 0.0
        overlap = len(query_tokens & candidate_tokens) / max(1, len(query_tokens))
        query_vector = self._hashed_embedding(query_tokens)
        candidate_vector = self._hashed_embedding(candidate_tokens)
        return self._cosine_similarity(query_vector, candidate_vector) + (0.35 * overlap)

    def _hashed_embedding(self, tokens: set[str]) -> list[float]:
        vector = [0.0] * self.SEMANTIC_EMBED_DIM
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.SEMANTIC_EMBED_DIM
            vector[index] += 1.0
        return vector

    def _semantic_tokens(self, text: str) -> list[str]:
        cleaned = text.lower().strip()
        if not cleaned:
            return []
        tokens: list[str] = []
        tokens.extend(re.findall(r"[a-z0-9_]+", cleaned))
        cjk_only = "".join(re.findall(r"[\u4e00-\u9fff]", cleaned))
        tokens.extend(list(cjk_only))
        tokens.extend(cjk_only[idx : idx + 2] for idx in range(max(0, len(cjk_only) - 1)))
        seen: list[str] = []
        for token in tokens:
            token = token.strip()
            if token and token not in seen:
                seen.append(token)
        return seen[:256]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _section_query_text(self, state: NovelProjectState, chapter_index: int) -> str:
        parts = [state.user_brief]
        chapter_goal = self._chapter_goal(state, chapter_index)
        if chapter_goal:
            parts.append(chapter_goal)
        draft = state.draft.chapter_drafts.get(chapter_index, "")
        if draft:
            parts.append(draft[:600])
        if chapter_index > 1:
            previous_summary = state.draft.chapter_summaries.get(chapter_index - 1, "")
            if previous_summary:
                parts.append(previous_summary)
        parts.extend(self._relevant_character_names(state, chapter_index))
        return "\n".join(part for part in parts if part).strip()

    def _normalize_section_key(self, key: str) -> str:
        return re.sub(r"^#+\s*", "", key).strip()

    def _format_review_issue(self, issue: ReviewIssue) -> str:
        prefix = f"[{issue.status}]"
        if issue.chapter_index is not None:
            prefix += f"[第{issue.chapter_index}章]"
        return f"{prefix} {issue.text}"

    @staticmethod
    def _looks_like_relation_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        keywords = (
            "父亲",
            "母亲",
            "儿子",
            "女儿",
            "兄长",
            "弟弟",
            "姐姐",
            "妹妹",
            "叔",
            "伯",
            "舅",
            "姑",
            "姨",
            "师父",
            "徒弟",
            "恋人",
            "爱人",
            "朋友",
            "仇",
            "敌",
            "关系",
            "对主角",
        )
        return any(keyword in stripped for keyword in keywords)
