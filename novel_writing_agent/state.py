"""State models for the NovelWritingAgent framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from .models import CanonAsset, NovelMode, NovelStage


@dataclass(slots=True)
class StoryOutlineState:
    """Canonical story outline and its iteration history."""

    current: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    versions: list[str] = field(default_factory=list)
    review_feedback: list[str] = field(default_factory=list)
    working_notes: list[str] = field(default_factory=list)
    frozen: bool = False


@dataclass(slots=True)
class CharacterProfileState:
    """Canonical character bible."""

    current: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    versions: list[str] = field(default_factory=list)
    review_feedback: list[str] = field(default_factory=list)
    working_notes: list[str] = field(default_factory=list)
    frozen: bool = False


@dataclass(slots=True)
class ChapterOutlineState:
    """Canonical chapter plan."""

    chapters: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    review_feedback: list[str] = field(default_factory=list)
    working_notes: list[str] = field(default_factory=list)
    frozen: bool = False


@dataclass(slots=True)
class DraftState:
    """Drafting progress and chapter outputs."""

    chapter_drafts: dict[int, str] = field(default_factory=dict)
    chapter_sections: dict[int, dict[str, str]] = field(default_factory=dict)
    draft_versions: dict[int, list[str]] = field(default_factory=dict)
    chapter_working_notes: dict[int, list[str]] = field(default_factory=dict)
    chapter_summaries: dict[int, str] = field(default_factory=dict)
    chapter_status: dict[int, str] = field(default_factory=dict)
    review_feedback_by_chapter: dict[int, list[str]] = field(default_factory=dict)
    revision_rounds_by_chapter: dict[int, int] = field(default_factory=dict)
    latest_feedback: list[str] = field(default_factory=list)
    completed_chapters: list[int] = field(default_factory=list)
    canon_change_log: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReviewIssue:
    """Tracked review issue with lifecycle state."""

    issue_id: str
    text: str
    status: str = "open"
    chapter_index: int | None = None
    source: str = ""
    created_stage: str = ""
    resolution_note: str = ""


@dataclass(slots=True)
class MemoryEvent:
    """Structured long-term memory event for retrieval and continuation."""

    event_id: str
    event_type: str
    summary: str
    detail: str = ""
    status: str = "active"
    chapter_index: int | None = None
    source: str = ""
    characters: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CanonState:
    """Semi-static story canon used by all later stages."""

    story_outline: StoryOutlineState = field(default_factory=StoryOutlineState)
    character_profiles: CharacterProfileState = field(default_factory=CharacterProfileState)
    chapter_outline: ChapterOutlineState = field(default_factory=ChapterOutlineState)
    world_rules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryState:
    """Long-lived memory buckets for retrieval and auditing."""

    outline_memory: list[str] = field(default_factory=list)
    character_memory: list[str] = field(default_factory=list)
    world_memory: list[str] = field(default_factory=list)
    chapter_memory: dict[int, list[str]] = field(default_factory=dict)
    narrative_history: list[str] = field(default_factory=list)
    review_memory: list[ReviewIssue] = field(default_factory=list)
    event_memory: list[MemoryEvent] = field(default_factory=list)


@dataclass(slots=True)
class NovelProjectState:
    """Top-level state shared across the novel writing framework."""

    project_id: str
    title: str
    user_brief: str
    mode: NovelMode
    output_language: str = "zh-CN"
    stage: NovelStage = NovelStage.IDEATION
    canon: CanonState = field(default_factory=CanonState)
    draft: DraftState = field(default_factory=DraftState)
    memory: MemoryState = field(default_factory=MemoryState)
    pending_questions: list[str] = field(default_factory=list)
    stage_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize the full project state for persistence and recovery."""
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["stage"] = self.stage.value
        payload["memory"]["review_memory"] = [asdict(item) for item in self.memory.review_memory]
        payload["memory"]["event_memory"] = [asdict(item) for item in self.memory.event_memory]
        payload["draft"]["chapter_drafts"] = self._stringify_int_keys(self.draft.chapter_drafts)
        payload["draft"]["chapter_sections"] = self._stringify_nested_int_keys(self.draft.chapter_sections)
        payload["draft"]["draft_versions"] = self._stringify_int_keys(self.draft.draft_versions)
        payload["draft"]["chapter_working_notes"] = self._stringify_int_keys(self.draft.chapter_working_notes)
        payload["draft"]["chapter_summaries"] = self._stringify_int_keys(self.draft.chapter_summaries)
        payload["draft"]["chapter_status"] = self._stringify_int_keys(self.draft.chapter_status)
        payload["draft"]["review_feedback_by_chapter"] = self._stringify_int_keys(self.draft.review_feedback_by_chapter)
        payload["memory"]["chapter_memory"] = self._stringify_int_keys(self.memory.chapter_memory)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> NovelProjectState:
        """Rebuild a full project state from persisted JSON."""
        state = cls(
            project_id=str(payload["project_id"]),
            title=str(payload["title"]),
            user_brief=str(payload["user_brief"]),
            mode=NovelMode(str(payload["mode"])),
            output_language=str(payload.get("output_language", "zh-CN")),
            stage=NovelStage(str(payload.get("stage", NovelStage.IDEATION.value))),
        )

        canon = payload.get("canon", {})
        if isinstance(canon, dict):
            story_outline = canon.get("story_outline", {})
            if isinstance(story_outline, dict):
                state.canon.story_outline = StoryOutlineState(**story_outline)
            character_profiles = canon.get("character_profiles", {})
            if isinstance(character_profiles, dict):
                state.canon.character_profiles = CharacterProfileState(**character_profiles)
            chapter_outline = canon.get("chapter_outline", {})
            if isinstance(chapter_outline, dict):
                state.canon.chapter_outline = ChapterOutlineState(**chapter_outline)
            world_rules = canon.get("world_rules", [])
            state.canon.world_rules = list(world_rules) if isinstance(world_rules, list) else []

        draft = payload.get("draft", {})
        if isinstance(draft, dict):
            state.draft.chapter_drafts = cls._int_keys(draft.get("chapter_drafts", {}))
            state.draft.chapter_sections = cls._nested_int_keys(draft.get("chapter_sections", {}))
            state.draft.draft_versions = cls._int_keys(draft.get("draft_versions", {}))
            state.draft.chapter_working_notes = cls._int_keys(draft.get("chapter_working_notes", {}))
            state.draft.chapter_summaries = cls._int_keys(draft.get("chapter_summaries", {}))
            state.draft.chapter_status = cls._int_keys(draft.get("chapter_status", {}))
            state.draft.review_feedback_by_chapter = cls._int_keys(draft.get("review_feedback_by_chapter", {}))
            state.draft.revision_rounds_by_chapter = cls._int_keys(draft.get("revision_rounds_by_chapter", {}))
            state.draft.latest_feedback = list(draft.get("latest_feedback", []))
            state.draft.completed_chapters = list(draft.get("completed_chapters", []))
            state.draft.canon_change_log = list(draft.get("canon_change_log", []))

        memory = payload.get("memory", {})
        if isinstance(memory, dict):
            state.memory.outline_memory = list(memory.get("outline_memory", []))
            state.memory.character_memory = list(memory.get("character_memory", []))
            state.memory.world_memory = list(memory.get("world_memory", []))
            state.memory.chapter_memory = cls._int_keys(memory.get("chapter_memory", {}))
            state.memory.narrative_history = list(memory.get("narrative_history", []))
            state.memory.review_memory = [
                ReviewIssue(**item) for item in memory.get("review_memory", []) if isinstance(item, dict)
            ]
            state.memory.event_memory = [
                MemoryEvent(**item) for item in memory.get("event_memory", []) if isinstance(item, dict)
            ]

        state.pending_questions = list(payload.get("pending_questions", []))
        state.stage_notes = list(payload.get("stage_notes", []))
        return state

    def memory_snapshot(self) -> dict[str, object]:
        """Serialize only long-lived memory for persistence and inspection."""
        return {
            "outline_memory": list(self.memory.outline_memory),
            "character_memory": list(self.memory.character_memory),
            "world_memory": list(self.memory.world_memory),
            "chapter_memory": self._stringify_int_keys(self.memory.chapter_memory),
            "narrative_history": list(self.memory.narrative_history),
            "review_memory": [asdict(item) for item in self.memory.review_memory],
            "event_memory": [asdict(item) for item in self.memory.event_memory],
        }

    @staticmethod
    def _stringify_int_keys(mapping: dict[int, object] | dict[str, object]) -> dict[str, object]:
        return {str(key): value for key, value in mapping.items()}

    @staticmethod
    def _stringify_nested_int_keys(mapping: dict[int, dict[str, str]]) -> dict[str, dict[str, str]]:
        return {str(key): value for key, value in mapping.items()}

    @staticmethod
    def _int_keys(mapping: dict[str, object] | dict[int, object]) -> dict[int, object]:
        return {int(key): value for key, value in mapping.items()}

    @staticmethod
    def _nested_int_keys(mapping: dict[str, dict[str, str]] | dict[int, dict[str, str]]) -> dict[int, dict[str, str]]:
        return {int(key): value for key, value in mapping.items()}

    def snapshot_for_main_agent(self) -> dict[str, object]:
        """Return a compact, clean state view for orchestration decisions."""
        return {
            "project_id": self.project_id,
            "title": self.title,
            "user_brief": self.user_brief,
            "mode": self.mode.value,
            "output_language": self.output_language,
            "stage": self.stage.value,
            "story_outline_ready": self.canon.story_outline.frozen,
            "character_profiles_ready": self.canon.character_profiles.frozen,
            "chapter_outline_ready": self.canon.chapter_outline.frozen,
            "completed_chapters": len(self.draft.completed_chapters),
            "latest_feedback": list(self.draft.latest_feedback[-3:]),
            "pending_questions": list(self.pending_questions),
            "stage_notes": list(self.stage_notes[-5:]),
        }

    def canon_text(self, asset: CanonAsset) -> str:
        """Return the current text for a canon asset."""
        if asset == CanonAsset.STORY_OUTLINE:
            return self.canon.story_outline.current
        if asset == CanonAsset.CHARACTER_PROFILES:
            return self.canon.character_profiles.current
        if asset == CanonAsset.CHAPTER_OUTLINE:
            return "\n".join(self.canon.chapter_outline.chapters)
        raise ValueError(f"Unsupported canon asset: {asset}")

    def update_canon_asset(self, asset: CanonAsset, content: str) -> None:
        """Merge section-level updates and re-assemble a complete canon asset."""
        if asset == CanonAsset.STORY_OUTLINE:
            if self.canon.story_outline.current:
                self.canon.story_outline.versions.append(self.canon.story_outline.current)
            merged_sections = self._merge_asset_sections(
                asset,
                self.canon.story_outline.sections,
                content,
            )
            self.canon.story_outline.sections = merged_sections
            self.canon.story_outline.current = self._assemble_asset(asset, merged_sections, content)
            return
        if asset == CanonAsset.CHARACTER_PROFILES:
            if self.canon.character_profiles.current:
                self.canon.character_profiles.versions.append(self.canon.character_profiles.current)
            merged_sections = self._merge_asset_sections(
                asset,
                self.canon.character_profiles.sections,
                content,
            )
            self.canon.character_profiles.sections = merged_sections
            self.canon.character_profiles.current = self._assemble_asset(asset, merged_sections, content)
            return
        if asset == CanonAsset.CHAPTER_OUTLINE:
            if self.canon.chapter_outline.chapters:
                self.canon.chapter_outline.review_feedback.append("Previous chapter outline replaced.")
            merged_sections = self._merge_asset_sections(
                asset,
                self.canon.chapter_outline.sections,
                content,
            )
            self.canon.chapter_outline.sections = merged_sections
            assembled = self._assemble_asset(asset, merged_sections, content)
            self.canon.chapter_outline.chapters = [line.strip() for line in assembled.splitlines() if line.strip()]
            return
        raise ValueError(f"Unsupported canon asset: {asset}")

    def add_asset_feedback(self, asset: CanonAsset, feedback: list[str]) -> None:
        """Append review feedback to a canon asset."""
        if not feedback:
            return
        if asset == CanonAsset.STORY_OUTLINE:
            self.canon.story_outline.review_feedback.extend(feedback)
            return
        if asset == CanonAsset.CHARACTER_PROFILES:
            self.canon.character_profiles.review_feedback.extend(feedback)
            return
        if asset == CanonAsset.CHAPTER_OUTLINE:
            self.canon.chapter_outline.review_feedback.extend(feedback)
            return
        raise ValueError(f"Unsupported canon asset: {asset}")

    def add_asset_working_note(self, asset: CanonAsset, note: str) -> None:
        """Store process-only notes outside the final canon text."""
        if not note.strip():
            return
        if asset == CanonAsset.STORY_OUTLINE:
            self.canon.story_outline.working_notes.append(note.strip())
            return
        if asset == CanonAsset.CHARACTER_PROFILES:
            self.canon.character_profiles.working_notes.append(note.strip())
            return
        if asset == CanonAsset.CHAPTER_OUTLINE:
            self.canon.chapter_outline.working_notes.append(note.strip())
            return
        raise ValueError(f"Unsupported canon asset: {asset}")

    def asset_working_notes(self, asset: CanonAsset) -> list[str]:
        """Return recent working notes for one canon asset."""
        if asset == CanonAsset.STORY_OUTLINE:
            return list(self.canon.story_outline.working_notes)
        if asset == CanonAsset.CHARACTER_PROFILES:
            return list(self.canon.character_profiles.working_notes)
        if asset == CanonAsset.CHAPTER_OUTLINE:
            return list(self.canon.chapter_outline.working_notes)
        raise ValueError(f"Unsupported canon asset: {asset}")

    def freeze_asset(self, asset: CanonAsset) -> None:
        """Freeze a canon asset once it is sufficiently converged."""
        if asset == CanonAsset.STORY_OUTLINE:
            self.canon.story_outline.frozen = True
            return
        if asset == CanonAsset.CHARACTER_PROFILES:
            self.canon.character_profiles.frozen = True
            return
        if asset == CanonAsset.CHAPTER_OUTLINE:
            self.canon.chapter_outline.frozen = True
            return
        raise ValueError(f"Unsupported canon asset: {asset}")

    def canon_ready(self) -> bool:
        """Whether all required canon assets are frozen."""
        return (
            self.canon.story_outline.frozen
            and self.canon.character_profiles.frozen
            and self.canon.chapter_outline.frozen
        )

    def latest_chapter_index(self) -> int:
        """Return the latest chapter that has a draft or should be drafted next."""
        if self.draft.chapter_drafts:
            return max(self.draft.chapter_drafts)
        return 1

    def start_chapter(self, chapter_index: int) -> None:
        """Mark a chapter as actively being drafted."""
        self.draft.chapter_status[chapter_index] = "drafting"
        self.draft.revision_rounds_by_chapter.setdefault(chapter_index, 0)

    def store_chapter_draft(self, chapter_index: int, content: str) -> None:
        """Persist a chapter draft while keeping section-level continuity."""
        previous = self.draft.chapter_drafts.get(chapter_index)
        if previous:
            self.draft.draft_versions.setdefault(chapter_index, []).append(previous)
        cleaned_content = self._normalize_draft_text(content)
        merged_sections = self._merge_draft_sections(
            self.draft.chapter_sections.get(chapter_index, {}),
            cleaned_content,
        )
        self.draft.chapter_sections[chapter_index] = merged_sections
        self.draft.chapter_drafts[chapter_index] = self._assemble_draft(merged_sections, cleaned_content)
        self.draft.chapter_status[chapter_index] = "drafted"
        if chapter_index not in self.memory.chapter_memory:
            self.memory.chapter_memory[chapter_index] = []
        self.memory.chapter_memory[chapter_index].append(self.draft.chapter_drafts[chapter_index])

    def store_chapter_summary(self, chapter_index: int, summary: str) -> None:
        """Persist summary memory for a chapter."""
        self.draft.chapter_summaries[chapter_index] = summary
        self.memory.narrative_history.append(summary)
        self.memory.chapter_memory.setdefault(chapter_index, []).append(summary)
        self.record_memory_event(
            event_type="plot_event",
            summary=summary,
            detail=summary,
            chapter_index=chapter_index,
            source="chapter_summary",
            keywords=self._extract_keywords(summary),
        )
        for event_type, event_summary in self._derive_events_from_summary(summary):
            self.record_memory_event(
                event_type=event_type,
                summary=event_summary,
                detail=summary,
                chapter_index=chapter_index,
                source="chapter_summary.derived",
                keywords=self._extract_keywords(event_summary),
            )

    def add_review_feedback(self, chapter_index: int, feedback: list[str]) -> None:
        """Persist review feedback for later revision and memory retrieval."""
        if not feedback:
            return
        self.draft.review_feedback_by_chapter.setdefault(chapter_index, []).extend(feedback)
        self.draft.latest_feedback = feedback
        next_index = len(self.memory.review_memory) + 1
        for offset, item in enumerate(feedback, start=0):
            issue = ReviewIssue(
                issue_id=f"review-{chapter_index:03d}-{next_index + offset:03d}",
                text=item.strip(),
                chapter_index=chapter_index,
                source="reviewer",
                created_stage=self.stage.value,
            )
            self.memory.review_memory.append(issue)
            self.record_memory_event(
                event_type="open_question",
                summary=item.strip(),
                detail=self._format_review_event_detail(issue),
                chapter_index=chapter_index,
                source="review_feedback",
                status="active",
                keywords=self._extract_keywords(item),
            )
        self.draft.chapter_status[chapter_index] = "reviewed"

    def resolve_review_feedback(
        self,
        chapter_index: int,
        resolution_note: str,
        status: str = "absorbed",
    ) -> None:
        """Resolve active review issues after revision or canon absorption."""
        note = resolution_note.strip()
        for issue in self.memory.review_memory:
            if issue.chapter_index != chapter_index:
                continue
            if issue.status not in {"open", "active"}:
                continue
            issue.status = status
            if note:
                issue.resolution_note = note
        for event in self.memory.event_memory:
            if event.chapter_index != chapter_index:
                continue
            if event.event_type != "open_question":
                continue
            if event.status not in {"active", "open"}:
                continue
            event.status = status
            if note and note not in event.detail:
                event.detail = (event.detail + "\n\n解决说明：" + note).strip()

    def active_review_items(
        self,
        chapter_index: int | None = None,
        limit: int = 10,
    ) -> list[ReviewIssue]:
        """Return unresolved review issues, optionally scoped to one chapter."""
        items = [
            issue
            for issue in self.memory.review_memory
            if issue.status in {"open", "active"}
            and (chapter_index is None or issue.chapter_index == chapter_index)
        ]
        return items[-limit:]

    def recent_review_items(self, limit: int = 10) -> list[ReviewIssue]:
        """Return recent issues regardless of status for auditing."""
        return self.memory.review_memory[-limit:]

    def record_memory_event(
        self,
        event_type: str,
        summary: str,
        detail: str = "",
        chapter_index: int | None = None,
        source: str = "",
        status: str = "active",
        characters: list[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        """Append a structured event to long-term memory."""
        clean_summary = summary.strip()
        for event in self.memory.event_memory:
            if (
                event.event_type == event_type
                and event.summary == clean_summary
                and event.chapter_index == chapter_index
                and event.source == source
            ):
                return
        next_index = len(self.memory.event_memory) + 1
        self.memory.event_memory.append(
            MemoryEvent(
                event_id=f"event-{next_index:04d}",
                event_type=event_type,
                summary=clean_summary,
                detail=(detail or summary).strip(),
                status=status,
                chapter_index=chapter_index,
                source=source,
                characters=list(characters or []),
                keywords=list(keywords or []),
            )
        )

    def active_memory_events(
        self,
        chapter_index: int | None = None,
        event_types: set[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryEvent]:
        """Return active structured events for retrieval."""
        items = [
            event
            for event in self.memory.event_memory
            if event.status in {"active", "open"}
            and (chapter_index is None or event.chapter_index is None or event.chapter_index <= chapter_index)
            and (event_types is None or event.event_type in event_types)
        ]
        return items[-limit:]

    def recent_memory_events(self, limit: int = 20) -> list[MemoryEvent]:
        """Return recent structured events regardless of status."""
        return self.memory.event_memory[-limit:]

    def add_chapter_working_note(self, chapter_index: int, note: str) -> None:
        """Store process-only notes for a chapter outside the final draft."""
        if not note.strip():
            return
        self.draft.chapter_working_notes.setdefault(chapter_index, []).append(note.strip())

    def chapter_working_notes(self, chapter_index: int) -> list[str]:
        """Return process-only notes for a chapter."""
        return list(self.draft.chapter_working_notes.get(chapter_index, []))

    def mark_revision_complete(self, chapter_index: int) -> None:
        """Finalize a revised chapter."""
        self.draft.chapter_status[chapter_index] = "revised"
        self.draft.revision_rounds_by_chapter[chapter_index] = (
            self.draft.revision_rounds_by_chapter.get(chapter_index, 0) + 1
        )
        if chapter_index not in self.draft.completed_chapters:
            self.draft.completed_chapters.append(chapter_index)
            self.draft.completed_chapters.sort()

    def mark_chapter_needs_another_revision(self, chapter_index: int) -> None:
        """Keep chapter active for another review-revision cycle."""
        self.draft.chapter_status[chapter_index] = "needs_revision"
        self.draft.revision_rounds_by_chapter[chapter_index] = (
            self.draft.revision_rounds_by_chapter.get(chapter_index, 0) + 1
        )

    def log_canon_change(self, note: str) -> None:
        """Track why canon changed during drafting."""
        self.draft.canon_change_log.append(note)
        self.stage_notes.append(note)

    def _merge_draft_sections(
        self,
        existing_sections: dict[str, str],
        content: str,
    ) -> dict[str, str]:
        parsed_sections = self._parse_draft_sections(content)
        if not parsed_sections:
            return existing_sections.copy()
        merged = existing_sections.copy()
        for key, value in parsed_sections.items():
            merged[key] = value.strip()
        return merged

    @staticmethod
    def _assemble_draft(sections: dict[str, str], fallback_content: str) -> str:
        if not sections:
            return fallback_content.strip()
        return "\n\n".join(value.strip() for value in sections.values() if value.strip()).strip()

    def _parse_draft_sections(self, text: str) -> dict[str, str]:
        content = text.strip()
        if not content:
            return {}
        markdown_sections = self._parse_markdown_sections(
            content,
            header_patterns=(r"^##\s+.+$", r"^###\s+.+$"),
        )
        if markdown_sections:
            return markdown_sections
        scene_sections = self._parse_plain_chapter_sections(content)
        if scene_sections:
            return scene_sections
        return {"__full__": content}

    @staticmethod
    def _normalize_draft_text(text: str) -> str:
        content = text.strip()
        if not content:
            return content
        lines = content.splitlines()
        cleaned_lines: list[str] = []
        skip_meta_section = False
        meta_section_patterns = (
            r"^##\s+revision notes",
            r"^##\s+changes made",
            r"^##\s+修改说明",
            r"^##\s+修订说明",
            r"^##\s+本轮修改",
        )
        content_resume_patterns = (
            r"^##\s+.+$",
            r"^###\s+.+$",
            r"^scene\s+\d+",
            r"^场景[一二三四五六七八九十0-9]",
        )
        for line in lines:
            stripped = line.strip()
            lowered = stripped.lower()
            if any(re.match(pattern, lowered) for pattern in meta_section_patterns):
                skip_meta_section = True
                continue
            if skip_meta_section:
                if any(re.match(pattern, lowered) for pattern in content_resume_patterns):
                    skip_meta_section = False
                else:
                    continue
            if any(token in lowered for token in ("changes made", "revision notes", "修改说明", "修订说明", "本轮修改")):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned or content

    def _merge_asset_sections(
        self,
        asset: CanonAsset,
        existing_sections: dict[str, str],
        content: str,
    ) -> dict[str, str]:
        parsed_sections = self._parse_asset_sections(asset, content)
        if not parsed_sections:
            return existing_sections.copy()
        merged = existing_sections.copy()
        for key, value in parsed_sections.items():
            merged[key] = value.strip()
        return merged

    def _assemble_asset(
        self,
        asset: CanonAsset,
        sections: dict[str, str],
        fallback_content: str,
    ) -> str:
        if not sections:
            return fallback_content.strip()

        if asset == CanonAsset.CHAPTER_OUTLINE:
            return "\n\n".join(value.strip() for value in sections.values() if value.strip()).strip()

        return "\n\n".join(value.strip() for value in sections.values() if value.strip()).strip()

    def _parse_asset_sections(self, asset: CanonAsset, content: str) -> dict[str, str]:
        text = content.strip()
        if not text:
            return {}

        if asset == CanonAsset.STORY_OUTLINE:
            return self._parse_markdown_sections(text, header_patterns=(r"^##\s+.+$", r"^###\s+.+$"))
        if asset == CanonAsset.CHARACTER_PROFILES:
            return self._parse_markdown_sections(text, header_patterns=(r"^##\s+.+$",))
        if asset == CanonAsset.CHAPTER_OUTLINE:
            chapter_sections = self._parse_markdown_sections(
                text,
                header_patterns=(r"^##\s+.+$", r"^###\s+.+$"),
            )
            if chapter_sections:
                return chapter_sections
            return self._parse_plain_chapter_sections(text)
        raise ValueError(f"Unsupported canon asset: {asset}")

    @staticmethod
    def _parse_markdown_sections(text: str, header_patterns: tuple[str, ...]) -> dict[str, str]:
        lines = text.splitlines()
        sections: dict[str, str] = {}
        current_key: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_key, current_lines
            if current_key and any(line.strip() for line in current_lines):
                sections[current_key] = "\n".join(current_lines).strip()
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if any(re.match(pattern, stripped) for pattern in header_patterns):
                flush()
                current_key = stripped
                current_lines = [line]
            else:
                if current_key is None:
                    current_key = "__preamble__"
                current_lines.append(line)
        flush()
        return {key: value for key, value in sections.items() if value.strip()}

    @staticmethod
    def _parse_plain_chapter_sections(text: str) -> dict[str, str]:
        lines = text.splitlines()
        sections: dict[str, str] = {}
        current_key: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            nonlocal current_key, current_lines
            if current_key and any(line.strip() for line in current_lines):
                sections[current_key] = "\n".join(current_lines).strip()
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if re.match(r"^(第\s*\d+\s*章|chapter\s*\d+|##\s*第.+章|##\s*chapter)", stripped, flags=re.I):
                flush()
                current_key = stripped
                current_lines = [line]
            else:
                if current_key is None:
                    current_key = "__preamble__"
                current_lines.append(line)
        flush()
        return {key: value for key, value in sections.items() if value.strip()}

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        seen: list[str] = []
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,12}", text):
            if token not in seen:
                seen.append(token)
        return seen[:8]

    @staticmethod
    def _format_review_event_detail(issue: ReviewIssue) -> str:
        chapter_text = f"第{issue.chapter_index}章" if issue.chapter_index is not None else "全局"
        return f"{chapter_text}审核问题：{issue.text}"

    @classmethod
    def _derive_events_from_summary(cls, summary: str) -> list[tuple[str, str]]:
        text = summary.strip()
        if not text:
            return []

        events: list[tuple[str, str]] = []
        if cls._contains_any(text, ("线索", "令牌", "预感", "异样", "伏笔", "暗示", "信物", "秘密")):
            events.append(("foreshadowing", text))
        if cls._contains_any(text, ("发现", "得知", "揭开", "真相", "原来", "认出", "证实", "暴露")):
            events.append(("reveal", text))
        if cls._contains_any(text, ("决裂", "和解", "背叛", "结盟", "联手", "反目", "疏远", "信任")):
            events.append(("relationship_change", text))
        if cls._contains_any(text, ("决定", "怀疑", "明白", "意识到", "不再", "开始", "发誓", "下定决心")):
            events.append(("character_state_change", text))
        return events

    @staticmethod
    def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
        return any(needle in text for needle in needles)
