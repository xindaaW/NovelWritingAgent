"""Artifact persistence for NovelWritingAgent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime

from .models import CanonAsset
from .memory import WorkingMemoryBundle
from .state import NovelProjectState


class ProjectArtifactStore:
    """Persist intermediate and final project artifacts to disk."""

    def __init__(self, project_id: str, workspace_root: str | Path = "workspace/novel_projects") -> None:
        self.project_id = project_id
        self.workspace_root = Path(workspace_root).resolve()
        self.project_root = self.workspace_root / project_id
        self.canon_current_dir = self.project_root / "canon" / "current"
        self.canon_history_dir = self.project_root / "canon" / "history"
        self.output_dir = self.project_root / "outputs"
        self.chapter_output_dir = self.output_dir / "chapters"
        self.memory_output_dir = self.output_dir / "memory"
        self.review_dir = self.project_root / "reviews"
        self.state_dir = self.project_root / "state"
        self.memory_dir = self.project_root / "memory"
        for directory in (
            self.project_root,
            self.canon_current_dir,
            self.canon_history_dir,
            self.output_dir,
            self.chapter_output_dir,
            self.memory_output_dir,
            self.review_dir,
            self.state_dir,
            self.memory_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _write_text(self, target: Path, content: str) -> Path:
        """Write a file atomically to reduce partial-write recovery issues."""
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        os.replace(temp_path, target)
        return target

    def _write_json(self, target: Path, payload: dict[str, object]) -> Path:
        """Serialize JSON atomically using UTF-8 and stable formatting."""
        return self._write_text(
            target,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    def persist_canon_asset(self, asset: CanonAsset, content: str, round_index: int) -> None:
        """Persist current and versioned canon files."""
        filename = self._asset_filename(asset)
        text = content.strip() + "\n"
        self._write_text(self.canon_current_dir / filename, text)
        history_name = f"{asset.value}_round_{round_index:02d}.md"
        self._write_text(self.canon_history_dir / history_name, text)

    def persist_review(self, asset: CanonAsset, feedback: list[str], round_index: int) -> None:
        """Persist review notes."""
        target = self.review_dir / f"{asset.value}_round_{round_index:02d}_review.md"
        lines = [f"# {asset.value} review round {round_index}", ""]
        lines.extend(f"- {item}" for item in (feedback or ["<empty>"]))
        self._write_text(target, "\n".join(lines).strip() + "\n")

    def persist_decision(self, asset: CanonAsset, decision: str, rationale: str, round_index: int) -> None:
        """Persist convergence decisions."""
        target = self.review_dir / f"{asset.value}_round_{round_index:02d}_decision.md"
        lines = [
            f"# {asset.value} convergence round {round_index}",
            "",
            f"- decision: {decision}",
            f"- rationale: {rationale.strip() or '<empty>'}",
        ]
        self._write_text(target, "\n".join(lines).strip() + "\n")

    def persist_chapter_decision(
        self,
        chapter_index: int,
        round_index: int,
        decision: str,
        rationale: str,
    ) -> Path:
        """Persist chapter-level convergence decisions."""
        target = self.review_dir / f"chapter_{chapter_index:03d}_round_{round_index:02d}_decision.md"
        lines = [
            f"# Chapter {chapter_index} convergence round {round_index}",
            "",
            f"- decision: {decision}",
            f"- rationale: {rationale.strip() or '<empty>'}",
        ]
        return self._write_text(target, "\n".join(lines).strip() + "\n")

    def persist_state(self, state: NovelProjectState) -> None:
        """Persist the full project state."""
        target = self.state_dir / "state_snapshot.json"
        self._write_json(target, state.to_dict())

    def persist_memory(self, state: NovelProjectState) -> None:
        """Persist the memory snapshot."""
        target = self.memory_dir / "memory_snapshot.json"
        self._write_json(target, state.memory_snapshot())

    def persist_run_manifest(self, state: NovelProjectState) -> Path:
        """Persist a compact operational manifest for resume and audit tooling."""
        target = self.state_dir / "run_manifest.json"
        payload = {
            "project_id": state.project_id,
            "title": state.title,
            "user_brief": state.user_brief,
            "mode": state.mode.value,
            "stage": state.stage.value,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "canon": {
                "story_outline_frozen": state.canon.story_outline.frozen,
                "character_profiles_frozen": state.canon.character_profiles.frozen,
                "chapter_outline_frozen": state.canon.chapter_outline.frozen,
                "chapter_count": len(state.canon.chapter_outline.chapters),
            },
            "draft": {
                "latest_chapter_index": state.latest_chapter_index(),
                "completed_chapters": state.draft.completed_chapters,
                "chapter_status": {
                    str(chapter): status
                    for chapter, status in sorted(state.draft.chapter_status.items())
                },
            },
            "memory": {
                "outline_items": len(state.memory.outline_memory),
                "character_items": len(state.memory.character_memory),
                "world_items": len(state.memory.world_memory),
                "review_items": len(state.memory.review_memory),
                "event_items": len(state.memory.event_memory),
            },
            "artifact_paths": self.artifact_paths(),
            "recent_stage_notes": state.stage_notes[-10:],
        }
        return self._write_json(target, payload)

    def persist_current_canon_bundle(self, state: NovelProjectState) -> None:
        """Persist the latest canon bundle even if no per-round history was written."""
        story_outline = state.canon.story_outline.current.strip()
        if story_outline:
            self._write_text(self.canon_current_dir / "story_outline.md", story_outline + "\n")

        character_profiles = state.canon.character_profiles.current.strip()
        if character_profiles:
            self._write_text(self.canon_current_dir / "character_profiles.md", character_profiles + "\n")

        chapter_outline = "\n".join(state.canon.chapter_outline.chapters).strip()
        if chapter_outline:
            self._write_text(self.canon_current_dir / "chapter_outline.md", chapter_outline + "\n")

    def persist_ideation_result(self, state: NovelProjectState) -> None:
        """Persist a human-readable canon bundle for the ideation stage."""
        chapter_outline = "\n".join(state.canon.chapter_outline.chapters).strip()
        lines = [
            f"# Ideation Result: {state.title}",
            "",
            f"project_id: {state.project_id}",
            f"mode: {state.mode.value}",
            f"stage: {state.stage.value}",
            "",
            "## User Brief",
            state.user_brief.strip(),
            "",
            "## Story Outline",
            state.canon.story_outline.current.strip() or "<empty>",
            "",
            "## Character Profiles",
            state.canon.character_profiles.current.strip() or "<empty>",
            "",
            "## Chapter Outline",
            chapter_outline or "<empty>",
            "",
            "## Recent Stage Notes",
        ]
        recent_notes = state.stage_notes[-10:] or ["<empty>"]
        lines.extend(f"- {note}" for note in recent_notes)
        lines.append("")
        lines.append("## Artifact Directories")
        paths = self.artifact_paths()
        lines.extend(f"- {key}: {value}" for key, value in paths.items())
        lines.append("")
        self._write_text(self.output_dir / "ideation_result.md", "\n".join(lines))

    def persist_chapter_draft(self, chapter_index: int, content: str, version: str = "draft") -> Path:
        """Persist a chapter draft or revision."""
        target = self.chapter_output_dir / f"chapter_{chapter_index:03d}_{version}.md"
        return self._write_text(target, content.strip() + "\n")

    def persist_chapter_review(self, chapter_index: int, feedback: list[str]) -> Path:
        """Persist chapter review findings."""
        target = self.chapter_output_dir / f"chapter_{chapter_index:03d}_review.md"
        lines = [f"# Chapter {chapter_index} Review", ""]
        lines.extend(f"- {item}" for item in feedback)
        return self._write_text(target, "\n".join(lines).strip() + "\n")

    def persist_chapter_working_notes(self, chapter_index: int, notes: list[str]) -> Path:
        """Persist process-only notes for a chapter revision cycle."""
        target = self.chapter_output_dir / f"chapter_{chapter_index:03d}_working_notes.md"
        lines = [f"# Chapter {chapter_index} Working Notes", ""]
        lines.extend(f"- {item}" for item in notes or ["<empty>"])
        return self._write_text(target, "\n".join(lines).strip() + "\n")

    def persist_memory_bundle(
        self,
        role: str,
        bundle: WorkingMemoryBundle,
        chapter_index: int | None = None,
    ) -> Path:
        """Persist the explicit task-scoped memory passed to a sub-agent."""
        suffix = f"_chapter_{chapter_index:03d}" if chapter_index is not None else ""
        target = self.memory_output_dir / f"{role}{suffix}.json"
        payload = {
            "role": role,
            "chapter_index": chapter_index,
            "task_context": bundle.task_context,
            "canon_context": bundle.canon_context,
            "relation_context": bundle.relation_context,
            "scene_cast_context": bundle.scene_cast_context,
            "narrative_context": bundle.narrative_context,
            "review_context": bundle.review_context,
            "planning_context": bundle.planning_context,
            "retrieval_index": bundle.retrieval_index,
        }
        return self._write_json(target, payload)

    def persist_memory_overview(self, state: NovelProjectState) -> Path:
        """Persist a concise snapshot of long-lived project memory."""
        target = self.memory_output_dir / "memory_overview.md"
        lines = [
            f"# Memory Overview: {state.title}",
            "",
            f"- project_id: {state.project_id}",
            f"- mode: {state.mode.value}",
            f"- stage: {state.stage.value}",
            "",
            "## Outline Memory",
        ]
        lines.extend(f"- {item}" for item in state.memory.outline_memory[-5:] or ["<empty>"])
        lines.extend(["", "## Character Memory"])
        lines.extend(f"- {item}" for item in state.memory.character_memory[-5:] or ["<empty>"])
        lines.extend(["", "## World Memory"])
        lines.extend(f"- {item}" for item in state.memory.world_memory[-5:] or ["<empty>"])
        lines.extend(["", "## Narrative History"])
        lines.extend(f"- {item}" for item in state.memory.narrative_history[-8:] or ["<empty>"])
        lines.extend(["", "## Event Memory"])
        recent_events = state.recent_memory_events(limit=10)
        if recent_events:
            lines.extend(
                f"- [{item.status}][{item.event_type}]"
                + (f"[第{item.chapter_index}章] " if item.chapter_index is not None else " ")
                + item.summary
                for item in recent_events
            )
        else:
            lines.append("- <empty>")
        lines.extend(["", "## Review Memory"])
        recent_reviews = state.recent_review_items(limit=8)
        if recent_reviews:
            lines.extend(
                f"- [{item.status}]"
                + (f"[第{item.chapter_index}章] " if item.chapter_index is not None else " ")
                + item.text
                for item in recent_reviews
            )
        else:
            lines.append("- <empty>")
        return self._write_text(target, "\n".join(lines).strip() + "\n")

    def scan_chapter_artifacts(self) -> dict[str, object]:
        """Inspect chapter artifacts and report chapters that need reruns."""
        chapter_numbers: set[int] = set()
        for path in self.chapter_output_dir.glob("chapter_*_*.md"):
            match = re.match(r"chapter_(\d+)_", path.name)
            if match:
                chapter_numbers.add(int(match.group(1)))

        issues: list[dict[str, object]] = []
        rerun_chapters: list[int] = []
        valid_completed: list[int] = []
        for chapter_index in sorted(chapter_numbers):
            draft_path = self.chapter_output_dir / f"chapter_{chapter_index:03d}_draft.md"
            review_path = self.chapter_output_dir / f"chapter_{chapter_index:03d}_review.md"
            revision_path = self.chapter_output_dir / f"chapter_{chapter_index:03d}_revision.md"
            problems: list[str] = []

            if not draft_path.exists():
                problems.append("missing_draft")
            elif self._looks_like_invalid_model_output(draft_path.read_text(encoding="utf-8")):
                problems.append("invalid_draft")

            if not review_path.exists():
                problems.append("missing_review")
            else:
                review_lines = [
                    line for line in review_path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")
                ]
                if not review_lines:
                    problems.append("empty_review")

            if not revision_path.exists():
                problems.append("missing_revision")
            elif self._looks_like_invalid_model_output(revision_path.read_text(encoding="utf-8")):
                problems.append("invalid_revision")

            if problems:
                rerun_chapters.append(chapter_index)
                issues.append({"chapter": chapter_index, "issues": problems})
            else:
                valid_completed.append(chapter_index)

        return {
            "total_detected": len(chapter_numbers),
            "valid_completed": valid_completed,
            "rerun_chapters": rerun_chapters,
            "issues": issues,
        }

    def persist_recovery_report(self, report: dict[str, object]) -> Path:
        """Persist a recovery scan report for resumable projects."""
        target = self.output_dir / "recovery_report.md"
        lines = [
            f"# Recovery Report: {self.project_id}",
            "",
            f"- total_detected: {report.get('total_detected', 0)}",
            f"- valid_completed: {len(report.get('valid_completed', []))}",
            f"- rerun_chapters: {', '.join(str(item) for item in report.get('rerun_chapters', [])) or '<none>'}",
            "",
            "## Issues",
        ]
        issues = report.get("issues", [])
        if issues:
            for item in issues:
                lines.append(
                    f"- chapter {item['chapter']}: {', '.join(item['issues'])}"
                )
        else:
            lines.append("- <none>")
        return self._write_text(target, "\n".join(lines).strip() + "\n")

    def artifact_paths(self) -> dict[str, str]:
        """Return key artifact locations."""
        return {
            "project_root": str(self.project_root.resolve()),
            "canon_current": str(self.canon_current_dir.resolve()),
            "outputs": str(self.output_dir.resolve()),
            "chapters": str(self.chapter_output_dir.resolve()),
            "memory": str(self.memory_output_dir.resolve()),
            "state": str(self.state_dir.resolve()),
        }

    @classmethod
    def load_project_state(
        cls,
        project_id: str,
        workspace_root: str | Path = "workspace/novel_projects",
    ) -> NovelProjectState:
        """Load a previously persisted project state for continuation."""
        project_root = Path(workspace_root).resolve() / project_id
        state_path = project_root / "state" / "state_snapshot.json"
        if state_path.exists():
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            return NovelProjectState.from_dict(payload)
        return cls._reconstruct_project_state(project_root, project_id)

    @classmethod
    def _reconstruct_project_state(cls, project_root: Path, project_id: str) -> NovelProjectState:
        """Best-effort recovery for older projects created before state snapshots existed."""
        canon_current = project_root / "canon" / "current"
        output_dir = project_root / "outputs"
        chapter_dir = output_dir / "chapters"
        ideation_path = output_dir / "ideation_result.md"
        manifest_path = project_root / "state" / "run_manifest.json"

        title = "Recovered Novel Project"
        user_brief = ""
        mode = "long_novel"
        stage = "drafting"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            title = str(manifest.get("title", title)) or title
            user_brief = str(manifest.get("user_brief", user_brief))
            mode = str(manifest.get("mode", mode)) or mode
            stage = str(manifest.get("stage", stage)) or stage
        if ideation_path.exists():
            text = ideation_path.read_text(encoding="utf-8")
            title_match = next((line.split(": ", 1)[1].strip() for line in text.splitlines() if line.startswith("# Ideation Result: ")), "")
            if title_match:
                title = title_match
            mode_match = next((line.split(": ", 1)[1].strip() for line in text.splitlines() if line.startswith("mode: ")), "")
            if mode_match:
                mode = mode_match
            marker = "## User Brief"
            if marker in text:
                after = text.split(marker, 1)[1].strip()
                user_brief = after.split("\n## ", 1)[0].strip()

        from .models import NovelMode, NovelStage  # local import to avoid cycles

        state = NovelProjectState(
            project_id=project_id,
            title=title,
            user_brief=user_brief or "恢复已有小说项目并继续续写。",
            mode=NovelMode(mode),
            stage=NovelStage(stage),
        )

        story_outline_path = canon_current / "story_outline.md"
        if story_outline_path.exists():
            content = story_outline_path.read_text(encoding="utf-8").strip()
            state.update_canon_asset(CanonAsset.STORY_OUTLINE, content)
            state.freeze_asset(CanonAsset.STORY_OUTLINE)
            if content:
                state.memory.outline_memory.append(content[:1000])

        character_profiles_path = canon_current / "character_profiles.md"
        if character_profiles_path.exists():
            content = character_profiles_path.read_text(encoding="utf-8").strip()
            state.update_canon_asset(CanonAsset.CHARACTER_PROFILES, content)
            state.freeze_asset(CanonAsset.CHARACTER_PROFILES)
            if content:
                state.memory.character_memory.append(content[:1000])

        chapter_outline_path = canon_current / "chapter_outline.md"
        if chapter_outline_path.exists():
            content = chapter_outline_path.read_text(encoding="utf-8").strip()
            state.update_canon_asset(CanonAsset.CHAPTER_OUTLINE, content)
            state.freeze_asset(CanonAsset.CHAPTER_OUTLINE)
            if content:
                state.memory.world_memory.append(content[:1000])

        chapter_numbers: set[int] = set()
        for path in chapter_dir.glob("chapter_*_*.md"):
            match = re.match(r"chapter_(\d+)_", path.name)
            if match:
                chapter_numbers.add(int(match.group(1)))

        for chapter_index in sorted(chapter_numbers):
            revision_path = chapter_dir / f"chapter_{chapter_index:03d}_revision.md"
            draft_path = chapter_dir / f"chapter_{chapter_index:03d}_draft.md"
            review_path = chapter_dir / f"chapter_{chapter_index:03d}_review.md"
            notes_path = chapter_dir / f"chapter_{chapter_index:03d}_working_notes.md"
            revision_valid = False
            draft_valid = False

            chosen_path = revision_path if revision_path.exists() else draft_path
            if chosen_path.exists():
                content = chosen_path.read_text(encoding="utf-8").strip()
                if content and not cls._looks_like_invalid_model_output(content):
                    state.store_chapter_draft(chapter_index, content)
                    if chosen_path == revision_path:
                        revision_valid = True
                    if chosen_path == draft_path:
                        draft_valid = True
            if not revision_valid and draft_path.exists():
                draft_content = draft_path.read_text(encoding="utf-8").strip()
                if draft_content and not cls._looks_like_invalid_model_output(draft_content):
                    state.store_chapter_draft(chapter_index, draft_content)
                    draft_valid = True

            if review_path.exists():
                review_lines = [
                    line[2:].strip()
                    for line in review_path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("- ")
                ]
                if review_lines:
                    state.add_review_feedback(chapter_index, review_lines)

            if notes_path.exists():
                note_lines = [
                    line[2:].strip()
                    for line in notes_path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("- ")
                ]
                for note in note_lines:
                    state.add_chapter_working_note(chapter_index, note)

            if revision_valid:
                state.mark_revision_complete(chapter_index)
            elif draft_valid:
                state.draft.chapter_status[chapter_index] = "drafted"

        state.stage_notes.append("Recovered project state from filesystem artifacts.")
        return state

    @staticmethod
    def _looks_like_invalid_model_output(text: str) -> bool:
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

    @staticmethod
    def _asset_filename(asset: CanonAsset) -> str:
        mapping = {
            CanonAsset.STORY_OUTLINE: "story_outline.md",
            CanonAsset.CHARACTER_PROFILES: "character_profiles.md",
            CanonAsset.CHAPTER_OUTLINE: "chapter_outline.md",
        }
        return mapping[asset]
