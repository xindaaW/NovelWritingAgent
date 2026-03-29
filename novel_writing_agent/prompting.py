"""Prompt loading helpers for NovelWritingAgent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PromptBundle:
    """Prompt material for a sub-agent."""

    system_prompt: str
    skill_text: str


def load_skill_prompt(skill_dir: Path | None, fallback_prompt: str) -> PromptBundle:
    """Load a skill markdown file and merge it into the system prompt."""
    skill_text = ""
    if skill_dir:
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            skill_text = skill_file.read_text(encoding="utf-8").strip()

    system_parts = [fallback_prompt.strip()]
    if skill_text:
        system_parts.append("角色技能说明：\n" + skill_text)
    return PromptBundle(
        system_prompt="\n\n".join(system_parts),
        skill_text=skill_text,
    )
