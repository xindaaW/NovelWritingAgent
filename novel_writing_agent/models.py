"""Shared models for the NovelWritingAgent framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NovelMode(str, Enum):
    """Supported creation modes."""

    SHORT = "short_story"
    LONG = "long_novel"


class NovelStage(str, Enum):
    """High-level stages controlled by the main agent."""

    IDEATION = "ideation"
    DRAFTING = "drafting"
    REVIEW = "review"
    REVISION = "revision"
    PUBLISHING = "publishing"
    COMPLETE = "complete"


class AgentRole(str, Enum):
    """Roles in the multi-agent writing system."""

    MAIN = "main"
    PREMISE = "premise"
    OUTLINE = "outline"
    CHARACTER = "character"
    CHAPTER_PLANNER = "chapter_planner"
    CREATIVE_REVIEWER = "creative_reviewer"
    CONVERGENCE = "convergence"
    WRITER = "writer"
    CHARACTER_REVIEWER = "character_reviewer"
    CONTINUITY_REVIEWER = "continuity_reviewer"
    STYLE_REVIEWER = "style_reviewer"
    CHAPTER_CONVERGENCE = "chapter_convergence"
    PUBLISHING = "publishing"


class AgentTaskKind(str, Enum):
    """Common task types passed between agents."""

    GENERATE = "generate"
    REVIEW = "review"
    REVISE = "revise"
    SUMMARIZE = "summarize"
    PACKAGE = "package"
    DECIDE = "decide"


class CanonAsset(str, Enum):
    """Semi-static canon artifacts created before drafting."""

    STORY_OUTLINE = "story_outline"
    CHARACTER_PROFILES = "character_profiles"
    CHAPTER_OUTLINE = "chapter_outline"


class DeviationAction(str, Enum):
    """Resolution strategy when drafting deviates from canon."""

    UPDATE_WRITER = "update_writer"
    UPDATE_CANON = "update_canon"


@dataclass(slots=True)
class AgentExecutionRequest:
    """Task envelope sent from the main agent to a sub-agent."""

    role: AgentRole
    task_kind: AgentTaskKind
    objective: str
    context: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    expected_output: str = ""
    target_asset: CanonAsset | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentExecutionResult:
    """Structured response returned by a sub-agent."""

    role: AgentRole
    task_kind: AgentTaskKind
    output: str
    structured_output: dict[str, Any] = field(default_factory=dict)
    feedback: list[str] = field(default_factory=list)
    confidence: float = 0.0
    should_update_canon: bool = False
    canon_update_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
