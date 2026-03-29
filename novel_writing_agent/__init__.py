"""NovelWritingAgent framework.

This package provides the first-pass framework for a multi-agent novel writing
system. It keeps the orchestration layer separate from concrete agent prompts,
skills, and runtime integrations so the project can evolve iteratively.
"""

from .agent import NovelMainAgent
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

__all__ = [
    "AgentExecutionRequest",
    "AgentExecutionResult",
    "AgentRegistry",
    "AgentRole",
    "AgentTaskKind",
    "CanonAsset",
    "DeviationAction",
    "MemoryOrchestrator",
    "NovelMainAgent",
    "NovelMode",
    "NovelProjectState",
    "NovelStage",
]
