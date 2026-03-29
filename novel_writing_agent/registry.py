"""Registration and lookup for novel writing sub-agents."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import AgentRole


class AgentRegistry:
    """Simple in-memory registry for sub-agent instances."""

    def __init__(self) -> None:
        self._agents: dict[AgentRole, object] = {}
        self._groups: dict[str, list[AgentRole]] = defaultdict(list)

    def register(self, role: AgentRole, agent: object, group: str) -> None:
        """Register an agent under a role and group."""
        self._agents[role] = agent
        if role not in self._groups[group]:
            self._groups[group].append(role)

    def get(self, role: AgentRole) -> object:
        """Fetch a registered agent by role."""
        return self._agents[role]

    def roles_in_group(self, group: str) -> list[AgentRole]:
        """Return the roles registered in a group."""
        return list(self._groups[group])

    def items(self) -> Iterable[tuple[AgentRole, object]]:
        """Iterate over all registered agents."""
        return self._agents.items()
