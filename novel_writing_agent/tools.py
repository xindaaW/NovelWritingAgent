"""Novel-writing specific tools."""

from __future__ import annotations

from .tool_base import Tool, ToolResult


class RetrieveMemoryTool(Tool):
    """Progressive memory retrieval tool for writing agents."""

    def __init__(self, retrieval_index: dict[str, dict[str, list[dict[str, str]]]]):
        self.retrieval_index = retrieval_index

    @property
    def name(self) -> str:
        return "retrieve_memory"

    @property
    def description(self) -> str:
        return "按层级检索小说 memory。先取 immediate，再按需要取 contextual 或 deep 中和当前情节最相关的 canon、character、narrative、review 记忆。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["immediate", "contextual", "deep"],
                    "description": "要检索的 memory 层级。",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["canon", "character", "narrative", "review"],
                    "description": "要检索的 memory 类型。",
                },
                "focus": {
                    "type": "string",
                    "description": "当前情节或问题的焦点关键词，用于缩小返回结果。",
                },
                "limit": {
                    "type": "integer",
                    "description": "最多返回多少条 memory。",
                    "minimum": 1,
                    "maximum": 8,
                },
                "reveal": {
                    "type": "string",
                    "enum": ["summary", "detail"],
                    "description": "默认只返回事件摘要；只有在需要深挖时才返回 detail。",
                },
            },
            "required": ["level", "memory_type"],
        }

    async def execute(
        self,
        level: str,
        memory_type: str,
        focus: str = "",
        limit: int = 4,
        reveal: str = "summary",
    ) -> ToolResult:
        pool = self.retrieval_index.get(level, {}).get(memory_type, [])
        if not pool:
            return ToolResult(success=True, content="未检索到可用 memory。")

        focus = (focus or "").strip().lower()
        items = pool
        if focus:
            ranked = [
                item
                for item in pool
                if focus in item.get("summary", "").lower()
                or focus in item.get("body", "").lower()
                or focus in item.get("source", "").lower()
            ]
            if ranked:
                items = ranked

        chosen = items[: max(1, min(limit, 8))]
        lines = [f"[{level}/{memory_type}/{reveal}]"]
        for item in chosen:
            summary = item.get("summary", "").strip()
            body = item.get("body", "").strip()
            source = item.get("source", "").strip()
            if reveal == "detail":
                lines.append(f"- 摘要：{summary}")
                if source:
                    lines.append(f"  来源：{source}")
                if body:
                    lines.append(f"  正文：{body}")
            else:
                bullet = f"- {summary}"
                if source:
                    bullet += f" [{source}]"
                lines.append(bullet)
        return ToolResult(success=True, content="\n".join(lines))
