"""Example 7: NovelWritingAgent framework skeleton."""

import asyncio
from datetime import datetime
from pathlib import Path

from novel_writing_agent import NovelMainAgent, NovelMode, NovelProjectState
from novel_writing_agent.ui import Colors


async def main() -> None:
    """Run one skeleton pass of the NovelWritingAgent framework."""
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_id = f"demo_short_{run_stamp}"
    workspace_root = Path("workspace/novel_projects").resolve()
    project_root = workspace_root / project_id
    project_root.mkdir(parents=True, exist_ok=True)
    log_path = project_root / f"run_{run_stamp}.log"
    short_story_state = NovelProjectState(
        project_id=project_id,
        title="Unnamed Short Story",
        user_brief="写一个关于未来都市中孤独感的短篇小说。",
        mode=NovelMode.SHORT,
    )
    agent = NovelMainAgent(
        state=short_story_state,
        verbose=True,
        log_path=log_path,
        artifact_workspace_root=workspace_root,
    )

    print(
        agent.ui.banner(
            "NovelWritingAgent Skeleton Demo",
            f"Progress log: {log_path}",
        )
    )
    print()
    snapshot = agent.clean_context()
    print(
        agent.ui.summary(
            "Clean Main-Agent Context",
            [
                f"project_id: {snapshot['project_id']}",
                f"title: {snapshot['title']}",
                f"mode: {snapshot['mode']}",
                f"stage: {snapshot['stage']}",
                f"ideation_max_rounds: {snapshot['policy']['ideation_max_rounds']}",
                f"artifacts: {snapshot['artifact_paths']['project_root']}",
                f"result: {snapshot['artifact_paths']['outputs']}/ideation_result.md",
            ],
            color=Colors.BRIGHT_CYAN,
        )
    )
    print()
    outputs = await agent.run_stage()
    print()
    print(
        agent.ui.summary(
            "Stage Outputs",
            outputs,
            color=Colors.BRIGHT_GREEN,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
