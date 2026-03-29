"""Example 8: Run or resume NovelWritingAgent with a real LLM."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from novel_writing_agent.config import Config
from novel_writing_agent.llm import LLMClient
from novel_writing_agent import NovelMainAgent, NovelMode, NovelProjectState, NovelStage
from novel_writing_agent.storage import ProjectArtifactStore
from novel_writing_agent.ui import Colors
from novel_writing_agent.schema import LLMProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume NovelWritingAgent with a configured LLM.")
    parser.add_argument("--resume-project-id", dest="resume_project_id", help="Resume an existing project id.")
    parser.add_argument("--brief", dest="brief", help="User brief when starting a new project.")
    parser.add_argument("--title", dest="title", help="Project title when starting a new project.")
    return parser.parse_args()


async def main() -> None:
    """Run the NovelWritingAgent with a configured LLM."""
    args = parse_args()
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("config.yaml not found.")
        print("Create it from config/config-example.yaml first.")
        return

    config = Config.from_yaml(config_path)
    llm_client = LLMClient(
        api_key=config.llm.api_key,
        api_base=config.llm.api_base,
        model=config.llm.model,
        provider=LLMProvider(config.llm.provider),
        retry_config=config.llm.retry,
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_project_id:
        state = ProjectArtifactStore.load_project_state(args.resume_project_id)
        project_id = state.project_id
    else:
        project_id = f"novel_live_demo_{run_stamp}"
        state = NovelProjectState(
            project_id=project_id,
            title=args.title or "NovelWritingAgent Demo",
            user_brief=args.brief
            or "一个被视为废柴的少年，在失去一切后凭借隐忍、机缘与成长一路逆袭，最终踏上强者之巅，并夺回尊严、爱情与命运的主导权。以此为主题写一个短篇小说。",
            mode=NovelMode.SHORT,
        )

    workspace_root = Path("workspace/novel_projects").resolve()
    project_root = workspace_root / project_id
    project_root.mkdir(parents=True, exist_ok=True)
    log_path = project_root / f"run_{run_stamp}.log"
    agent = NovelMainAgent(
        state=state,
        llm_client=llm_client,
        verbose=True,
        log_path=log_path,
        artifact_workspace_root=workspace_root,
    )
    recovery_report = agent.artifact_store.scan_chapter_artifacts()
    recovery_report_path = agent.artifact_store.persist_recovery_report(recovery_report)

    print(
        agent.ui.banner(
            "NovelWritingAgent Live LLM Demo",
            f"Progress log: {log_path}",
        )
    )
    print()
    print(agent.ui.section("Running NovelWritingAgent With Configured LLM"))
    print()
    print(
        agent.ui.summary(
            "Artifact Paths",
            [
                f"project_root: {agent.artifact_store.project_root}",
                f"canon_current: {agent.artifact_store.canon_current_dir}",
                f"outputs: {agent.artifact_store.output_dir}",
                f"state: {agent.artifact_store.state_dir / 'state_snapshot.json'}",
                f"memory: {agent.artifact_store.memory_dir / 'memory_snapshot.json'}",
                f"recovery_report: {recovery_report_path}",
                f"log: {log_path}",
            ],
            color=Colors.BRIGHT_CYAN,
        )
    )
    if args.resume_project_id:
        print()
        print(
            agent.ui.summary(
                "Recovery Scan",
                [
                    f"valid_completed: {len(recovery_report['valid_completed'])}",
                    f"rerun_chapters: {', '.join(str(item) for item in recovery_report['rerun_chapters']) or '<none>'}",
                ]
                + [
                    f"chapter {item['chapter']}: {', '.join(item['issues'])}"
                    for item in recovery_report["issues"][:12]
                ],
                color=Colors.BRIGHT_YELLOW,
            )
        )

    chapter_run_limit = 0

    if not state.canon_ready():
        outputs = await agent.run_stage()
        agent.persist_project_state()
        print()
        print(agent.ui.summary("Stage Outputs", outputs, color=Colors.BRIGHT_GREEN))
        print()

    if state.canon_ready():
        chapter_count = max(1, len(state.canon.chapter_outline.chapters))
        invalid_chapters = set(recovery_report["rerun_chapters"]) if args.resume_project_id else set()
        if invalid_chapters:
            for chapter_index in invalid_chapters:
                state.draft.completed_chapters = [idx for idx in state.draft.completed_chapters if idx != chapter_index]
                state.draft.chapter_status[chapter_index] = "needs_redraft"
        chapter_run_limit = max(0, chapter_count - len(state.draft.completed_chapters))
        while len(state.draft.completed_chapters) < chapter_count:
            current_chapter = len(state.draft.completed_chapters) + 1
            if current_chapter in invalid_chapters:
                state.draft.chapter_drafts.pop(current_chapter, None)
                state.draft.chapter_status[current_chapter] = "drafting"
            if current_chapter not in state.draft.chapter_drafts:
                state.stage = NovelStage.DRAFTING
                stage_outputs = await agent.run_stage()
                agent.persist_project_state()
                print()
                print(agent.ui.summary("Drafting Outputs", stage_outputs, color=Colors.BRIGHT_MAGENTA))
                print()

            while current_chapter not in state.draft.completed_chapters:
                for stage in (NovelStage.REVIEW, NovelStage.REVISION):
                    state.stage = stage
                    stage_outputs = await agent.run_stage()
                    agent.persist_project_state()
                    print()
                    print(agent.ui.summary(f"{stage.value.title()} Outputs", stage_outputs, color=Colors.BRIGHT_MAGENTA))
                    print()
                if current_chapter in state.draft.completed_chapters:
                    break
                agent.log_progress(
                    f"Chapter {current_chapter} requires another review-revision cycle; continuing autonomously.",
                    label="Decision",
                    color=Colors.BRIGHT_YELLOW,
                )

    print(
        agent.ui.summary(
            "Run Scope",
            [
                f"Remaining chapters processed this run: {chapter_run_limit}",
                f"Final ideation bundle: {agent.artifact_store.output_dir / 'ideation_result.md'}",
                f"Chapter outputs: {agent.artifact_store.chapter_output_dir}",
                f"Memory outputs: {agent.artifact_store.memory_output_dir}",
                f"State snapshot: {agent.artifact_store.state_dir / 'state_snapshot.json'}",
                f"Long-term memory snapshot: {agent.artifact_store.memory_dir / 'memory_snapshot.json'}",
            ],
            color=Colors.BRIGHT_YELLOW,
        )
    )

    print("\nCanon snapshot:")
    print(agent.ui.preview("Story outline", state.canon.story_outline.current[:800] or "<empty>", max_lines=10))
    print()
    print(
        agent.ui.preview(
            "Character profiles",
            state.canon.character_profiles.current[:800] or "<empty>",
            color=Colors.BRIGHT_MAGENTA,
            max_lines=10,
        )
    )
    print()
    print(
        agent.ui.preview(
            "Chapter outline",
            "\n".join(state.canon.chapter_outline.chapters[:10]) or "<empty>",
            color=Colors.BRIGHT_GREEN,
            max_lines=10,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
