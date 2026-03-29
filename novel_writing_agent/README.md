# NovelWritingAgent

`NovelWritingAgent` is a standalone multi-agent fiction-writing system.

## What It Does

- Keeps the main-agent context compact.
- Separates canon ideation from chapter drafting.
- Uses one writer agent for prose generation and revision.
- Adds reviewer agents for character, continuity, and style checks.
- Persists long-term memory, project state, and chapter artifacts for resume.

## Current Status

This package already includes:

- state models and stage policies
- ideation, drafting, review, and revision orchestration
- memory orchestration and progressive retrieval tooling
- LLM wrappers for Anthropic-style and OpenAI-style APIs
- skill-driven sub-agent prompts with a skeleton fallback mode

The runnable entry points live under `examples/`, and the top-level `README.md` describes setup and execution.
