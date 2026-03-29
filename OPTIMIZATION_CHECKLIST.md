# Optimization Checklist

This checklist tracks the post-open-source optimization path for `NovelWritingAgent`.

## Phase 0: Open-Source Readiness

- [x] Split the project into a standalone open-source directory
- [x] Keep `requirements.txt` as the single dependency source
- [x] Add a public OSS license
- [x] Clean generated artifacts from the repository surface
- [ ] Create the public GitHub repository and push `main`

## Phase 1: Robustness and Operability

- [x] Make artifact persistence atomic to reduce partial-write recovery issues
- [x] Add prompt-side memory budgets so long projects do not explode context size
- [x] Support JSON-first parsing for reviewer and convergence outputs while keeping freeform fallback
- [x] Add regression tests for structured parsing, prompt budgeting, and artifact persistence
- [x] Add CI to run tests on push / PR
- [x] Add repository issue templates and pull-request template

## Phase 2: Memory and Canon Quality

- [ ] Add embedding-based retrieval instead of relying only on heuristics and LLM selectors
- [ ] Add long-horizon memory compression / summarization to control chapter growth
- [ ] Introduce explicit token / prompt budgets instead of only character budgets
- [ ] Upgrade canon patching from text notes to structured section-level updates
- [ ] Add chapter-risk scoring to decide when deep review is required

## Phase 3: Multi-Agent Collaboration

- [ ] Add a meta-reviewer that merges character / continuity / style findings
- [ ] Let reviewers cross-reference one another instead of acting as isolated critics
- [ ] Add selective parallelism for independent reviewer steps
- [ ] Add explicit cost / latency budgets per stage and per chapter
- [ ] Strengthen the publishing stage beyond placeholder packaging behavior

## Phase 4: Productization

- [ ] Add sample projects and reproducible demo runs
- [ ] Add architecture diagrams and state / memory flow docs
- [ ] Add a CLI entry point instead of relying only on example modules
- [x] Add structured run manifests for safer resume and audit trails
