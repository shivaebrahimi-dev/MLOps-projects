# Agentic RAG Knowledge Assistant

Work in progress for MLOps Homework 4: a Retrieval-Augmented Generation assistant built spec-first with Spec Kit,
answered by a CrewAI crew, and traced in Langfuse. The full README is written in the last phase.

The assignment text is in [homework04-assignment.md](homework04-assignment.md).

## Agent skills installed in this project

| Skill | Source | Install command |
|---|---|---|
| Spec Kit (`speckit-*`) | [github/spec-kit](https://github.com/github/spec-kit) v1.2.5 | `specify init --here --ai claude --script ps` |
| CrewAI (`getting-started`, `design-agent`, `design-task`, `test-crewai-project`, ...) | [docs.crewai.com/en/skills](https://docs.crewai.com/en/skills) | `npx skills add crewaiinc/skills` |
| Langfuse (`langfuse`) | [langfuse.com agent skill](https://langfuse.com/docs/api-and-data-platform/features/agent-skill) | `npx skills add langfuse/skills --skill langfuse` |

The installed versions are recorded in `skills-lock.json`.
