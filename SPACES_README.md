---
title: OpenEnv DataCleaning
emoji: 🧹
colorFrom: cyan
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
  - agent-benchmark
  - real-world
license: mit
---

# OpenEnv · DataCleaning Environment

A real-world data cleaning and validation environment for AI agent training and evaluation.

See [README.md](README.md) for full documentation.

## Quick API Reference

- `POST /reset?task_id=task_easy&seed=42` — Reset environment
- `POST /step` — Apply a cleaning action  
- `GET /state` — Current environment state
- `GET /tasks` — List all tasks
- `GET /health` — Health check
