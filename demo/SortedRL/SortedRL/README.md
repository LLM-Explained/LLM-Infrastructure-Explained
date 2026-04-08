# SortedRL Scheduling Backbone

This repository provides a **core backbone** for the main systems idea behind **SortedRL**:

RL training efficiency for reasoning LLMs can improve when rollout samples are scheduled by output length rather than consumed in naive order.

## What this code covers

This scaffold includes the major algorithmic pieces:

- rollout trajectories with heterogeneous lengths
- naive grouping
- online length-aware grouping
- a bubble-ratio proxy for synchronized idle time
- a simple policy-lag proxy for data freshness

## Core algorithm

The backbone implemented here is:

1. generate a rollout batch with different trajectory lengths
2. reorder the trajectories by length
3. group shorter trajectories for earlier updates
4. compare utilization and freshness against naive grouping

This captures the main SortedRL idea:
**use length-aware scheduling to reduce rollout/update pipeline inefficiency while keeping updates fresher.**

## Why this backbone is useful

The important part is not the exact RL objective.

The important part is the **training pipeline structure**:

- long trajectories create stragglers
- stragglers create idle bubbles
- scheduling policy changes how much of that waste appears
- update timing affects policy freshness

That is why SortedRL is interesting: it treats rollout scheduling as a central part of RL training quality and efficiency.

## What this code is

A **foundational prototype** for online length-aware rollout scheduling in RL training.

It is designed to make the scheduling logic concrete in the smallest runnable form.

## Scope

This repository does not reproduce:

- the full SortedRL trainer
- the stateful controller
- the cache-based off-policy control mechanism
- large-model rollout infrastructure
- benchmark evaluation on reasoning datasets

Instead, it isolates the main algorithmic design choice in a concise runnable scaffold.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
