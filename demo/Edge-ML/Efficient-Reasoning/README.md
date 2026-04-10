# Efficient Reasoning on the Edge Backbone

This repository provides a **core backbone** for the main system design behind **Efficient Reasoning on the Edge**:

reasoning on-device should be **conditional, concise, and budget-aware** rather than always-on.

## What this code covers

This scaffold includes the major architectural and algorithmic pieces:

- a **frozen backbone**
- a **shared prompt-encoding cache**
- a **fast adapter**
  - lightweight query typing
  - concise answer generation
- a **reasoning adapter**
  - problem classification
  - subgoal construction
  - iterative reasoning refinement
  - concise final synthesis
- **dynamic adapter switching**
- a **budget-forced reasoning loop**

## Core architecture

The backbone implemented here is:

1. encode the prompt once into a reusable cache
2. build a shared backbone state from prompt cache + query
3. choose between:
   - a **fast path** for cheap/direct queries
   - a **reasoning path** for harder multi-step queries
4. if reasoning is needed:
   - classify the problem
   - build subgoals
   - iteratively refine reasoning under a fixed budget
   - synthesize a concise final answer

Conceptually:

```text
prompt -> shared cache
query  -> route to fast adapter or reasoning adapter
reasoning adapter -> bounded multi-step refinement
```
