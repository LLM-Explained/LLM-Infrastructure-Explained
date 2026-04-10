# HillInfer Backbone: SmartSSD-Assisted Hierarchical KV Management

This repository provides a **core backbone** for the main systems idea behind **HillInfer**:

long-context edge LLM inference should manage KV cache as a coordinated **GPU–CPU–SmartSSD hierarchy**, with lightweight importance evaluation happening near storage and adaptive prefetching moving useful blocks closer to compute.

## What this code covers

This scaffold includes the major architectural pieces:

- **importance-aware KV scoring**
- **hierarchical placement**
  - GPU
  - CPU
  - SmartSSD
- **eviction / demotion**
  - hot blocks stay close
  - colder blocks move down the hierarchy
- **adaptive prefetch**
  - likely-needed blocks are pulled upward before use

## Core architecture

The backbone implemented here is:

1. evaluate KV block importance with a lightweight storage-side score
2. place high-importance blocks closer to compute
3. demote lower-importance blocks to lower tiers
4. prefetch predicted future blocks upward through the hierarchy

Conceptually:

```text
GPU    <- hottest actively useful KV
CPU    <- warm near-term KV
SSD    <- colder but still reusable KV
```
