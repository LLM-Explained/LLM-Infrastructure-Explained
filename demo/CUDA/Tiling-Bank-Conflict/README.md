# Shared-Memory Tiling + Bank-Conflict Avoidance Backbone

This repository provides a **core backbone** for one of the most proven CUDA optimization patterns:

**shared-memory tiling plus bank-conflict avoidance**.

It demonstrates two key ideas together:

1. stage data into shared memory so global-memory traffic is reused
2. choose the shared-memory layout so warp accesses do not serialize on the same bank

## What this code covers

This scaffold includes two CUDA transpose-style kernels:

- **transpose_no_padding**
  - shared-memory tile with no padding
  - classic bank-conflict pattern on the transposed read

- **transpose_with_padding**
  - shared-memory tile with `TILE_DIM + 1` stride
  - same tiling idea, but with bank-conflict avoidance

## Core idea

The optimization has two parts.

### Part 1: shared-memory tiling

Tiling reduces repeated global-memory traffic by loading a tile once and reusing it locally.

Conceptually:

```text
Global traffic without tiling ∝ R × B
Global traffic with tiling ∝ B
```
