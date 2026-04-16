# Coalesced Global Memory Access Backbone

This repository provides a **core backbone** for one of the most proven CUDA optimization techniques:

**coalesced global memory access**.

The goal is to show, with actual CUDA kernels, how warp address patterns change the memory-transaction behavior of global memory accesses.

## What this code covers

This scaffold includes three CUDA kernels:

- **coalesced copy**
  - thread k reads word k

- **misaligned copy**
  - same sequential pattern, but shifted by a small offset

- **strided copy**
  - thread k reads word k × stride

These three patterns are enough to demonstrate the main coalescing principle.

## Core idea

A warp’s memory cost is dominated by how many memory transactions the hardware must issue to serve that warp.

Conceptually:

```text
T_mem ∝ N_tx
```
