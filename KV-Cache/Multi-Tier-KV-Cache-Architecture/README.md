# Multi-tier KV Cache Demo

This is a minimal runnable demo showing the core idea behind a multi-tier KV cache architecture for long-context, high-concurrency LLM serving.

It simulates:

- hot / warm / cold KV tiers
- GPU / CPU / SSD placement
- promotion and demotion between tiers
- prefix-aware routing
- reuse-aware eviction

## What this is

A lightweight simulation of KV placement and routing policy.

## What this is not

This is not a real inference engine. It does not use CUDA, vLLM, SGLang, or real KV tensors.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
