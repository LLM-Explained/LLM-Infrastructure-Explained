# Atrex-Style Trace-Weighted Roofline Scoring

This small, CPU-only demo reproduces the **central evaluation idea** from *Are LLM-Generated GPU Kernels Production-Ready? A Trace-Driven Benchmark and Optimization Agent* (Atrex-Bench): a generated kernel should be judged not only by whether it is correct, but by how much of the hardware roofline it reaches **on the operators that consume production GPU time**.

The full Atrex release evaluates 30 production-derived operators and 440 shapes using real GPU compilation, numerical checks, profiling, hidden provenance, and per-device roofline artifacts. This demo intentionally does **not** reproduce that expensive stack. It isolates the metric and demonstrates the ranking behavior in a deterministic experiment that runs with the Python standard library.

## Core idea

For a shape `j`, the speed-of-light latency is the slower of its compute and memory lower bounds:

```text
T_roofline,j = max(F_j / P_peak, M_j / B_peak)
```

A correct candidate receives:

```text
S_j = T_roofline,j / T_candidate,j
```

Shapes are summarized by the median within each operator. Operator scores are then weighted by the operator's production GPU-time share:

```text
S_aggregate = sum_i weight_i * S_i
```

A failed operator contributes zero. A measured latency below the semantic roofline is treated as an evaluation error rather than silently clipped.

## What the experiment shows

`example.py` compares two synthetic kernel agents:

- **Fallback-friendly** is correct on every operator, but is slow and barely uses the target DSL on production-heavy attention and MoE kernels.
- **Production-focused** misses one rare tail operator, but writes much faster native kernels for attention and MoE.

Raw correctness prefers the first agent. Production-weighted roofline achievement prefers the second. This is the paper's key deployment lesson in miniature: **correctness is necessary, but it is not a proxy for deployability**.

## Layout

```text
Atrex-Trace-Weighted-Roofline-Scoring/
├── README.md
├── ARTICLE_DRAFT.md
├── Makefile
├── requirements.txt
├── atrex_mini.py
├── example.py
└── tests/
    └── test_atrex_mini.py
```

## Run

```bash
cd demo/Benchmarking/Atrex-Trace-Weighted-Roofline-Scoring
python example.py
python -m unittest discover -s tests -v
```

Or:

```bash
make demo
make test
```

No GPU or third-party package is required.

## Expected output

The exact values are deterministic. The important result is the ranking reversal:

```text
Raw-correctness winner:   Fallback-friendly
Production-score winner: Production-focused
```

The output also reports target-DSL share, a simplified proxy for the paper's observation that apparent correctness can come from PyTorch or precompiled fallbacks rather than a kernel written in the requested DSL.

## Baseline and ablation

The demo exposes three views of the same candidates:

1. **Correctness rate** — ignores latency and production importance.
2. **Uniform operator score** — adds roofline efficiency but weights every operator equally.
3. **Production-weighted score** — emphasizes kernels according to deployed GPU-time share.

Comparing these views is the ablation: removing trace weights changes what the benchmark rewards.

## What this validates

This demo validates that:

- roofline-normalized performance is comparable across compute- and memory-bound shapes;
- correctness gating prevents failed kernels from receiving performance credit;
- per-operator aggregation avoids rewarding an operator merely because it has many shapes;
- production weights can reverse rankings relative to raw correctness or uniform averaging;
- target-DSL adoption should be inspected separately from correctness.

## What this does not validate

It does not reproduce:

- Alibaba's production traces or importance weights;
- the 30-operator, 440-shape Atrex-Bench corpus;
- FlyDSL, Triton, Gluon, or CuteDSL compilation;
- GPU timing, L2 cache flushing, frequency locking, profiler counters, or ISA analysis;
- Atrex-Kernel-Agent's measure-revise loop, optimization dropout, or knowledge base;
- the paper's reported model rankings or speedups.

Use the official repositories for full experiments.

## Article

A publication-ready deep-dive is included in [`ARTICLE_DRAFT.md`](ARTICLE_DRAFT.md). It is intended for the **LLM Infrastructure Explained** section of the Notion notebook.

## Sources

- Paper: https://arxiv.org/abs/2607.14541
- Atrex-Bench: https://github.com/alibaba/atrex-bench
- Atrex-Kernel-Agent: https://github.com/alibaba/atrex-kernel-agent
- Notion notebook: https://app.notion.com/p/calvinfei/LLM-Research-Explained-16e2d34d69a88077a7c4cc1a24f47041

The standalone Notion article could not be linked during the initial publication because authenticated Notion access was unavailable to the automation runtime.
