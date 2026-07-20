# PagedWeight: Treating MoE Precision as a Runtime Memory Resource

## Executive Summary

Mixture-of-Experts models reduce active computation by routing each token through only a subset of experts, but the complete expert-weight set still competes with a growing KV cache for GPU memory. Conventional deployment chooses one weight precision before the server starts, so it cannot respond when concurrency, sequence length, or expert usage changes.

PagedWeight reframes expert precision as a runtime-managed resource. It stores expert linear blocks in a representation that can expose different bitwidths, predicts the quality cost of reducing each block, and dynamically trades selected weight pages for KV-cache capacity. Its controller combines offline sensitivity with online expert-routing and prompt-conditioned signals, while its runtime coordinates asynchronous page movement and mixed-precision execution.

**Reported result:** The paper reports FP16-equivalent accuracy with up to 72.0% GPU-memory savings and 1.94x higher throughput. At a similar memory budget, it reports up to 39.3% higher quality than quantization baselines with at most 4.1% throughput loss.

**Interpretation:** The main contribution is not merely another low-bit format. It is a systems abstraction in which model weights and KV cache become elastic memory tenants, and precision becomes a scheduling decision rather than a deployment constant.

## What Was Released

PagedWeight was submitted to arXiv on July 17, 2026 by Yuchen Yang, Yifan Zhao, Anisha Dasgupta, and Sasa Misailovic. The work presents:

1. a runtime weight-memory manager for MoE serving;
2. an Any-Precision-style representation that permits expert blocks to be materialized at multiple bitwidths;
3. a quality-aware planner that decides which precision transitions to apply;
4. online importance signals derived from expert routing and prompt-conditioned residual behavior;
5. asynchronous CPU/GPU page movement with safe state transitions;
6. fused execution support for experts whose blocks may use different precisions;
7. experiments on Qwen1.5-MoE-A2.7B, Mixtral, and Gemma-4-26B-A4B-class models on RTX 6000 Ada and GH200 systems.

The work belongs under **LLM Infrastructure Explained** because its primary contribution is dynamic memory management and mixed-precision serving rather than a new model architecture or training objective.

## Problem and Motivation

A model server has at least two major GPU-memory tenants:

- **weights**, which are normally fixed after loading;
- **KV cache**, which grows with active sequences and context length.

MoE models add a useful asymmetry. Only a few experts execute for each token, yet future routing decisions are input-dependent, so the server generally keeps all expert weights available. A fixed quantization policy must select one compromise among quality, cache capacity, and kernel performance for every workload.

That compromise is often wrong in both directions. Static low-bit quantization may sacrifice quality during light load, when spare memory could support higher precision. Static high precision can block long-context requests or reduce sustainable concurrency during bursts. Even a static per-expert allocation cannot react to which experts current prompts actually use.

PagedWeight therefore asks: **Can a server resize routed-expert weights at runtime, much as virtual memory changes page residency, while protecting the blocks most important to active requests?**

## Core Technical Idea

For a legal precision transition on expert block \(i\), from bitwidth \(b\) to a lower bitwidth \(b'\), the idealized memory release is

\[
\Delta M_i = \frac{N_i(b-b')}{8},
\]

where \(N_i\) is the block's parameter count. The planner also predicts a quality cost \(\Delta Q_i\). Conceptually, it prefers transitions with a small ratio

\[
\rho_i = \frac{\Delta Q_i}{\Delta M_i},
\]

and keeps selecting low-cost transitions until enough memory is available for the KV cache.

The quality estimate combines three forms of evidence:

- **global sensitivity:** an offline estimate of how vulnerable a block is to quantization;
- **routing importance:** recently hot experts receive greater protection;
- **prompt-conditioned evidence:** current residual behavior indicates when a block is more important than its global average suggests.

Two equal-size experts can therefore receive different decisions. A cold, low-sensitivity expert may move from 16 to 8 to 4 bits, while a hot expert remains at 16 bits. When requests finish and memory pressure decreases, the system can restore the pages with the greatest expected quality benefit per byte.

## Architecture or System Design

PagedWeight can be understood as five cooperating layers.

### 1. Multi-precision weight representation

Expert linear blocks are represented with bit-plane-like pages so that lower- and higher-precision forms can be exposed without storing a completely independent tensor for every bitwidth. This makes precision changes compatible with memory paging.

### 2. Offline sensitivity profiler

Before serving, the system estimates how damaging precision reduction is for each block. This prior prevents the runtime controller from treating all reclaimable bytes as equally safe.

### 3. Online importance monitor

During inference, routing probabilities or token assignments identify the experts currently receiving traffic. Prompt-conditioned residual statistics add request-specific evidence.

### 4. Memory-pressure planner

When the KV-cache allocator asks for more space, the planner enumerates legal precision changes, predicts quality cost and released bytes, respects per-block floors, and produces a transition plan. A complementary restoration policy uses newly available headroom to raise precision.

### 5. Transactional movement and mixed-precision execution

Pages move asynchronously between host and device. A new precision state must not become visible to kernels until all required data is resident and metadata is consistent. Fused kernels then execute blocks at their selected precisions without forcing the entire layer to one common bitwidth.

This last layer is crucial: a theoretically good allocation loses value if irregular layouts, dispatch, synchronization, or data movement erase the capacity and throughput gains.

## Training/Inference Workflow

PagedWeight does not alter pre-training or post-training. Its workflow is on the serving path:

1. Profile block sensitivity and convert expert weights into the multi-precision representation.
2. Load an initial precision state and establish conservative per-block floors.
3. Observe routing and prompt-conditioned signals as requests execute.
4. Detect KV-cache pressure from growing sequences or concurrency.
5. Rank legal precision reductions by expected damage per released byte.
6. Move or expose the necessary lower-precision pages asynchronously.
7. Commit page-table and precision metadata only after residency is safe.
8. Dispatch mixed-precision expert kernels.
9. Restore high-value pages after KV memory is released.

A useful mental model is a feedback controller: KV pressure is the demand signal, expert importance is the state estimate, and bitwidth transitions are the actuator.

## Benchmarks and Evidence

The paper evaluates multiple MoE model families and RTX 6000 Ada and GH200 GPUs. Its headline results are:

- up to **72.0% GPU-memory savings** while matching FP16 accuracy;
- up to **1.94x throughput improvement**;
- up to **39.3% quality improvement** over quantization baselines at a similar memory budget;
- no more than **4.1% throughput loss** in that matched-memory quality comparison.

A useful long-context result uses Qwen1.5-MoE-A2.7B. At roughly 9.86 GB, PagedWeight reportedly matches the FP16 average LongBench score of 17.0%, while the FP16 configuration uses about 35.25 GB. A three-bit Any-Precision baseline at approximately 9.40 GB scores 12.2%. The paper reports especially large improvements on passage retrieval and NarrativeQA.

The reported throughput comparison indicates that runtime heterogeneity is not free but remains bounded: at batch sizes 1 and 4, PagedWeight is within approximately 3.3% and 4.1% of the corresponding uniform-precision baseline.

The ablation study also supports the multi-signal design. On WikiText-2/C4 perplexity, the full method reports 7.22/10.06. Removing routing, prompt residuals, movement-aware logic, or sensitivity worsens the result, with the largest listed degradation occurring after sensitivity is removed.

**Evidence boundary:** These are author-reported results. They establish a promising quality-memory-throughput frontier on the tested models and systems, but they are not independent replications.

## Why It Matters

### Weight memory becomes elastic

PagedAttention made KV storage pageable. PagedWeight applies a related systems instinct to another dominant memory tenant. The server can now adapt both cache capacity and expert precision instead of treating weights as immutable.

### MoE routing creates useful heterogeneity

The experts that are cold for one traffic window can be hot in another. That workload dependence creates an opportunity for online allocation that does not exist in a fixed global bitwidth.

### Model quality enters the scheduler's objective

Traditional schedulers reason about tokens, blocks, batches, deadlines, and utilization. PagedWeight asks the runtime to reason about predicted quality degradation as well. This points toward serving systems that optimize explicit utility rather than throughput alone.

### It joins systems and model research

The approach spans activation statistics, routing, memory virtualization, asynchronous movement, low-bit representation, and kernels. The mechanism only succeeds if it respects model behavior, and the quality policy only matters if the hardware path executes it efficiently.

## Limitations and Open Questions

1. **Proxy accuracy:** Sensitivity and routing statistics may miss rare semantic or safety failures.
2. **Thrashing:** Rapidly changing request mixes could cause repeated demotion and restoration; hysteresis or transition budgets may be required.
3. **Calibration transfer:** A prior measured on one domain may not generalize to another language, task, or instruction distribution.
4. **Distributed consistency:** Tensor and expert parallel ranks need coordinated precision metadata and page movement.
5. **Tail latency:** Average throughput can remain strong while a page transition delays an unlucky request.
6. **Hardware portability:** Kernel and paging benefits may differ across NVIDIA, AMD, TPU, Trainium, and NPU platforms.
7. **Observability:** Runtime precision changes complicate reproduction and debugging; the effective precision state should be recorded with request traces.
8. **Objective design:** A production controller may need to optimize quality, cache hit rate, preemption risk, energy, and SLOs jointly.

## Practical Takeaways

For serving engineers:

- Track weights and KV cache in one memory budget.
- Keep per-expert and per-block telemetry; layer averages hide the heterogeneity that enables dynamic allocation.
- Separate planning, page movement, and metadata commit states.
- Add hysteresis, minimum residence time, and transition-rate limits.
- Benchmark p95/p99 latency and quality by request class, not only aggregate throughput.
- Log effective precision state for incident analysis.

For researchers:

- Study calibrated combinations of routing mass, residuals, Hessian-style sensitivity, and semantic uncertainty.
- Explore online controllers with conservative precision floors.
- Evaluate joint weight/KV policies for prefill-decode disaggregation and heterogeneous memory tiers.
- Measure whether dynamic precision changes expert specialization or routing stability over long sessions.

## Reproduction Demo

The accompanying CPU-only demo implements the smallest honest reproduction of the planning principle. Each synthetic expert block has a parameter count, current bitwidth, minimum bitwidth, global sensitivity, routing mass, and prompt-conditioned residual.

It compares:

1. a quality-aware greedy planner minimizing predicted damage per released byte;
2. a uniform lockstep baseline;
3. a restoration policy that spends available headroom on the highest-value precision increases.

In the deterministic example, the quality-aware plan releases 84 MiB with predicted damage 1.2231. The uniform baseline releases 96 MiB with predicted damage 3.5050. The synthetic policy therefore reduces modeled damage by 65.1% and keeps the hottest block at 16 bits.

Five unit tests validate memory-target satisfaction, policy ordering, lower-bound enforcement, improvement over the baseline, and restoration behavior. The demo does **not** implement real quantization, bit-plane packing, DMA, vLLM integration, or mixed-precision GEMMs, and it does not validate the paper's measured model-quality or GPU-performance claims.

## Sources

- PagedWeight paper: https://arxiv.org/abs/2607.16184
- PagedWeight full HTML: https://arxiv.org/html/2607.16184v1
- PagedAttention/vLLM background: https://arxiv.org/abs/2309.06180
- LLM Research Explained notebook parent: https://app.notion.com/p/calvinfei/LLM-Research-Explained-16e2d34d69a88077a7c4cc1a24f47041
