# Prefill Context Parallelism for MLA: Balancing 100K-Token Attention Without Coupling KV Ownership

## Executive Summary

vLLM has merged a substantial MRV2 serving change that adds virtual-batch prefill context parallelism (PCP), initially for multi-head latent attention (MLA) and sparse MLA. The implementation lets several GPUs split the prefill computation of the same long request while preserving one global scheduler view. Its most important scheduling mechanism is DualChunkSwap: instead of assigning each rank one contiguous sequence interval, it divides every request into twice as many chunks as PCP ranks and pairs an early chunk with a late chunk. This counteracts the triangular work distribution of causal attention, where tokens near the end of a sequence attend to much longer prefixes.

The second important design choice is conceptual rather than merely operational: PCP and decode context parallelism (DCP) are orthogonal. PCP partitions prefill compute; DCP determines KV-cache ownership. Treating PCP as another KV-sharding axis would make memory capacity, cache block sizing, and replication semantics difficult to reason about. vLLM instead keeps the number of KV shards equal to DCP, even when PCP expands the process world.

On four B300 GPUs with a 100K-token input and GLM-5.2-NVFP4, the reported TP1+PCP4+EP4 configuration reaches 2.859 seconds TTFT, compared with 6.015 seconds for TP4 in the reported test. These measurements are not a universal comparison—the configurations differ in expert parallelism and memory behavior—but they show that sequence-parallel prefill is now a practical serving option rather than only a training technique.

**Interpretation:** This change is valuable because it turns a model-level property—causal attention's increasing work with token position—into a runtime partitioning policy, while cleanly separating compute distribution from state ownership. It is a good example of systems research emerging from production serving constraints.

## What Was Released

The merged vLLM change adds:

- a stateful `PCPManager` for converting a globally scheduled MRV2 input batch into rank-local virtual batches;
- per-request DualChunkSwap partitioning for prefill tokens;
- plumbing for local positions, sequence lengths, logits indices, and hidden-state restoration;
- MLA and sparse-MLA execution support;
- latent-KV or K/V all-gather so each participating rank can perform attention and cache insertion;
- sparse-MLA cache insertion for GLM/DSA-style models;
- process-group changes that make PCP and DCP independent axes;
- correctness and evaluation configurations for TP+PCP+EP deployments.

This is a serving-runtime contribution. It does not introduce a new attention equation or training objective; it changes how existing attention computation and state are distributed across devices.

## Problem and Motivation

Long-context prefill creates three related problems.

### Causal attention work is not uniformly distributed across sequence positions

For a dense causal attention layer, query token `i` can attend to roughly `i + 1` keys. Ignoring constants, total prefill attention work grows quadratically with sequence length:

```text
W(L) ≈ 1 + 2 + ... + L = L(L + 1) / 2
```

If a 100K-token request is split into four equal contiguous quarters, every rank receives the same number of query tokens, but not the same work. Queries in the final quarter see far longer prefixes than queries in the first quarter. Equal token counts therefore produce severe load imbalance.

### Tensor parallelism does not solve the sequence-length dimension

Tensor parallelism shards model heads or hidden dimensions. It can reduce per-rank matrix work, but for very long prefills it may be preferable to assign more devices to the sequence dimension while using expert parallelism for MoE layers. This is especially relevant to MLA, where latent KV representations change the memory and communication tradeoffs.

### Compute partitioning and KV ownership are easy to conflate

A context-parallel group can mean two different things:

- split the current computation across sequence positions;
- shard persistent KV state across devices.

Those choices need not use the same group. Coupling them causes PCP size to affect cache block sizing, capacity accounting, replication, prefix caching, and offloading—even though PCP's primary purpose is temporary prefill-compute distribution.

## Core Technical Idea

### DualChunkSwap

For PCP size `P`, each prefill request is divided into `2P` contiguous chunks:

```text
C0, C1, ..., C(2P-1)
```

Rank `r` receives:

```text
Cr and C(2P - 1 - r)
```

For `P=4`:

```text
rank 0: C0 + C7
rank 1: C1 + C6
rank 2: C2 + C5
rank 3: C3 + C4
```

The early chunk has relatively low causal-attention cost; the mirrored late chunk has relatively high cost. Under an idealized dense-attention work model, the pair sums are close to equal.

This is more precise than simply reversing every other contiguous partition. The pairing is performed independently per request, which matters when a global serving batch contains requests of different lengths.

### Virtual batches

The scheduler continues to construct one global batch. Immediately before input preparation, `PCPManager` transforms it into a rank-local virtual batch. This avoids forcing the scheduler itself to reason in rank-local rows.

After model execution, hidden states are gathered and restored to the scheduler's original token order before logits, sampling, and postprocessing. This preserves a clean boundary:

```text
global scheduling
  -> rank-local prefill execution
  -> global-order restoration
  -> normal output processing
```

### Orthogonal PCP and DCP

The design assigns distinct meanings to parallel axes:

- **TP:** shards model heads or tensor dimensions.
- **PCP:** splits prefill sequence computation.
- **DCP:** shards KV ownership.
- **EP:** shards experts.

The KV shard count is therefore `DCP`, not `PCP × DCP`. PCP can add prefill compute workers without pretending that every worker owns a distinct fraction of persistent KV state.

A useful mental model is:

```text
PCP answers: Who computes these query positions now?
DCP answers: Who owns these KV blocks over time?
```

## Architecture or System Design

The runtime path can be summarized as follows:

```text
1. Scheduler creates a global InputBatch.
2. PCPManager identifies prefill and decode rows.
3. Each prefill request is split independently with DualChunkSwap.
4. Each PCP rank receives its virtual rows, local positions, and metadata.
5. Decode rows are replicated across PCP ranks to keep model and cache state synchronized.
6. K/V or latent KV is all-gathered for attention and cache insertion.
7. The model executes rank-local prefill work.
8. Hidden states are gathered and restored to global scheduled-token order.
9. Logits, sampling, and postprocessing proceed against the global request state.
```

The stateful manager is important because partitioning touches more than token tensors. It must preserve enough global information to restore logits indices, hidden states, request ordering, and postprocessing semantics.

For sparse MLA, index selection and cache insertion add another complication: the implementation must preserve the relationship between logical positions, selected KV locations, and DCP-owned storage.

## Training and Inference Workflow

There is no training stage specific to PCP. It is applied at inference time to a pretrained model.

A production deployment chooses parallel dimensions based on model structure and workload:

1. select TP based on attention-head and tensor-sharding constraints;
2. select EP for MoE routing and expert placement;
3. select PCP when long-prefill compute dominates TTFT;
4. select DCP based on KV ownership, capacity, and replication requirements;
5. benchmark communication overhead against the saved prefill compute.

PCP is most attractive when individual prefills are long enough that causal-attention work dominates the extra all-gather and restoration costs. For short prompts or high decode-dominated workloads, distributing a single prefill may offer little benefit.

## Benchmarks and Evidence

The vLLM PR reports results on four B300 GPUs using `nvidia/GLM-5.2-NVFP4`, a 131,072-token maximum model length, 32,768 maximum batched tokens, eager execution, and FP8 KV cache.

For a 100,000-token input and one output token, the reported mean TTFT across three sequential requests was:

| Deployment | 100K TTFT | KV token capacity per GPU |
|---|---:|---:|
| TP1 + PCP4 + EP4 | **2.859 s** | 1,964,800 |
| TP1 + PCP4 | 3.996 s | 1,899,200 |
| TP2 + PCP2 + EP4 | 3.549 s | 2,568,320 |
| TP2 + PCP2 | 4.773 s | 2,590,528 |
| TP4 + EP4 | 4.746 s | 2,867,776 |
| TP4 | 6.015 s | **2,937,024** |

A 100-sample, five-shot GSM8K smoke check reports 92–95% across the configurations.

### What the evidence supports

The results support three conclusions:

1. PCP can materially reduce TTFT for a 100K-token prefill on the tested model and hardware.
2. Combining PCP with EP can be advantageous for a large MoE model.
3. Faster prefill can trade against KV capacity, so deployment selection is multi-objective.

### What the evidence does not establish

The table does not isolate one variable at a time. TP, PCP, and EP configurations differ together, and only three sequential requests were used for the TTFT measurement. It is therefore not a universal claim that PCP4 is twice as fast as TP4.

The GSM8K result is explicitly a 100-question smoke check, not a full statistical evaluation. It confirms that the configurations are plausibly functional, not that they are numerically identical under every workload.

## Why It Matters

### Long-context serving needs a sequence-parallel runtime

As context windows grow, prefill can become the dominant latency component. Chunked prefill improves scheduling fairness, but it does not by itself reduce the work of one long prompt. PCP uses additional devices to attack that critical path directly.

### MLA changes the economics of context parallelism

MLA compresses KV representations and is increasingly common in frontier MoE models. A serving system must integrate sequence partitioning with latent-KV communication, sparse attention, and cache ownership rather than assuming conventional multi-head attention.

### Orthogonal parallel axes improve operational reasoning

Separating PCP and DCP gives operators a cleaner capacity model. They can choose compute parallelism for TTFT and state sharding for KV memory independently, then measure the communication cost of the combination.

### Relevance to an infra-to-research path

This work sits at the boundary of implementation and research:

- workload structure motivates a partition algorithm;
- the algorithm interacts with scheduler semantics and distributed state;
- process-group design determines memory capacity;
- benchmark results reveal a Pareto frontier rather than one winning configuration.

Potential follow-up research includes cost models that automatically choose TP/PCP/DCP/EP dimensions from prompt-length distributions, topology, model architecture, and service-level objectives.

## Limitations and Open Questions

1. **Communication cost:** K/V or latent-KV all-gather and hidden-state restoration can erase compute savings for shorter prompts.
2. **Request heterogeneity:** DualChunkSwap balances each request under an approximate causal-work model, but mixed sparse/dense layers, padding, and kernel tiling can change actual load.
3. **Topology awareness:** The best PCP group may depend on NVLink/NVSwitch boundaries and competition with TP or EP collectives.
4. **Prefix caching:** Cached prefixes reduce new prefill work and can change whether PCP is worthwhile.
5. **Sparse MLA:** Selected-token patterns may make work less triangular than dense attention.
6. **Decode synchronization:** Replicating decode rows across PCP ranks simplifies state consistency but consumes compute and may interact with speculative decoding.
7. **Dynamic configuration:** It remains open whether PCP size should vary by request or batch without excessive process-group and graph-management complexity.
8. **Measurement scale:** The published test uses one model, one four-GPU platform, and a small number of TTFT samples.

## Practical Takeaways

- Do not balance causal prefill by token count alone; estimate attention work by position.
- Pair early and late sequence chunks when using static sequence partitioning.
- Keep scheduler state global when possible, and treat rank-local batches as an execution view.
- Restore hidden states and logits indices explicitly before normal postprocessing.
- Separate transient compute partitioning from persistent cache ownership.
- Benchmark TTFT, KV capacity, collective time, and decode throughput together.
- Test long-context boundaries and mixed prefill/decode batches, not only single-request kernels.
- Treat parallel dimensions as a search space rather than fixed model properties.

## Reproduction Demo

The accompanying CPU-only demo implements:

- contiguous per-request partitioning as a baseline;
- DualChunkSwap partitioning;
- an idealized causal-attention work model;
- per-rank load and imbalance measurement;
- global-order restoration;
- replicated decode-row semantics;
- DCP-only KV ownership;
- a deliberately incorrect PCP×DCP ownership ablation;
- five deterministic unit tests.

For a synthetic batch containing 100K-, 32K-, and 8K-token requests with PCP=4, the contiguous baseline has an imbalance score of 1.500, while DualChunkSwap reaches 0.000 under the idealized model. This is an explanatory result, not a reproduction of B300 performance.

The demo lives in:

```text
demo/Parallelism/MLA-Prefill-Context-Parallelism
```

## Sources

- vLLM PR #46570, “Add MRV2 virtual-batch PCP for MLA”: https://github.com/vllm-project/vllm/pull/46570
- vLLM repository: https://github.com/vllm-project/vllm
- Notion notebook parent: https://app.notion.com/p/calvinfei/LLM-Research-Explained-16e2d34d69a88077a7c4cc1a24f47041
