# MLA Prefill Context Parallelism: DualChunkSwap Miniature

This CPU-only demo reproduces the **core scheduling idea** behind vLLM's MRV2 prefill context parallelism (PCP) implementation for MLA and sparse MLA:

1. split each prefill request across PCP ranks;
2. pair an early causal-attention chunk with a late chunk using **DualChunkSwap**;
3. restore rank-local outputs to global scheduled-token order;
4. keep prefill-compute partitioning (PCP) independent from KV-cache ownership (DCP).

The full vLLM implementation includes distributed process groups, latent-KV all-gathers, input-batch mutation and restoration, cache insertion, logits handling, and model execution. This educational simulator deliberately models only the concepts that can be inspected honestly on a laptop.

## Why the partition matters

For dense causal attention, token `i` attends to approximately `i + 1` earlier positions. A contiguous quarter of a 100K-token request therefore does not represent one quarter of the attention work: the final quarter is much more expensive than the first.

The baseline gives rank `r` the `r`-th contiguous fraction. DualChunkSwap divides the request into `2 × PCP` chunks and gives rank `r`:

```text
chunk r  +  chunk (2 × PCP - 1 - r)
```

With PCP=4:

```text
Global prefill: | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |

rank 0:           C0                                 C7
rank 1:                C1                       C6
rank 2:                     C2             C5
rank 3:                          C3   C4
```

Pairing low-cost early chunks with high-cost late chunks balances the triangular causal-attention workload.

## Layout

```text
MLA-Prefill-Context-Parallelism/
├── README.md
├── ARTICLE_DRAFT.md
├── Makefile
├── requirements.txt
├── pcp_mini.py
├── example.py
└── tests/
    └── test_pcp_mini.py
```

## Run

```bash
cd demo/Parallelism/MLA-Prefill-Context-Parallelism
python example.py
python -m unittest discover -s tests -v
```

Or:

```bash
make demo
make test
```

No GPU or third-party dependency is required. Python 3.10+ is sufficient.

## Expected result

For deterministic 100K-, 32K-, and 8K-token requests with PCP=4:

```text
Contiguous baseline imbalance: 1.500
DualChunkSwap imbalance:       0.000
```

The exact zero is a property of this idealized triangular-work model and evenly divisible example. Real kernels have communication, padding, tiling, sparse-indexing, and request-mix effects, so production balance will not be perfect.

The demo also shows the ownership distinction:

```text
total tokens: 140,960
DCP=2 correct KV ownership: 70,480 tokens/rank
PCP=4, DCP=2 coupled model: 17,620 tokens/rank  # intentionally wrong
```

PCP partitions prefill computation. It does **not** automatically multiply the number of KV owners. In the vLLM design, DCP determines KV sharding.

## Baseline and ablations

- `contiguous_partition`: naïve contiguous sequence partitioning.
- `dual_chunk_swap_partition`: early/late chunk pairing.
- `naive_coupled_kv_tokens_per_rank`: deliberately wrong PCP×DCP ownership model.
- `kv_tokens_per_rank`: orthogonal DCP-only ownership model.

## What this validates

The demo validates that:

- all prefill positions can be assigned exactly once;
- early/late pairing reduces the imbalance created by causal-attention growth;
- rank-local outputs can be restored to original token order;
- decode rows can be modeled as replicated across PCP ranks;
- PCP compute partitioning and DCP KV ownership are distinct axes.

## What this does not validate

It does not reproduce:

- vLLM's MRV2 runtime or scheduler;
- GPU collectives, latent-KV all-gather, or hidden-state gather;
- MLA/sparse-MLA kernels;
- prefix-cache insertion or offloading;
- B300 performance numbers;
- communication/computation overlap;
- numerical equivalence of a real distributed model.

## Sources

- vLLM implementation PR: https://github.com/vllm-project/vllm/pull/46570
- vLLM repository: https://github.com/vllm-project/vllm
- Notebook parent: https://app.notion.com/p/calvinfei/LLM-Research-Explained-16e2d34d69a88077a7c4cc1a24f47041

The publication-ready article is included in [`ARTICLE_DRAFT.md`](ARTICLE_DRAFT.md). Authenticated Notion publication requires a connected Notion integration.
