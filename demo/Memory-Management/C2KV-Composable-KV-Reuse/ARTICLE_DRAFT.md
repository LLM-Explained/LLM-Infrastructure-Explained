# C²KV: Making Compressed KV Caches Composable for Non-Prefix Reuse

## Executive Summary

Long-context LLM serving increasingly spends time and memory not only computing KV caches, but also storing, transferring, and reading them. Prefix caching helps when requests share the same leading tokens, yet retrieval-augmented generation and multi-document agents often reuse the same documents in different orders and positions. Standard KV states are context- and position-dependent, so independently cached document segments cannot simply be concatenated without quality loss.

C²KV addresses this by learning a **compressed, position-agnostic, and composable KV representation**. A lightweight sidecar Extractor attaches to a frozen base LLM. Learnable compression tokens collect block-local document information through a structured attention pattern, while dedicated trainable QKV heads produce compact per-layer KV states. Documents are extracted independently, but training supervision is applied only after multiple compressed document caches are concatenated and used to answer a query. This compression–concatenation co-training forces the sidecar to produce states that remain useful after reordering and composition.

The paper reports strong long-context quality across multiple model families, graceful degradation from 4× to 16× compression, and up to 17× inference speedup in long-context settings. The most important systems insight is broader than the reported implementation: **cache compression and cache reuse cannot be optimized independently when the cached representation must survive relocation and composition.**

## What Was Released

C²KV was submitted on July 20, 2026 and accepted to ACM SIGKDD 2026. The authors released the paper and source code.

The framework targets non-prefix document reuse in workloads such as:

- retrieval-augmented generation;
- multi-document question answering;
- reusable agent memory;
- modular tool or knowledge contexts whose ordering changes across requests.

Unlike approaches that modify the entire base model, C²KV freezes the LLM and trains only an attached Extractor composed of a shared compression-token embedding and dedicated QKV projections for those tokens.

## Problem and Motivation

For a prompt of length `N`, prefill constructs KV states for all input tokens, and every decode step then reads an attention context that grows with `N`. Reusing document KV caches can eliminate repeated prefill, but long-context serving still faces two costs:

1. **Capacity:** stored KV states grow linearly with context length, layer count, head count, and head dimension.
2. **Bandwidth:** cached states may need to move from host memory or slower tiers into accelerator memory, then be read during every decode step.

Prefix caches are exact because a repeated prefix keeps the same causal history and positions. A document reused at a different location does not. Its standard KV vectors already encode the original causal context and positional transformation.

Training-free non-prefix reuse methods can selectively recompute tokens or blend cached states, but the paper observes a growing “KV deviation gap” as recomputation decreases. Training-based block-attention methods can make reuse more native, but modifying the base model may reduce general capability and requires expensive adaptation for every new model.

A tempting alternative is to compress existing cached KVs and then reuse them. The paper shows this fails badly: generic compression introduces additional distortion into a representation that was already not designed for composition. In the reported Llama-3.1-8B experiments, adding SnapKV compression to several reuse methods causes large accuracy drops across HotpotQA, MuSiQue, SAMSum, and 2WikiMultiHopQA.

## Core Technical Idea

C²KV learns a cache manifold with three simultaneous properties:

- **compressed:** one memory slot represents a block of original document tokens;
- **position-agnostic:** stored states can be assigned new positions when reused;
- **composable:** independently extracted document caches remain valid after concatenation with other cached documents.

The key shift is to avoid treating compression as a post-processing step over ordinary KV tensors. Instead, the Extractor is trained to emit a different kind of KV representation whose downstream purpose is known during training.

### Compression tokens as latent KV carriers

For a block size `k`, the Extractor introduces approximately one compression token for every `k` document tokens. These auxiliary tokens have a shared learnable embedding and dedicated QKV projection matrices at each Transformer layer. Original document tokens continue to use the frozen base model projections.

After extraction, only the KV pairs produced by the compression tokens are retained. The original token KVs are discarded. At 4× compression, four original-token positions are represented by roughly one reusable cache slot.

### Structured information flow

The attention mask enforces an asymmetric flow:

- original tokens follow the base model's normal causal attention and cannot attend to compression tokens;
- each compression token reads its associated local block, plus a designated sink block;
- compression tokens can attend causally to preceding compression tokens.

This structure is important for two reasons.

First, original-token states remain invariant because the sidecar cannot feed information back into them. Second, the compressed slots receive information from well-defined local regions rather than becoming globally entangled with the entire extraction prompt.

The paper contrasts this with anchor-token compression, where bidirectional interaction can change original states and entangle the compressed representation with its source context.

### Position reassignment at reuse time

The resulting compressed cache is stored before its final position-dependent transformation. When cached documents are assembled for a request, each segment receives positions corresponding to its new location, including positional reassignment and RoPE re-rotation. The segments can then be directly concatenated without a separate blending or selective-recompute phase.

## Architecture or System Design

C²KV has an offline or amortized extraction path and an online serving path.

### Extraction path

1. Split each reusable document into local token blocks.
2. Add one compression token per block.
3. Run the frozen base model together with the lightweight sidecar under the structured attention mask.
4. Retain only compression-token KVs at every layer.
5. Store those compact states outside the request-specific context.

The paper reports that only the compression-token embeddings and dedicated QKV heads are trained; the base LLM remains frozen.

### Online composition path

1. Retrieve the compressed KVs for the documents required by the request.
2. Assign each compressed segment its new logical position in the assembled context.
3. Apply the corresponding positional transformation or RoPE re-rotation.
4. Concatenate system, document, and online query KVs.
5. Decode using the frozen base model.

Because no blending or token recomputation is needed for the reusable document blocks, the query-time preparation becomes primarily a compressed-cache loading operation plus lightweight position reassignment.

## Training and Inference Workflow

### Compression–concatenation co-training

The Extractor is trained on multi-document supervised QA data assembled from HotpotQA, 2WikiMultiHopQA, and LongMagpie. Documents are encoded independently, but the answer loss is calculated only after their compressed KV segments are concatenated with the query context.

The objective is ordinary autoregressive language-model loss on the answer. There is no auxiliary reconstruction target. This matters: a cache that reconstructs its source document well but fails when reordered will still produce a high answer loss. Conversely, a representation that only works at one fixed position or with one fixed document set will encounter varied order and composition during training.

The paper also explores dynamic-ratio training, sampling different compression ratios during training and evaluating one Extractor across multiple inference-time budgets. The reported results show that this improves robustness, including at an unseen 10× compression setting.

## Benchmarks and Evidence

The paper evaluates Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and a larger Qwen3-14B configuration.

### Quality under 4× compression

On Llama-3.1-8B, the reported 4× C²KV configuration reaches:

| Task | Full recompute | C²KV 4× |
|---|---:|---:|
| HotpotQA | 0.5343 | 0.4828 |
| 2WikiMQA | 0.4018 | 0.4477 |
| MuSiQue | 0.3198 | 0.3587 |
| Qasper | 0.4417 | 0.3755 |
| SAMSum | 0.3652 | 0.3904 |

Some compressed results exceed full recomputation on individual tasks. That should not be interpreted as a universally better representation; dataset variance, learned extraction, and answer-format effects can all contribute. The stronger conclusion is that aggressive compression does not necessarily destroy useful multi-document information when composition is part of the training objective.

### Scaling compression ratio

For Llama-3.1-8B, fixed-ratio and dynamic-ratio models degrade gradually from 4× through 16× compression. On WikiMQA, for example, the reported fixed-ratio scores remain around 0.448 at 4×, 0.446 at 8×, and 0.397 at 16×. Dynamic-ratio training improves several intermediate settings.

### Long context and larger models

At 4× compression on RULER, the authors report stable retrieval as context grows from 4K to 64K tokens. On Qwen3-14B, the 4× configuration remains close to or above the full-context baseline on several reported tasks.

### Systems performance

The paper reports up to 17× inference speedup under long contexts. TTFT is reduced because reusable documents no longer require full online prefill or expensive blending, while decode time improves because fewer KV slots must be transferred and read. As context grows from 16K to 128K, the compressed configuration's time-per-token curve rises much more slowly than full-length caches.

The measurement protocol excludes offline extraction and the short system/query prefill. Therefore, the result is most applicable when documents are reused enough times to amortize extraction and when cached states reside outside GPU memory before a request.

## Why It Matters

### Reuse changes the representation contract

Most KV compression work assumes that compressed states remain in the same sequence where they were produced. Non-prefix reuse imposes a stronger contract: a segment must remain meaningful after relocation, reordering, and concatenation. C²KV makes that contract explicit in both architecture and training.

### The serving bottleneck is shifting

Once prefill is avoided, storage and transfer can dominate. A reuse system that preserves full-size KVs may save compute yet remain limited by host memory, PCIe or fabric transfer, HBM capacity, and decode bandwidth. Compressing the reusable artifact attacks all of these costs at once.

### Sidecars can change system behavior without rewriting the base model

Freezing the base LLM reduces adaptation cost and preserves its original token path. The sidecar pattern is attractive for serving research because it can attach a new systems capability—cache extraction—without requiring complete continued pretraining of every target model.

## Limitations and Open Questions

- **Offline extraction is excluded from latency.** Benefits depend on reuse frequency and cache lifetime.
- **Per-model sidecar training remains necessary.** The base is frozen, but dedicated QKV heads still depend on the model's internal dimensions and representations.
- **Quality varies by task.** Summarization and exact retrieval may respond differently to block-local compression.
- **Position reassignment is not free.** Real implementations need efficient per-layer RoPE re-rotation and cache-layout kernels.
- **Cache invalidation is unresolved.** Updating source documents requires re-extraction and version management.
- **Multi-tenant serving raises placement questions.** Compressed caches still need admission, eviction, tiering, and transfer policies.
- **Security and provenance remain open.** Reusable cache objects may encode sensitive information and are harder to inspect than source text.
- **The 17× maximum is workload-dependent.** It should not be generalized to short prompts, low reuse, or HBM-resident caches without measuring the actual bottleneck.

## Practical Takeaways

1. Treat cache artifacts as an interface, not merely stored activations.
2. Co-design compression with the transformations the cache will undergo later.
3. Keep reusable semantic content separate from request-specific position state whenever possible.
4. Measure extraction amortization, host-to-device transfer, TTFT, and decode bandwidth separately.
5. Compare against equal-compression baselines, not only equal-reuse baselines.
6. Test arbitrary document ordering and subset composition; prefix-style tests are insufficient.
7. Preserve a fallback to recomputation when a cache version, model revision, or positional configuration is incompatible.

## Reproduction Demo

The accompanying CPU-only demo isolates position-aware composition using a deterministic RoPE-like attention model.

It creates four synthetic documents, compresses every four tokens into one semantic slot, and evaluates all document permutations. The C²KV-style path stores slots before positional rotation and applies their new positions during composition. The baseline stores already-rotated keys and reuses them at stale extraction positions.

Expected result:

| Metric | C²KV-style | Stale-position baseline |
|---|---:|---:|
| Compression ratio | 4× | 4× |
| Key reconstruction MSE | 0.000000 | 0.748379 |
| Top-1 attention agreement | 100.0% | 18.2% |
| Mean attention KL | 0.000000 | 0.294563 |

This validates one invariant: equal-size compressed caches can behave very differently under non-prefix composition depending on whether position is bound at extraction or reuse time.

It does not reproduce the learned Extractor, real model quality, GPU kernels, or performance claims.

## Sources

- [C²KV paper](https://arxiv.org/abs/2607.17715)
- [C²KV full HTML](https://arxiv.org/html/2607.17715)
- [Authors' code](https://github.com/s7a9/C2KV)
- [LongBench](https://github.com/THUDM/LongBench)
- [RULER](https://github.com/NVIDIA/RULER)
