# C²KV: Composable Compressed KV Reuse — Educational Demo

This directory is a small, CPU-only reproduction of one fundamental idea from **C²KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference**:

> A reusable document cache should keep compressed semantic content separate from position-dependent encoding, then assign positions when independently cached documents are composed for a new request.

The paper implements this with a learned sidecar Extractor, compression tokens, dedicated QKV heads, structured attention, and compression–concatenation co-training. This demo does **not** reproduce that full model. It isolates the composability property with a deterministic RoPE-like toy attention system.

## What the demo compares

1. **C²KV-style cache**
   - compresses every local token block into one semantic slot;
   - stores the slot before position-dependent rotation;
   - applies the slot's new position after documents are reordered and concatenated.

2. **Stale-position baseline**
   - uses the same compression ratio;
   - stores keys after extraction-time positional rotation;
   - directly reuses those keys when documents move to new positions.

The ideal reference is a compressed cache recomputed at the new positions. The demo evaluates every permutation of four independently cached documents and compares reconstructed keys and attention distributions.

## Run

```bash
python example.py
python -m unittest discover -s tests -v
```

Or:

```bash
make demo
make test
```

No third-party packages are required. Python 3.10+ is sufficient.

## Expected output

```text
C2KV composability miniature
Compression ratio:                 4.0x
C2KV key reconstruction MSE:       0.000000
Stale-position baseline MSE:       0.748379
C2KV top-1 attention agreement:    100.0%
Stale-position top-1 agreement:    18.2%
C2KV mean attention KL:            0.000000
Stale-position mean attention KL:  0.294563
```

The exact values are deterministic for the checked-in seed.

## What this validates

- Position-agnostic compressed slots can be reassigned to new positions without changing their semantic content.
- Caching position-entangled keys and reusing them after non-prefix reordering can substantially distort attention.
- Compression and composability are separate requirements: using the same number of cache slots does not guarantee that independently extracted segments can be safely concatenated.

## What this does not validate

This miniature does not implement or verify:

- the paper's trainable C² Extractor or dedicated per-layer QKV heads;
- block-structured attention inside a real Transformer;
- compression–concatenation co-training on QA data;
- real KV tensors across multiple layers and heads;
- RoPE re-rotation kernels, host-to-GPU transfer, TTFT, TBT, or the reported 17× speedup;
- LongBench, RULER, Qwen, or Llama quality results.

The perfect C²KV score occurs because the toy implementation explicitly stores pre-position semantic slots and uses the same deterministic rotation for recomposition and the ideal reference. It demonstrates the invariant by construction, not learned quality preservation.

## Sources

- [Paper](https://arxiv.org/abs/2607.17715)
- [Paper HTML](https://arxiv.org/html/2607.17715)
- [Authors' implementation](https://github.com/s7a9/C2KV)
- [Publication-ready article draft](./ARTICLE_DRAFT.md)
