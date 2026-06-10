# TurboQuant for Cosmos 3-Style World Model KV Cache Compression

This demo implements a **TurboQuant-inspired KV cache compression pipeline** for a Cosmos 3-style physical world model serving workload.

It does **not** depend on NVIDIA Cosmos 3 weights. Instead, it builds a synthetic multimodal world-model token stream and demonstrates the systems idea behind applying TurboQuant-style compression to long-context physical AI inference.

```text
multimodal world-model tokens
    -> Transformer K/V cache
    -> randomized Hadamard rotation
    -> low-bit scalar quantization
    -> optional 1-bit QJL residual correction
    -> approximate attention scores
```

## Why this matters

Cosmos-style world models may attend over language, image/video tokens, action tokens, robot state, temporal history, and scene memory. That creates a long-context serving problem where KV cache memory and HBM bandwidth can become first-order bottlenecks.

KV cache size scales as:

```text
layers x heads x sequence_length x head_dim x 2
```

The final factor of `2` is for keys and values.

TurboQuant-style compression is relevant because attention is driven by inner products:

```text
attention_score = query · key
```

So a useful quantizer should preserve not only reconstruction quality, but also query-key dot-product quality.

## What this demo includes

```text
TurboQuant-Cosmos3-KV-Compression/
├── README.md
├── requirements.txt
├── Makefile
├── turboquant_cosmos_demo.py
└── src/
    ├── __init__.py
    ├── cosmos_world_tokens.py
    └── turboquant.py
```

| File | Purpose |
|---|---|
| `src/cosmos_world_tokens.py` | Builds synthetic text, video, action, robot-state, and scene-memory tokens |
| `src/turboquant.py` | Implements randomized Hadamard rotation, low-bit scalar quantization, and QJL residual sketching |
| `turboquant_cosmos_demo.py` | Runs an end-to-end attention-score preservation experiment |

## Quick start

```bash
cd demo/Quantization/TurboQuant-Cosmos3-KV-Compression
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python turboquant_cosmos_demo.py
```

CPU is enough. If CUDA is available, PyTorch will use it automatically.

You can also run:

```bash
make demo
```

## Example commands

```bash
python turboquant_cosmos_demo.py --bits 3 --qjl-dim 64
python turboquant_cosmos_demo.py --bits 4 --qjl-dim 128
python turboquant_cosmos_demo.py --video-tokens 4096 --bits 4
python turboquant_cosmos_demo.py --heads 16 --head-dim 128 --bits 3
```

## What to look at

The script reports:

| Metric | Meaning |
|---|---|
| `cosine(score)` | Cosine similarity between full-precision and compressed attention scores |
| `relative error` | Relative L2 score error |
| `KL` | KL divergence between full-precision and compressed softmax attention distributions |
| `estimated KV memory` | Approximate memory footprint of the compressed cache |

## Algorithm overview

### 1. Generate a Cosmos-style multimodal context

The demo creates a sequence containing:

```text
[text tokens]
[video latent tokens]
[action tokens]
[robot state tokens]
[scene memory tokens]
```

Video and scene-memory tokens intentionally include heavier tails to mimic activation outliers.

### 2. Project tokens into synthetic K/V cache tensors

```text
K, V = project(tokens)
```

The key cache shape is:

```text
[num_heads, sequence_length, head_dim]
```

### 3. Rotate vectors before quantization

A randomized Hadamard transform spreads outlier-heavy coordinates:

```text
z = H(Dx)
```

This makes scalar quantization more robust.

### 4. Quantize in the rotated domain

```text
q_z = round(z / scale)
```

The demo uses symmetric per-vector low-bit quantization.

### 5. Add optional QJL residual correction

The residual is:

```text
r = x - x_hat
```

A 1-bit QJL-style sketch stores signs of random projections:

```text
code = sign(Sr)
```

At query time, the demo estimates residual inner products and adds them back to the MSE-only score estimate.

## Production notes

A real implementation would need:

- fused rotation and quantization during prefill
- packed INT2/INT3/INT4 KV layouts
- dequantization fused into attention
- RoPE-aware key handling
- separate policies for K and V cache
- CUDA/Triton kernels for low-bit unpacking
- evaluation on video quality, action prediction, rollout success, and temporal consistency

This demo focuses on algorithmic clarity rather than production kernel performance.
