"""Run a TurboQuant-inspired demo on a Cosmos-style world-model KV cache.

This demo is self-contained and does not require Cosmos 3 weights.

It simulates a long multimodal context, projects it into K/V cache tensors,
compresses K with a rotation-based low-bit quantizer, then measures how well
the compressed keys preserve next-token/query attention scores.

Usage:
    python turboquant_cosmos_demo.py
    python turboquant_cosmos_demo.py --bits 3 --qjl-dim 128 --video-tokens 4096
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import torch.nn.functional as F

from src.cosmos_world_tokens import WorldTokenConfig, build_world_tokens, project_to_kv_cache
from src.turboquant import (
    build_qjl_residual_sketch,
    dequantize_rotated_vectors,
    estimate_residual_inner_product,
    estimated_quantized_memory_bytes,
    quantize_rotated_vectors,
)


def bytes_to_mib(num_bytes: float) -> float:
    return num_bytes / (1024**2)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def relative_l2_error(reference: torch.Tensor, approx: torch.Tensor) -> float:
    return ((reference - approx).norm() / reference.norm().clamp_min(1e-8)).item()


def attention_kl(reference_scores: torch.Tensor, approx_scores: torch.Tensor) -> float:
    p = F.softmax(reference_scores, dim=-1).clamp_min(1e-9)
    q = F.softmax(approx_scores, dim=-1).clamp_min(1e-9)
    return (p * (p.log() - q.log())).sum(dim=-1).mean().item()


def compress_keys_per_head(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    bits: int,
    qjl_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compresses key cache head-by-head and returns approximate scores.

    Args:
        q:
            Query tensor [num_heads, head_dim].
        k:
            Key cache [num_heads, seq_len, head_dim].
        bits:
            Low-bit quantization width.
        qjl_dim:
            Residual sketch dimension. Use 0 to disable QJL.
        seed:
            Base random seed.

    Returns:
        scores_mse:
            Attention scores from dequantized MSE-only keys.
        scores_qjl:
            Attention scores with QJL residual correction. If qjl_dim == 0,
            this is the same as scores_mse.
    """

    num_heads, _, head_dim = k.shape
    mse_scores = []
    qjl_scores = []

    scale = head_dim ** -0.5

    for h in range(num_heads):
        vectors = k[h]
        quantized = quantize_rotated_vectors(vectors, bits=bits, seed=seed + h)
        k_hat = dequantize_rotated_vectors(quantized)

        base_score = (q[h].unsqueeze(0) * k_hat).sum(dim=-1) * scale
        mse_scores.append(base_score)

        if qjl_dim > 0:
            sketch = build_qjl_residual_sketch(
                vectors,
                k_hat,
                qjl_dim=qjl_dim,
                seed=seed + 1000 + h,
            )
            residual_score = estimate_residual_inner_product(q[h], sketch) * scale
            qjl_scores.append(base_score + residual_score)
        else:
            qjl_scores.append(base_score)

    return torch.stack(mse_scores, dim=0), torch.stack(qjl_scores, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TurboQuant-style KV compression demo for Cosmos-style world models.")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--text-tokens", type=int, default=64)
    parser.add_argument("--video-tokens", type=int, default=1024)
    parser.add_argument("--action-tokens", type=int, default=128)
    parser.add_argument("--robot-state-tokens", type=int, default=64)
    parser.add_argument("--scene-memory-tokens", type=int, default=256)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--qjl-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32

    cfg = WorldTokenConfig(
        hidden_dim=args.hidden_dim,
        text_tokens=args.text_tokens,
        video_tokens=args.video_tokens,
        action_tokens=args.action_tokens,
        robot_state_tokens=args.robot_state_tokens,
        scene_memory_tokens=args.scene_memory_tokens,
        seed=args.seed,
    )

    tokens, _ = build_world_tokens(cfg, device=device, dtype=dtype)
    q, k, _ = project_to_kv_cache(tokens, num_heads=args.heads, head_dim=args.head_dim, seed=args.seed + 17)

    baseline_scores = torch.einsum("hd,hsd->hs", q, k) * (args.head_dim ** -0.5)
    mse_scores, qjl_scores = compress_keys_per_head(
        q,
        k,
        bits=args.bits,
        qjl_dim=args.qjl_dim,
        seed=args.seed + 100,
    )

    # Memory estimates. Baseline assumes fp16 K and V in production serving.
    seq_len = k.shape[1]
    n_vectors = args.heads * seq_len
    baseline_kv_bytes = 2 * n_vectors * args.head_dim * 2  # K+V, fp16

    quantized_k_bytes = estimated_quantized_memory_bytes(
        n_vectors=n_vectors,
        dim=args.head_dim,
        bits=args.bits,
        qjl_dim=args.qjl_dim if args.qjl_dim > 0 else None,
    )
    # Keep V as fp16 for this demo. Many real systems quantize K and V differently.
    fp16_v_bytes = n_vectors * args.head_dim * 2
    quantized_total_bytes = quantized_k_bytes + fp16_v_bytes

    print(f"Device: {device}")
    print("World-model token mix:")
    print(f"  text tokens:         {cfg.text_tokens}")
    print(f"  video tokens:        {cfg.video_tokens}")
    print(f"  action tokens:       {cfg.action_tokens}")
    print(f"  robot-state tokens:  {cfg.robot_state_tokens}")
    print(f"  scene-memory tokens: {cfg.scene_memory_tokens}")
    print(f"  total tokens:        {seq_len}")
    print()
    print("KV cache memory estimate:")
    print(f"  Baseline fp16 K+V:   {bytes_to_mib(baseline_kv_bytes):.2f} MiB")
    print(f"  INT{args.bits} K + fp16 V: {bytes_to_mib(quantized_total_bytes):.2f} MiB estimated")
    print(f"  Compression ratio:   {baseline_kv_bytes / quantized_total_bytes:.2f}x")
    print()
    print("Attention score metrics:")
    print(f"  INT{args.bits} MSE-only cosine(score):        {cosine_similarity(baseline_scores, mse_scores):.4f}")
    print(f"  INT{args.bits} + QJL residual cosine(score):  {cosine_similarity(baseline_scores, qjl_scores):.4f}")
    print(f"  INT{args.bits} MSE-only relative error:       {relative_l2_error(baseline_scores, mse_scores):.4f}")
    print(f"  INT{args.bits} + QJL residual relative error: {relative_l2_error(baseline_scores, qjl_scores):.4f}")
    print()
    print("Attention distribution metrics:")
    print(f"  INT{args.bits} MSE-only KL:        {attention_kl(baseline_scores, mse_scores):.6f}")
    print(f"  INT{args.bits} + QJL residual KL:  {attention_kl(baseline_scores, qjl_scores):.6f}")
    print()
    print("Config:")
    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    print(f"  heads: {args.heads}")
    print(f"  head_dim: {args.head_dim}")
    print(f"  bits: {args.bits}")
    print(f"  qjl_dim: {args.qjl_dim}")


if __name__ == "__main__":
    main()
