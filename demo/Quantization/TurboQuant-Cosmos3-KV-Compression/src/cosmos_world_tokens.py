"""Synthetic Cosmos-style world-model token generator.

This module does not depend on Cosmos 3 weights. It creates a realistic-ish
multimodal token distribution for infrastructure experiments:

- text tokens: relatively well-behaved activations
- video tokens: heavier tails and spatial/temporal correlation
- action tokens: lower variance control-like embeddings
- robot state tokens: compact proprioceptive state embeddings
- scene memory tokens: heavy-tail object/trajectory memory embeddings

The point is to stress the quantizer with outliers, long context, and mixed
modalities the way a physical world model serving workload might.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class WorldTokenConfig:
    """Configuration for synthetic world-model token generation."""

    hidden_dim: int = 1024
    text_tokens: int = 64
    video_tokens: int = 1024
    action_tokens: int = 128
    robot_state_tokens: int = 64
    scene_memory_tokens: int = 256
    seed: int = 7


def _add_low_rank_temporal_structure(x: torch.Tensor, strength: float = 0.15) -> torch.Tensor:
    """Adds simple smooth temporal structure to a token sequence.

    This makes the synthetic video/action streams less i.i.d. and closer to
    real sequence activations where neighboring tokens are correlated.
    """

    if x.size(0) < 3:
        return x
    smooth = x.clone()
    smooth[1:-1] = 0.25 * x[:-2] + 0.5 * x[1:-1] + 0.25 * x[2:]
    return (1.0 - strength) * x + strength * smooth


def build_world_tokens(
    cfg: WorldTokenConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, Dict[str, slice]]:
    """Build a synthetic multimodal world-model context.

    Returns:
        tokens:
            Tensor of shape [seq_len, hidden_dim].
        spans:
            Dictionary from modality name to slice in the sequence dimension.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(cfg.seed)

    def randn(n: int, scale: float = 1.0) -> torch.Tensor:
        return scale * torch.randn(n, cfg.hidden_dim, generator=generator, dtype=dtype)

    # Text: fairly normal activations.
    text = randn(cfg.text_tokens, scale=0.75)

    # Video: heavier-tailed activations with temporal structure.
    video = randn(cfg.video_tokens, scale=0.90)
    video = video + 0.10 * torch.randn_like(video) ** 3
    video = _add_low_rank_temporal_structure(video, strength=0.25)

    # Actions: smaller magnitude and smoother.
    action = randn(cfg.action_tokens, scale=0.45)
    action = _add_low_rank_temporal_structure(action, strength=0.35)

    # Robot state: compact state-like embeddings.
    robot_state = randn(cfg.robot_state_tokens, scale=0.35)

    # Scene memory: sparse/heavy-tail object memory and trajectory embeddings.
    scene = randn(cfg.scene_memory_tokens, scale=0.65)
    sparse_mask = torch.rand(scene.shape, generator=generator) < 0.03
    scene = scene + sparse_mask.to(scene.dtype) * randn(cfg.scene_memory_tokens, scale=5.0)

    spans: Dict[str, slice] = {}
    cursor = 0
    parts = [
        ("text", text),
        ("video", video),
        ("action", action),
        ("robot_state", robot_state),
        ("scene_memory", scene),
    ]

    for name, tensor in parts:
        spans[name] = slice(cursor, cursor + tensor.size(0))
        cursor += tensor.size(0)

    tokens = torch.cat([part for _, part in parts], dim=0).to(device=device, dtype=dtype)
    # LayerNorm-like normalization, but keep some modality-specific outliers.
    tokens = tokens / (tokens.std(dim=-1, keepdim=True).clamp_min(1e-6))
    return tokens, spans


def project_to_kv_cache(
    tokens: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    seed: int = 123,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project token embeddings into synthetic K/V cache tensors.

    Args:
        tokens:
            [seq_len, hidden_dim] input activations.
        num_heads:
            Number of attention heads.
        head_dim:
            Per-head dimension.
        seed:
            Random seed for projection matrices.

    Returns:
        q:
            A single query tensor of shape [num_heads, head_dim].
        k:
            Key cache of shape [num_heads, seq_len, head_dim].
        v:
            Value cache of shape [num_heads, seq_len, head_dim].
    """

    seq_len, hidden_dim = tokens.shape
    out_dim = num_heads * head_dim

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # Keep projection on CPU for deterministic initialization, then move.
    wk = torch.randn(hidden_dim, out_dim, generator=generator, dtype=tokens.dtype) / hidden_dim**0.5
    wv = torch.randn(hidden_dim, out_dim, generator=generator, dtype=tokens.dtype) / hidden_dim**0.5
    wq = torch.randn(hidden_dim, out_dim, generator=generator, dtype=tokens.dtype) / hidden_dim**0.5

    wk = wk.to(tokens.device)
    wv = wv.to(tokens.device)
    wq = wq.to(tokens.device)

    k = (tokens @ wk).view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()
    v = (tokens @ wv).view(seq_len, num_heads, head_dim).transpose(0, 1).contiguous()

    # Use the last token as the next-step query proxy.
    q = (tokens[-1:] @ wq).view(num_heads, head_dim).contiguous()
    return q, k, v
