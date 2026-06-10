"""TurboQuant-inspired vector quantization utilities.

This module implements a readable CPU/GPU PyTorch version of the core ideas:

1. Orthogonal Hadamard rotation to spread outliers.
2. Symmetric low-bit scalar quantization in the rotated domain.
3. Optional 1-bit QJL-style residual sketch for inner-product correction.

This is educational code. A production KV-cache implementation would fuse the
rotation, quantization, packing, unpacking, and attention kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class QuantizedVectors:
    """Container for rotated low-bit quantized vectors."""

    qvalues: torch.Tensor
    scales: torch.Tensor
    original_dim: int
    padded_dim: int
    bits: int
    random_signs: torch.Tensor


@dataclass
class QJLResidualSketch:
    """1-bit QJL-style residual sketch.

    signs:
        Sign sketch of residual projections, shape [n_vectors, qjl_dim].
    projection:
        Random projection matrix, shape [original_dim, qjl_dim].
    residual_norm:
        L2 norm of residual vectors, shape [n_vectors].
    """

    signs: torch.Tensor
    projection: torch.Tensor
    residual_norm: torch.Tensor


def _next_power_of_two(x: int) -> int:
    """Returns the next power of two >= x."""

    return 1 << (x - 1).bit_length()


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Applies an unnormalized Walsh-Hadamard transform along the last dim.

    The last dimension must be a power of two. This implementation is simple
    and readable; it is not optimized for very large tensors.

    Shape invariant:
        input  [..., n]
        output [..., n]
    """

    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard dimension must be power of two, got {n}")

    prefix = x.shape[:-1]
    y = x.reshape(*prefix, n)
    h = 1
    while h < n:
        y = y.reshape(*prefix, n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.cat((a + b, a - b), dim=-1)
        y = y.reshape(*prefix, n)
        h *= 2
    return y


def randomized_hadamard_rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Applies D * H random-sign Hadamard rotation with normalization."""

    if x.shape[-1] != signs.numel():
        raise ValueError(f"sign length {signs.numel()} != vector dim {x.shape[-1]}")
    y = x * signs
    y = hadamard_transform(y)
    return y / (x.shape[-1] ** 0.5)


def inverse_randomized_hadamard_rotate(z: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Inverse of randomized_hadamard_rotate.

    For normalized Hadamard H, H^{-1} = H. Since forward is H(Dx), inverse is
    D(H z).
    """

    y = hadamard_transform(z) / (z.shape[-1] ** 0.5)
    return y * signs


def quantize_rotated_vectors(
    vectors: torch.Tensor,
    *,
    bits: int = 4,
    seed: int = 0,
) -> QuantizedVectors:
    """Quantizes vectors after randomized Hadamard rotation.

    Args:
        vectors:
            Tensor of shape [n_vectors, dim].
        bits:
            Quantization bit width, usually 2-8.
        seed:
            Random seed for sign flips.

    Returns:
        QuantizedVectors object.
    """

    if bits < 2 or bits > 8:
        raise ValueError("bits should be in [2, 8] for this demo")

    original_dim = vectors.shape[-1]
    padded_dim = _next_power_of_two(original_dim)
    if padded_dim != original_dim:
        vectors_padded = F.pad(vectors, (0, padded_dim - original_dim))
    else:
        vectors_padded = vectors

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (padded_dim,), generator=generator, dtype=torch.int8)
    signs = (signs.to(vectors.device, dtype=vectors.dtype) * 2.0) - 1.0

    rotated = randomized_hadamard_rotate(vectors_padded, signs)

    qmax = (2 ** (bits - 1)) - 1
    scales = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / qmax
    qvalues = torch.round(rotated / scales).clamp(-qmax, qmax).to(torch.int8)

    return QuantizedVectors(
        qvalues=qvalues,
        scales=scales.to(vectors.dtype),
        original_dim=original_dim,
        padded_dim=padded_dim,
        bits=bits,
        random_signs=signs,
    )


def dequantize_rotated_vectors(q: QuantizedVectors) -> torch.Tensor:
    """Dequantizes and inverse-rotates vectors."""

    rotated_hat = q.qvalues.to(q.scales.dtype) * q.scales
    vectors_padded_hat = inverse_randomized_hadamard_rotate(rotated_hat, q.random_signs)
    return vectors_padded_hat[..., : q.original_dim].contiguous()


def build_qjl_residual_sketch(
    vectors: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    qjl_dim: int = 64,
    seed: int = 99,
) -> QJLResidualSketch:
    """Builds a simple 1-bit residual sketch.

    This is a pedagogical QJL-style sketch. It stores signs of random
    projections and residual norms. At query time, we approximate residual
    inner products using the projected query and the residual sign code.
    """

    if qjl_dim <= 0:
        raise ValueError("qjl_dim must be positive")

    residual = vectors - reconstructed
    dim = vectors.shape[-1]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(dim, qjl_dim, generator=generator, dtype=vectors.dtype)
    projection = projection / (qjl_dim**0.5)
    projection = projection.to(vectors.device)

    projected = residual @ projection
    signs = torch.sign(projected)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    residual_norm = residual.norm(dim=-1).clamp_min(1e-8)
    return QJLResidualSketch(signs=signs, projection=projection, residual_norm=residual_norm)


def estimate_residual_inner_product(
    query: torch.Tensor,
    sketch: QJLResidualSketch,
) -> torch.Tensor:
    """Estimates query dot residual for every sketched vector.

    Args:
        query:
            Tensor of shape [dim].
        sketch:
            QJLResidualSketch from build_qjl_residual_sketch.

    Returns:
        Approximate residual inner product, shape [n_vectors].
    """

    q_projected = query @ sketch.projection
    correction = (sketch.signs * q_projected.unsqueeze(0)).mean(dim=-1)
    return correction * sketch.residual_norm


def estimated_quantized_memory_bytes(
    *,
    n_vectors: int,
    dim: int,
    bits: int,
    qjl_dim: Optional[int] = None,
    include_scales: bool = True,
    include_values: bool = True,
) -> int:
    """Rough memory estimate for quantized vectors.

    Assumes:
    - quantized values are bit-packed
    - scales are fp16 per vector
    - QJL signs are bit-packed
    - residual norms are fp16 per vector
    """

    total = 0
    if include_values:
        total += (n_vectors * dim * bits + 7) // 8
    if include_scales:
        total += n_vectors * 2
    if qjl_dim is not None and qjl_dim > 0:
        total += (n_vectors * qjl_dim + 7) // 8
        total += n_vectors * 2
    return total
