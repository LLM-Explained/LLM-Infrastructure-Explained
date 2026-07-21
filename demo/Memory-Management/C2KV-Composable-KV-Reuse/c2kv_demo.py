"""A small CPU-only model of C2KV's composable KV-cache idea.

The demo isolates one central systems property from C2KV:
compressed document caches should be stored before position-dependent rotation,
then assigned positions only when independently cached documents are composed.

This is not a language model and does not reproduce the paper's learned sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
import random
from typing import Sequence

Vector = tuple[float, ...]


@dataclass(frozen=True)
class Document:
    name: str
    tokens: tuple[Vector, ...]


@dataclass(frozen=True)
class CompressedSlot:
    document: str
    block_index: int
    span: int
    semantic: Vector
    extraction_position: int
    stale_positioned_key: Vector


@dataclass(frozen=True)
class Evaluation:
    compression_ratio: float
    c2kv_mse: float
    stale_mse: float
    c2kv_top1_agreement: float
    stale_top1_agreement: float
    c2kv_mean_kl: float
    stale_mean_kl: float


def _mean(vectors: Sequence[Vector]) -> Vector:
    if not vectors:
        raise ValueError("cannot average an empty block")
    dim = len(vectors[0])
    if dim % 2:
        raise ValueError("RoPE demo requires an even vector dimension")
    if any(len(v) != dim for v in vectors):
        raise ValueError("all vectors must have the same dimension")
    return tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(dim))


def rope(vector: Vector, position: int, base_frequency: float = 0.37) -> Vector:
    """Apply a deterministic RoPE-like pairwise rotation."""
    if len(vector) % 2:
        raise ValueError("RoPE requires an even dimension")
    out: list[float] = []
    pairs = len(vector) // 2
    for pair in range(pairs):
        x, y = vector[2 * pair], vector[2 * pair + 1]
        theta = position * base_frequency / (1.0 + 0.65 * pair)
        c, s = math.cos(theta), math.sin(theta)
        out.extend((x * c - y * s, x * s + y * c))
    return tuple(out)


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def squared_error(a: Vector, b: Vector) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) / len(a)


def softmax(values: Sequence[float]) -> tuple[float, ...]:
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return tuple(v / total for v in exps)


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    eps = 1e-12
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(p, q, strict=True))


def extract_document(document: Document, block_size: int) -> tuple[CompressedSlot, ...]:
    """Compress each local token block into one position-agnostic memory slot.

    ``stale_positioned_key`` is retained only as the non-composable baseline:
    it represents a conventional cache that baked in extraction-time positions.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    slots: list[CompressedSlot] = []
    for block_index, start in enumerate(range(0, len(document.tokens), block_size)):
        block = document.tokens[start : start + block_size]
        semantic = _mean(block)
        extraction_position = start + len(block) - 1
        slots.append(
            CompressedSlot(
                document=document.name,
                block_index=block_index,
                span=len(block),
                semantic=semantic,
                extraction_position=extraction_position,
                stale_positioned_key=rope(semantic, extraction_position),
            )
        )
    return tuple(slots)


def compose(
    caches: dict[str, tuple[CompressedSlot, ...]],
    order: Sequence[str],
    *,
    rerotate: bool,
) -> tuple[Vector, ...]:
    """Compose document caches in a new order.

    C2KV-style composition uses the position-agnostic semantic slot and applies
    its new position. The baseline reuses the extraction-time rotated key.
    """
    keys: list[Vector] = []
    token_offset = 0
    for name in order:
        slots = caches[name]
        local_end = 0
        for slot in slots:
            local_end += slot.span
            new_position = token_offset + local_end - 1
            keys.append(rope(slot.semantic, new_position) if rerotate else slot.stale_positioned_key)
        token_offset += sum(slot.span for slot in slots)
    return tuple(keys)


def make_documents(
    *,
    num_documents: int = 4,
    tokens_per_document: int = 16,
    dim: int = 12,
    seed: int = 7,
) -> tuple[Document, ...]:
    if dim % 2:
        raise ValueError("dim must be even")
    rng = random.Random(seed)
    documents: list[Document] = []
    for doc_idx in range(num_documents):
        tokens: list[Vector] = []
        anchor = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for token_idx in range(tokens_per_document):
            token = tuple(
                0.65 * anchor[j]
                + 0.35 * rng.gauss(0.0, 1.0)
                + 0.08 * math.sin((doc_idx + 1) * (token_idx + 1) * (j + 1))
                for j in range(dim)
            )
            tokens.append(token)
        documents.append(Document(name=f"doc-{doc_idx}", tokens=tuple(tokens)))
    return tuple(documents)


def evaluate(
    documents: Sequence[Document],
    *,
    block_size: int = 4,
    queries_per_order: int = 48,
    seed: int = 11,
) -> Evaluation:
    caches = {doc.name: extract_document(doc, block_size) for doc in documents}
    orders = tuple(permutations(doc.name for doc in documents))
    rng = random.Random(seed)
    dim = len(documents[0].tokens[0])

    c2kv_error = 0.0
    stale_error = 0.0
    c2kv_top1 = 0
    stale_top1 = 0
    c2kv_kl = 0.0
    stale_kl = 0.0
    comparisons = 0
    key_vectors = 0

    for order in orders:
        ideal = compose(caches, order, rerotate=True)
        c2kv = compose(caches, order, rerotate=True)
        stale = compose(caches, order, rerotate=False)

        for expected, actual, old in zip(ideal, c2kv, stale, strict=True):
            c2kv_error += squared_error(expected, actual)
            stale_error += squared_error(expected, old)
            key_vectors += 1

        final_position = sum(len(doc.tokens) for doc in documents)
        for _ in range(queries_per_order):
            semantic_query = tuple(rng.gauss(0.0, 1.0) for _ in range(dim))
            query = rope(semantic_query, final_position)
            ideal_scores = [dot(query, key) / math.sqrt(dim) for key in ideal]
            c2kv_scores = [dot(query, key) / math.sqrt(dim) for key in c2kv]
            stale_scores = [dot(query, key) / math.sqrt(dim) for key in stale]

            ideal_prob = softmax(ideal_scores)
            c2kv_prob = softmax(c2kv_scores)
            stale_prob = softmax(stale_scores)
            ideal_winner = max(range(len(ideal_scores)), key=ideal_scores.__getitem__)
            c2kv_winner = max(range(len(c2kv_scores)), key=c2kv_scores.__getitem__)
            stale_winner = max(range(len(stale_scores)), key=stale_scores.__getitem__)

            c2kv_top1 += int(c2kv_winner == ideal_winner)
            stale_top1 += int(stale_winner == ideal_winner)
            c2kv_kl += kl_divergence(ideal_prob, c2kv_prob)
            stale_kl += kl_divergence(ideal_prob, stale_prob)
            comparisons += 1

    original_tokens = sum(len(doc.tokens) for doc in documents)
    compressed_slots = sum(len(slots) for slots in caches.values())
    return Evaluation(
        compression_ratio=original_tokens / compressed_slots,
        c2kv_mse=c2kv_error / key_vectors,
        stale_mse=stale_error / key_vectors,
        c2kv_top1_agreement=c2kv_top1 / comparisons,
        stale_top1_agreement=stale_top1 / comparisons,
        c2kv_mean_kl=c2kv_kl / comparisons,
        stale_mean_kl=stale_kl / comparisons,
    )


def format_report(result: Evaluation) -> str:
    return "\n".join(
        [
            "C2KV composability miniature",
            f"Compression ratio:                 {result.compression_ratio:.1f}x",
            f"C2KV key reconstruction MSE:       {result.c2kv_mse:.6f}",
            f"Stale-position baseline MSE:       {result.stale_mse:.6f}",
            f"C2KV top-1 attention agreement:    {result.c2kv_top1_agreement:.1%}",
            f"Stale-position top-1 agreement:    {result.stale_top1_agreement:.1%}",
            f"C2KV mean attention KL:            {result.c2kv_mean_kl:.6f}",
            f"Stale-position mean attention KL:  {result.stale_mean_kl:.6f}",
        ]
    )
