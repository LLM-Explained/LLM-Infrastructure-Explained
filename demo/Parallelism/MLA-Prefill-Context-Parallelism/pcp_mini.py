"""A CPU-only miniature of vLLM's prefill context parallelism design.

The demo models two separable ideas:
1. DualChunkSwap balances causal-attention work across PCP ranks.
2. PCP partitions prefill compute while DCP, independently, owns KV shards.

It is an educational simulator, not an implementation of distributed attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from statistics import mean
from typing import Sequence


@dataclass(frozen=True)
class Chunk:
    request_id: str
    chunk_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def causal_work(self) -> int:
        """Proxy for dense causal attention work: sum of visible prefix lengths."""
        return sum(position + 1 for position in range(self.start, self.end))


@dataclass(frozen=True)
class PartitionResult:
    assignments: tuple[tuple[Chunk, ...], ...]

    @property
    def rank_work(self) -> tuple[int, ...]:
        return tuple(sum(chunk.causal_work for chunk in rank) for rank in self.assignments)

    @property
    def imbalance(self) -> float:
        work = self.rank_work
        avg = mean(work)
        return 0.0 if avg == 0 else (max(work) - min(work)) / avg


def _boundaries(length: int, pieces: int) -> list[int]:
    if length < 0:
        raise ValueError("length must be non-negative")
    if pieces <= 0:
        raise ValueError("pieces must be positive")
    return [(length * i) // pieces for i in range(pieces + 1)]


def split_request(request_id: str, length: int, pieces: int) -> list[Chunk]:
    boundaries = _boundaries(length, pieces)
    return [
        Chunk(request_id, i, boundaries[i], boundaries[i + 1])
        for i in range(pieces)
        if boundaries[i] < boundaries[i + 1]
    ]


def contiguous_partition(
    requests: Sequence[tuple[str, int]], pcp_size: int
) -> PartitionResult:
    """Baseline: rank r gets the r-th contiguous fraction of every request."""
    assignments: list[list[Chunk]] = [[] for _ in range(pcp_size)]
    for request_id, length in requests:
        chunks = split_request(request_id, length, pcp_size)
        for chunk in chunks:
            assignments[chunk.chunk_id].append(chunk)
    return PartitionResult(tuple(tuple(rank) for rank in assignments))


def dual_chunk_swap_partition(
    requests: Sequence[tuple[str, int]], pcp_size: int
) -> PartitionResult:
    """vLLM-style pairing: rank r gets chunks r and 2*PCP-1-r."""
    assignments: list[list[Chunk]] = [[] for _ in range(pcp_size)]
    for request_id, length in requests:
        chunks = split_request(request_id, length, 2 * pcp_size)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for rank in range(pcp_size):
            for chunk_id in (rank, 2 * pcp_size - 1 - rank):
                chunk = by_id.get(chunk_id)
                if chunk is not None:
                    assignments[rank].append(chunk)
    return PartitionResult(tuple(tuple(rank) for rank in assignments))


def covered_positions(result: PartitionResult, request_id: str) -> list[int]:
    positions: list[int] = []
    for rank in result.assignments:
        for chunk in rank:
            if chunk.request_id == request_id:
                positions.extend(range(chunk.start, chunk.end))
    return sorted(positions)


def restore_global_order(
    result: PartitionResult, request_id: str, rank_outputs: Sequence[Sequence[int]]
) -> list[int]:
    """Restore per-chunk rank outputs into original token order.

    rank_outputs[r] must concatenate outputs for rank r's chunks in assignment order.
    """
    if len(rank_outputs) != len(result.assignments):
        raise ValueError("rank_outputs must match the PCP world size")

    placed: dict[int, int] = {}
    for chunks, outputs in zip(result.assignments, rank_outputs):
        cursor = 0
        for chunk in chunks:
            values = list(outputs[cursor : cursor + chunk.length])
            if len(values) != chunk.length:
                raise ValueError("rank output is too short")
            if chunk.request_id == request_id:
                for position, value in zip(range(chunk.start, chunk.end), values):
                    if position in placed:
                        raise ValueError("duplicate global position")
                    placed[position] = value
            cursor += chunk.length
        if cursor != len(outputs):
            raise ValueError("rank output has unused values")
    return [placed[position] for position in sorted(placed)]


def decode_replica_ranks(pcp_size: int) -> tuple[int, ...]:
    """Decode rows are present on every PCP rank to keep state synchronized."""
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive")
    return tuple(range(pcp_size))


def kv_shards(dcp_size: int) -> int:
    """Orthogonal design: KV shard count depends on DCP, not PCP."""
    if dcp_size <= 0:
        raise ValueError("dcp_size must be positive")
    return dcp_size


def kv_tokens_per_rank(total_tokens: int, dcp_size: int) -> int:
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    return ceil(total_tokens / kv_shards(dcp_size))


def naive_coupled_kv_tokens_per_rank(
    total_tokens: int, pcp_size: int, dcp_size: int
) -> int:
    """A deliberately wrong model that treats PCP*DCP as KV ownership shards."""
    if pcp_size <= 0:
        raise ValueError("pcp_size must be positive")
    return ceil(total_tokens / (pcp_size * kv_shards(dcp_size)))


def rank_summary(result: PartitionResult) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "chunks": [
                f"{chunk.request_id}[{chunk.start}:{chunk.end}]" for chunk in chunks
            ],
            "work": sum(chunk.causal_work for chunk in chunks),
        }
        for rank, chunks in enumerate(result.assignments)
    ]
