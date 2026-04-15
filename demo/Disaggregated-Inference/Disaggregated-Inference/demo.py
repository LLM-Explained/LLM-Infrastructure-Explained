from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ------------------------------------------------------------
# Prefill–decode disaggregation backbone demo
# ------------------------------------------------------------
# Major architecture pieces implemented:
#   1) request model with prefill cost, decode cost, and KV size
#   2) unified serving scheduler
#   3) disaggregated serving scheduler
#   4) KV transfer cost between prefill and decode pools
#   5) simple overlap model
#
# This is a simplified implementation of the PD-disaggregation architecture.
# ------------------------------------------------------------


@dataclass
class Request:
    req_id: int
    prefill_cost: int
    decode_steps: int
    kv_transfer_cost: int


@dataclass
class Timeline:
    time: int = 0
    events: List[str] = field(default_factory=list)

    def add(self, msg: str, delta: int) -> None:
        self.events.append(f"t={self.time:03d}: {msg}")
        self.time += delta

    def dump(self, title: str) -> None:
        print(title)
        for e in self.events:
            print(" ", e)
        print(f"  total time = {self.time}")
        print()


def make_requests() -> List[Request]:
    return [
        Request(req_id=0, prefill_cost=8, decode_steps=6, kv_transfer_cost=2),
        Request(req_id=1, prefill_cost=5, decode_steps=5, kv_transfer_cost=2),
        Request(req_id=2, prefill_cost=9, decode_steps=4, kv_transfer_cost=3),
    ]


def run_unified(requests: List[Request]) -> Timeline:
    """
    Unified engine:
    prefill and decode share one pool, so prefill interferes with decode.
    We model this conservatively as serialized phase execution.
    """
    tl = Timeline()

    for r in requests:
        tl.add(f"req {r.req_id}: prefill on shared pool", r.prefill_cost)
        for step in range(r.decode_steps):
            tl.add(f"req {r.req_id}: decode step {step} on shared pool", 1)

    return tl


def run_disaggregated(requests: List[Request]) -> Timeline:
    """
    Disaggregated engine:
    prefill runs on prefill pool, decode on decode pool.
    We model simple overlap:
      - while one request's KV transfer is happening, decode of older requests can continue
      - decode steps are no longer blocked by later prefill work
    """
    tl = Timeline()
    decode_ready_queue: List[Request] = []

    # prefill side
    for r in requests:
        tl.add(f"req {r.req_id}: prefill on prefill pool", r.prefill_cost)
        tl.add(f"req {r.req_id}: KV transfer to decode pool",
               r.kv_transfer_cost)
        decode_ready_queue.append(r)

        # overlap: after each transfer, let one ready request decode one token if available
        if decode_ready_queue:
            active = decode_ready_queue[0]
            active.decode_steps -= 1
            tl.add(
                f"req {active.req_id}: overlapping decode step on decode pool", 1)
            if active.decode_steps == 0:
                decode_ready_queue.pop(0)

    # drain remaining decode work
    while decode_ready_queue:
        active = decode_ready_queue[0]
        active.decode_steps -= 1
        tl.add(f"req {active.req_id}: decode drain step on decode pool", 1)
        if active.decode_steps == 0:
            decode_ready_queue.pop(0)

    return tl


def main():
    reqs1 = make_requests()
    reqs2 = make_requests()

    unified = run_unified(reqs1)
    disagg = run_disaggregated(reqs2)

    print("=== Prefill–decode disaggregation backbone demo ===\n")
    unified.dump("Unified serving timeline")
    disagg.dump("Disaggregated serving timeline")

    speedup = unified.time / disagg.time
    print(f"Approx speedup from disaggregation: {speedup:.2f}x\n")

    print("Interpretation:")
    print("- Unified serving serializes or heavily entangles prefill and decode on one shared pool.")
    print("- Disaggregation introduces explicit KV transfer cost.")
    print("- But it removes phase interference and allows overlap between prefill/transfer and decode.")
    print("- This is the core architectural tradeoff behind PD disaggregation.")


if __name__ == "__main__":
    main()
