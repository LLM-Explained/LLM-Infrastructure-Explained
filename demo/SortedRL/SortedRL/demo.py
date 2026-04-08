from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


random.seed(0)


# ------------------------------------------------------------
# SortedRL backbone demo
# ------------------------------------------------------------
# Simplified algorithmic structure:
#   1) generate trajectories with heterogeneous lengths
#   2) compare naive rollout/update scheduling vs length-aware scheduling
#   3) allow earlier micro-updates on shorter groups
#   4) track a toy bubble-ratio proxy and policy-lag proxy
#
# This demo implements the major algorithmic idea itself:
# online length-aware scheduling for RL rollout batches.
# ------------------------------------------------------------


@dataclass
class Trajectory:
    traj_id: int
    length: int
    reward: float
    creation_step: int


def generate_rollouts(n: int, step: int) -> List[Trajectory]:
    trajs = []
    for i in range(n):
        length = random.randint(20, 180)
        reward = random.random()
        trajs.append(Trajectory(traj_id=i, length=length,
                     reward=reward, creation_step=step))
    return trajs


def group_short_first(trajectories: List[Trajectory], group_size: int) -> List[List[Trajectory]]:
    ordered = sorted(trajectories, key=lambda t: t.length)
    return [ordered[i:i + group_size] for i in range(0, len(ordered), group_size)]


def group_naive(trajectories: List[Trajectory], group_size: int) -> List[List[Trajectory]]:
    return [trajectories[i:i + group_size] for i in range(0, len(trajectories), group_size)]


def bubble_ratio(groups: List[List[Trajectory]]) -> float:
    """
    Crude utilization proxy:
    within each synchronized group, everyone waits for the longest sequence.
    """
    total_busy = 0
    total_possible = 0
    for g in groups:
        mx = max(t.length for t in g)
        total_possible += mx * len(g)
        total_busy += sum(t.length for t in g)
    return 1.0 - total_busy / total_possible


def average_policy_lag(groups: List[List[Trajectory]], update_step: int) -> float:
    """
    Simplified freshness proxy:
    how old is the data when the update consumes it?
    """
    lags = []
    for g in groups:
        for t in g:
            lags.append(update_step - t.creation_step)
    return sum(lags) / max(1, len(lags))


def simulate_update_schedule(
    trajectories: List[Trajectory],
    group_size: int,
    sorted_mode: bool,
    start_update_step: int,
) -> tuple[List[List[Trajectory]], float, float]:
    if sorted_mode:
        groups = group_short_first(trajectories, group_size)
    else:
        groups = group_naive(trajectories, group_size)

    br = bubble_ratio(groups)

    # toy assumption:
    # shorter-first groups can be consumed earlier, so average effective lag is lower
    if sorted_mode:
        effective_update_step = start_update_step + len(groups) // 2
    else:
        effective_update_step = start_update_step + len(groups)

    lag = average_policy_lag(groups, effective_update_step)
    return groups, br, lag


def print_groups(title: str, groups: List[List[Trajectory]]) -> None:
    print(title)
    for i, g in enumerate(groups):
        print(f"  group {i}: {[t.length for t in g]}")
    print()


def main() -> None:
    step = 100
    trajectories = generate_rollouts(n=16, step=step)
    group_size = 4

    naive_groups, naive_bubble, naive_lag = simulate_update_schedule(
        trajectories=trajectories,
        group_size=group_size,
        sorted_mode=False,
        start_update_step=step,
    )

    sorted_groups, sorted_bubble, sorted_lag = simulate_update_schedule(
        trajectories=trajectories,
        group_size=group_size,
        sorted_mode=True,
        start_update_step=step,
    )

    print("=== SortedRL backbone demo ===\n")
    print("Raw rollout lengths:")
    print([t.length for t in trajectories])
    print()

    print_groups("Naive grouping", naive_groups)
    print_groups("SortedRL length-aware grouping", sorted_groups)

    print("Metrics:")
    print(f"  naive bubble ratio        : {naive_bubble:.3f}")
    print(f"  sorted bubble ratio       : {sorted_bubble:.3f}")
    print(f"  naive avg policy lag      : {naive_lag:.3f}")
    print(f"  sorted avg policy lag     : {sorted_lag:.3f}")
    print()

    print("Interpretation:")
    print("- Naive grouping lets short samples wait behind long ones.")
    print("- Length-aware grouping reduces synchronized idle time.")
    print("- Earlier short-group updates also improve the freshness proxy.")
    print("- This is the core algorithmic backbone of SortedRL's scheduling idea.")


if __name__ == "__main__":
    main()
