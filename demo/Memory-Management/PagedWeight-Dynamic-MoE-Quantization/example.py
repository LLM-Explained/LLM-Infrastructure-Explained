from pagedweight_mini import (
    ExpertBlock,
    mib,
    quality_aware_plan,
    restore_with_headroom,
    total_memory_bytes,
    uniform_plan,
)


def make_blocks() -> tuple[ExpertBlock, ...]:
    # Equal-size blocks make the planner's prioritization easy to inspect.
    params = 16 * 1024 * 1024
    return (
        ExpertBlock("layer0.hot_gate_up", params, 16, 8, 1.10, 0.40, 0.20),
        ExpertBlock("layer0.cold_gate_up", params, 16, 2, 0.18, 0.03, -0.20),
        ExpertBlock("layer0.hot_down", params, 16, 8, 0.95, 0.35, 0.10),
        ExpertBlock("layer0.cold_down", params, 16, 2, 0.22, 0.04, -0.10),
        ExpertBlock("layer1.medium_gate_up", params, 16, 4, 0.50, 0.12, 0.00),
        ExpertBlock("layer1.medium_down", params, 16, 4, 0.55, 0.15, 0.05),
    )


def summarize(label: str, result) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"target met:       {result.target_met}")
    print(f"released memory:  {mib(result.released_bytes):6.1f} MiB")
    print(f"predicted damage: {result.predicted_damage:8.4f}")
    print("actions:")
    for action in result.actions:
        print(
            f"  {action.block_name:28s} {action.from_bits:2d}->{action.to_bits:2d} "
            f"release={mib(action.released_bytes):5.1f} MiB "
            f"damage={action.predicted_damage:.4f}"
        )


def main() -> None:
    blocks = make_blocks()
    initial = total_memory_bytes(blocks)
    target = 72 * 1024 * 1024

    print("PagedWeight quality-aware planning miniature")
    print("=============================================")
    print(f"initial expert memory: {mib(initial):.1f} MiB")
    print(f"KV-cache pressure asks for: {mib(target):.1f} MiB")

    adaptive = quality_aware_plan(blocks, target)
    uniform = uniform_plan(blocks, target)
    summarize("Quality-aware dynamic plan", adaptive)
    summarize("Uniform lockstep baseline", uniform)

    print("\nComparison")
    print("----------")
    print(
        f"damage reduction: {(1 - adaptive.predicted_damage / uniform.predicted_damage) * 100:5.1f}%"
    )
    print(
        "hot blocks protected: "
        f"{next(b.bitwidth for b in adaptive.blocks if b.name == 'layer0.hot_gate_up')}bit"
    )

    restored, transitions, used = restore_with_headroom(
        adaptive.blocks, available_bytes=48 * 1024 * 1024
    )
    print("\nPressure eases: restore high-value pages")
    print("----------------------------------------")
    print(f"headroom consumed: {mib(used):.1f} MiB")
    for transition in transitions:
        print(f"  {transition}")
    print(f"final expert memory: {mib(total_memory_bytes(restored)):.1f} MiB")


if __name__ == "__main__":
    main()
