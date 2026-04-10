from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ------------------------------------------------------------
# Efficient Reasoning on the Edge backbone demo
# ------------------------------------------------------------
# Major architecture / algorithm pieces implemented:
#   1) frozen backbone with shared prompt encoding cache
#   2) fast adapter with lightweight query classification + concise answering
#   3) reasoning adapter with explicit multi-step reasoning state
#   4) dynamic adapter switching
#   5) budget-forced iterative refinement
#
# This version adds more detail to both adapters while keeping the code concise.
# ------------------------------------------------------------


@dataclass
class PromptCache:
    encoded_prompt: str
    system_style: str
    reusable_kv_tag: str


@dataclass
class FastState:
    query_type: str
    extracted_focus: str
    answer: str


@dataclass
class ReasoningState:
    problem_type: str
    subgoals: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    partial_conclusion: str = ""
    steps_used: int = 0


class FrozenBackbone:
    """
    Simplified frozen backbone.
    In the real system this would be the shared base model + prompt KV cache.
    """

    def encode_prompt(self, prompt: str) -> PromptCache:
        style = "concise, edge-optimized, budget-aware"
        return PromptCache(
            encoded_prompt=f"[ENCODED_PROMPT]{prompt}",
            system_style=style,
            reusable_kv_tag="PROMPT_KV_SHARED",
        )

    def build_query_context(self, cache: PromptCache, query: str) -> str:
        return (
            f"{cache.encoded_prompt} "
            f"| STYLE={cache.system_style} "
            f"| KV={cache.reusable_kv_tag} "
            f"| QUERY={query}"
        )


class FastAdapter:
    """
    Lightweight adapter for easy / direct queries.
    It performs:
      - simple query typing
      - focus extraction
      - concise answer formatting
    """

    def classify_query(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["what time", "when", "date", "deadline"]):
            return "factual_direct"
        if any(x in q for x in ["translate", "meaning", "define"]):
            return "language_lookup"
        if any(x in q for x in ["where", "location", "address"]):
            return "lookup_location"
        return "generic_fast"

    def extract_focus(self, query: str) -> str:
        q = query.strip().rstrip("?")
        return q[:80]

    def generate_answer(self, query_type: str, focus: str) -> str:
        if query_type == "factual_direct":
            return f"FAST_RESPONSE: direct answer for '{focus}'."
        if query_type == "language_lookup":
            return f"FAST_RESPONSE: concise definition/translation for '{focus}'."
        if query_type == "lookup_location":
            return f"FAST_RESPONSE: concise location-oriented answer for '{focus}'."
        return f"FAST_RESPONSE: concise response for '{focus}'."

    def run(self, backbone_state: str, query: str) -> FastState:
        query_type = self.classify_query(query)
        focus = self.extract_focus(query)
        answer = self.generate_answer(query_type, focus)
        return FastState(
            query_type=query_type,
            extracted_focus=focus,
            answer=answer,
        )


class ReasoningAdapter:
    """
    Reasoning-specialized adapter.
    This adapter now has a more explicit internal algorithm:
      1) classify reasoning problem
      2) build subgoals
      3) iteratively refine evidence / partial conclusion under budget
      4) synthesize final concise answer
    """

    def classify_problem(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in ["prove", "why", "because"]):
            return "explanatory_reasoning"
        if any(x in q for x in ["plan", "schedule", "route", "commute"]):
            return "planning_reasoning"
        if any(x in q for x in ["solve", "multi-step", "math"]):
            return "stepwise_problem_solving"
        return "general_reasoning"

    def initialize_state(self, query: str) -> ReasoningState:
        problem_type = self.classify_problem(query)

        if problem_type == "explanatory_reasoning":
            subgoals = [
                "identify the main claim",
                "find the causal mechanism",
                "compress into a concise explanation",
            ]
        elif problem_type == "planning_reasoning":
            subgoals = [
                "identify constraints",
                "enumerate candidate actions",
                "select the best plan under constraints",
            ]
        elif problem_type == "stepwise_problem_solving":
            subgoals = [
                "decompose into smaller steps",
                "solve substeps in order",
                "combine into final result",
            ]
        else:
            subgoals = [
                "identify the task",
                "collect relevant reasoning points",
                "produce a concise conclusion",
            ]

        return ReasoningState(problem_type=problem_type, subgoals=subgoals)

    def refinement_step(self, state: ReasoningState, query: str) -> ReasoningState:
        step_id = state.steps_used

        if step_id < len(state.subgoals):
            active_subgoal = state.subgoals[step_id]
        else:
            active_subgoal = "refine and compress final answer"

        evidence_item = f"STEP{step_id+1}: addressed '{active_subgoal}'"
        state.evidence.append(evidence_item)

        if state.problem_type == "planning_reasoning":
            state.partial_conclusion = (
                "Current plan prefers options that minimize risk and unnecessary steps."
            )
        elif state.problem_type == "explanatory_reasoning":
            state.partial_conclusion = (
                "Current explanation emphasizes the main mechanism rather than surface detail."
            )
        elif state.problem_type == "stepwise_problem_solving":
            state.partial_conclusion = (
                "Current reasoning decomposes the problem and preserves intermediate consistency."
            )
        else:
            state.partial_conclusion = (
                "Current reasoning isolates the main structure of the query and compresses it."
            )

        state.steps_used += 1
        return state

    def finalize_answer(self, state: ReasoningState, query: str) -> str:
        evidence_summary = " | ".join(
            state.evidence[-3:]) if state.evidence else "no evidence"
        return (
            f"REASONED_RESPONSE: {state.partial_conclusion} "
            f"Final answer generated after {state.steps_used} reasoning steps. "
            f"Evidence trace: {evidence_summary}"
        )

    def run(self, backbone_state: str, query: str, token_budget: int) -> tuple[ReasoningState, str]:
        state = self.initialize_state(query)

        # budget here is interpreted as the number of refinement iterations allowed
        while token_budget > 0:
            state = self.refinement_step(state, query)
            token_budget -= 1

        final_answer = self.finalize_answer(state, query)
        return state, final_answer


def needs_reasoning(query: str) -> bool:
    q = query.lower()
    keywords = [
        "why",
        "prove",
        "reason",
        "plan",
        "multi-step",
        "solve",
        "compare",
        "analyze",
        "tradeoff",
        "best option",
    ]
    return any(k in q for k in keywords)


class EdgeReasoningSystem:
    def __init__(self):
        self.backbone = FrozenBackbone()
        self.fast_adapter = FastAdapter()
        self.reasoning_adapter = ReasoningAdapter()

    def run_query(self, prompt: str, query: str, token_budget: int):
        cache = self.backbone.encode_prompt(prompt)
        backbone_state = self.backbone.build_query_context(cache, query)

        if needs_reasoning(query):
            state, answer = self.reasoning_adapter.run(
                backbone_state=backbone_state,
                query=query,
                token_budget=token_budget,
            )
            return {
                "mode": "reasoning",
                "prompt_cache": cache,
                "backbone_state": backbone_state,
                "reasoning_state": state,
                "answer": answer,
            }

        fast_state = self.fast_adapter.run(
            backbone_state=backbone_state, query=query)
        return {
            "mode": "fast",
            "prompt_cache": cache,
            "backbone_state": backbone_state,
            "fast_state": fast_state,
            "answer": fast_state.answer,
        }


def main() -> None:
    system = EdgeReasoningSystem()

    prompt = "You are an edge assistant optimized for concise helpful responses."

    queries_and_budgets = [
        ("What time does the store close?", 0),
        ("Why is the sky blue? Please reason step by step.", 2),
        ("Solve this multi-step planning task for my commute.", 4),
    ]

    print("=== Efficient Reasoning on the Edge backbone demo ===\n")

    for query, budget in queries_and_budgets:
        result = system.run_query(prompt, query, token_budget=budget)

        print(f"Query        : {query}")
        print(f"Mode         : {result['mode']}")
        print(f"Token budget : {budget}")

        if result["mode"] == "fast":
            fast_state: FastState = result["fast_state"]
            print(f"Fast type    : {fast_state.query_type}")
            print(f"Focus        : {fast_state.extracted_focus}")
            print(f"Answer       : {result['answer']}")
        else:
            reasoning_state: ReasoningState = result["reasoning_state"]
            print(f"Problem type : {reasoning_state.problem_type}")
            print(f"Subgoals     : {reasoning_state.subgoals}")
            print(f"Evidence     : {reasoning_state.evidence}")
            print(f"Partial concl: {reasoning_state.partial_conclusion}")
            print(f"Steps used   : {reasoning_state.steps_used}")
            print(f"Answer       : {result['answer']}")

        print()

    print("Interpretation:")
    print("- Prompt encoding is shared once through a reusable cache.")
    print("- Easy queries go through a lightweight fast adapter with query typing and concise output.")
    print("- Harder queries activate a reasoning adapter with explicit subgoals and iterative refinement.")
    print("- Budget forcing controls how many refinement steps are allowed.")
    print("- This is the core backbone of conditional, concise edge reasoning.")


if __name__ == "__main__":
    main()
