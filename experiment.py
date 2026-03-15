"""
Multi-Agent Sycophancy Experiment
==================================
Runs code-review conversations with either 2 normal agents (baseline) or
2 normal + 1 sycophantic agent (experimental) to study false-claim propagation.

Usage:
    python experiment.py [--trials N] [--task binary_search|count_vowels|all]
                         [--condition baseline|experimental|all]
"""

import argparse
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from config import (
    CLAUDE_MODEL,
    OPENAI_MODEL,
    GEMINI_MODEL,
    NORMAL_PROMPT,
    SYCOPHANTIC_PROMPT_TEMPLATE,
    TASKS,
    NUM_TURNS,
    MIN_TURNS_BEFORE_CONSENSUS,
    CONSENSUS_TOKEN,
)
from utils import (
    call_claude,
    call_openai,
    call_gemini,
    format_messages_for_claude,
    format_messages_for_openai,
    format_messages_for_gemini,
    build_log_entry,
    save_log,
    analyze_experiment,
)

# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENTS = {
    "baseline": [
        {"name": "claude_haiku", "alias": "Reviewer A", "provider": "claude", "role": "normal",      "model": CLAUDE_MODEL},
        {"name": "gpt4o_mini",   "alias": "Reviewer B", "provider": "openai",  "role": "normal",      "model": OPENAI_MODEL},
    ],
    "experimental": [
        {"name": "claude_haiku", "alias": "Reviewer A", "provider": "claude",  "role": "normal",      "model": CLAUDE_MODEL},
        {"name": "gpt4o_mini",   "alias": "Reviewer B", "provider": "openai",  "role": "normal",      "model": OPENAI_MODEL},
        {"name": "gemini_flash", "alias": "Reviewer C", "provider": "gemini",  "role": "sycophantic", "model": GEMINI_MODEL},
    ],
}

# Approximate cost per 1 M tokens (USD) — update as pricing changes
_COST_PER_M = {
    "claude":  {"input": 0.80,  "output": 4.00},   # Haiku 3.5
    "openai":  {"input": 0.15,  "output": 0.60},   # gpt-4o-mini
    "gemini":  {"input": 0.075, "output": 0.30},   # gemini-1.5-flash
}

# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

_CALLERS = {
    "claude": call_claude,
    "openai": call_openai,
    "gemini": call_gemini,
}

_FORMATTERS = {
    "claude": lambda history, _sys: format_messages_for_claude(history),
    "openai": format_messages_for_openai,
    "gemini": lambda history, _sys: format_messages_for_gemini(history),
}


_HISTORY_WINDOW = 6  # keep first message (task) + last N-1 messages


def _call_agent(agent: dict, history: list, system_prompt: str) -> dict:
    provider = agent["provider"]
    # Always keep the opening task message; truncate the rest to avoid context explosion
    if len(history) > _HISTORY_WINDOW:
        windowed = [history[0]] + history[-(  _HISTORY_WINDOW - 1):]
    else:
        windowed = history
    messages = _FORMATTERS[provider](windowed, system_prompt)
    return _CALLERS[provider](messages, system_prompt, agent["model"])


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(task_name: str, condition: str, trial: int = 0) -> str:
    """Run a single experiment and return the path to the saved log file."""
    task = TASKS[task_name]
    agents = AGENTS[condition]

    experiment_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\n{'='*60}")
    print(f"Experiment: {task_name} | {condition} | trial {trial}")
    print(f"ID: {experiment_id}")
    print(f"{'='*60}")

    # Build the opening user message
    opening = (
        f"Please review the following code:\n\n"
        f"```python\n{task['code']}```\n\n"
        f"{task['question']}"
    )

    # Build the sycophantic system prompt with the task-specific false claim
    sycophantic_prompt = SYCOPHANTIC_PROMPT_TEMPLATE.format(
        false_claim=task["false_claim"]
    )

    # Conversation history: list of {"role": ..., "agent": ..., "content": ...}
    history: list[dict] = [{"role": "user", "agent": "user", "content": opening}]

    log_entries: list[dict] = []
    token_totals: dict[str, dict] = {a["provider"]: {"input": 0, "output": 0} for a in agents}
    consensus_votes: set[str] = set()
    stopped_early = False

    for turn in range(1, NUM_TURNS + 1):
        for agent in agents:
            system_prompt = (
                sycophantic_prompt if agent["role"] == "sycophantic" else NORMAL_PROMPT
            )
            result = _call_agent(agent, history, system_prompt)

            prompt_summary = (
                f"[system: {system_prompt[:80]}...]\n"
                f"[history: {len(history)} messages]"
            )

            entry = build_log_entry(
                turn=turn,
                agent_name=agent["name"],
                model=agent["model"],
                prompt_sent=prompt_summary,
                response=result["response"],
                thinking=result["thinking"],
                tokens_in=result["tokens_input"],
                tokens_out=result["tokens_output"],
            )
            log_entries.append(entry)

            token_totals[agent["provider"]]["input"] += result["tokens_input"]
            token_totals[agent["provider"]]["output"] += result["tokens_output"]

            if result["response"]:
                history.append({
                    "role": "assistant",
                    "agent": agent["name"],
                    "alias": agent["alias"],
                    "content": result["response"],
                })

            # Track consensus votes
            if CONSENSUS_TOKEN in (result["response"] or ""):
                consensus_votes.add(agent["name"])

            total_tokens = result["tokens_input"] + result["tokens_output"]
            print(
                f"  Turn {turn}/{NUM_TURNS} | {agent['name']:15s} | "
                f"{total_tokens:5d} tokens"
                + (f" [CONSENSUS vote {len(consensus_votes)}/{len(agents)}]"
                   if agent["name"] in consensus_votes else "")
            )

        # After all agents in this turn: stop if every agent has voted consensus
        # but only after the minimum number of turns has passed
        if turn >= MIN_TURNS_BEFORE_CONSENSUS and len(consensus_votes) >= len(agents):
            print(f"  → All agents reached consensus. Stopping at turn {turn}.")
            stopped_early = True
            break

    # ------------------------------------------------------------------
    # Save log
    # ------------------------------------------------------------------
    log_path = f"logs/{task_name}_{condition}_{timestamp[:19].replace(':', '-')}.json"

    log_data = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "task": task_name,
        "condition": condition,
        "trial": trial,
        "opening_prompt": opening,
        "stopped_early": stopped_early,
        "agents": [
            {"name": a["name"], "alias": a["alias"], "provider": a["provider"], "role": a["role"], "model": a["model"]}
            for a in agents
        ],
        "conversation": log_entries,
        "analysis": {},  # filled after saving
    }

    # Analyse and embed results
    save_log(log_data, log_path)
    analysis = analyze_experiment(log_path)
    log_data["analysis"] = analysis
    save_log(log_data, log_path)

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------
    print("\n  Token usage & estimated cost:")
    for provider, counts in token_totals.items():
        rates = _COST_PER_M.get(provider, {"input": 0, "output": 0})
        cost = (counts["input"] * rates["input"] + counts["output"] * rates["output"]) / 1_000_000
        print(
            f"    {provider:8s} in={counts['input']:6d}  out={counts['output']:6d}  "
            f"≈${cost:.4f}"
        )

    # ------------------------------------------------------------------
    # Print analysis summary
    # ------------------------------------------------------------------
    print("\n  Analysis summary:")
    for k, v in analysis.items():
        print(f"    {k}: {v}")

    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-agent sycophancy experiments.")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per combination")
    parser.add_argument(
        "--task",
        choices=list(TASKS.keys()) + ["all"],
        default="all",
        help="Task(s) to run",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "experimental", "all"],
        default="all",
        help="Condition(s) to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    selected_tasks = list(TASKS.keys()) if args.task == "all" else [args.task]
    selected_conditions = (
        ["baseline", "experimental"] if args.condition == "all" else [args.condition]
    )

    log_files = []
    for trial in range(args.trials):
        for task in selected_tasks:
            for condition in selected_conditions:
                path = run_experiment(task, condition, trial)
                log_files.append(path)

    print(f"\n{'='*60}")
    print(f"All done. {len(log_files)} log file(s) written:")
    for p in log_files:
        print(f"  {p}")


if __name__ == "__main__":
    main()
