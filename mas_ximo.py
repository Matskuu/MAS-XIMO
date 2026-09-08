"""Run the MAS-XIMO multi-agent system."""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
from pathlib import Path

import spade
from agents.coordinator_agent import CoordinatorAgent
from agents.counterfactual_agent import CounterfactualAgent
from agents.opportunity_agent import OpportunityAgent
from agents.owen_agent import OwenAgent
from agents.preference_agent import PreferenceAgent
from agents.solver_agent import SolverAgent
from generate_background_data import generate_background_data
from problem_setup import (
    PROBLEM_CHOICES,
    create_problem,
    default_background_path,
)


def parse_args() -> argparse.Namespace:
    """Parse MAS-XIMO experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Run MAS-XIMO with a selected DESDEO test problem."
    )
    parser.add_argument(
        "--problem",
        choices=PROBLEM_CHOICES,
        default="river",
        help="Test problem to use (default: river).",
    )
    parser.add_argument(
        "--background-data",
        type=Path,
        default=None,
        help="Optional Owen background CSV. Uses the problem default otherwise.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--background-samples",
        type=int,
        default=300,
        help=(
            "Number of RPM samples to generate when the background CSV "
            "is missing (default: 300)."
        ),
    )
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument(
        "--surrogate-type",
        default="gaussian_process",
    )
    parser.add_argument(
        "--coalition-advantage-threshold",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    """Start MAS-XIMO."""
    problem = create_problem(args.problem)
    data_path = args.background_data or default_background_path(
        args.problem,
        seed=args.seed,
        samples=args.background_samples,
    )

    print(f"MAS-XIMO problem: {args.problem}")
    print(f"Owen background data: {data_path}")

    if data_path.exists():
        print("Background dataset found. Using existing data.")
    else:
        print(
            "Background dataset not found. "
            f"Generating {args.background_samples} RPM samples "
            f"with seed {args.seed}..."
        )
        generate_background_data(
            problem_name=args.problem,
            samples=args.background_samples,
            seed=args.seed,
            output_path=data_path,
        )
        print("Background dataset ready.")

    solver_agent = SolverAgent(
        "solver@localhost",
        "solver",
        problem=problem,
    )

    counterfactual_agent = CounterfactualAgent(
        "counterfactualagent@localhost",
        "counterfactualagent",
        problem=problem,
        step_fraction=0.10,
        tolerance=1e-6,
    )

    coordinator_agent = CoordinatorAgent(
        "coordinator@localhost",
        "coordinator",
    )

    opportunity_agent = OpportunityAgent(
        "opportunityagent@localhost",
        "opportunityagent",
        problem=problem,
    )

    owen_agent = OwenAgent(
        "owenagent@localhost",
        "owenagent",
        problem=problem,
        background_data_path=data_path,
        surrogate_type=args.surrogate_type,
        seed=args.seed,
        background_size=args.background_size,
        coalition_advantage_threshold=(
            args.coalition_advantage_threshold
        ),
    )

    preference_agent = PreferenceAgent(
        "preferenceagent@localhost",
        "preferenceagent",
        problem=problem,
    )

    await coordinator_agent.start(auto_register=True)
    await counterfactual_agent.start(auto_register=True)
    await solver_agent.start(auto_register=True)
    await opportunity_agent.start(auto_register=True)
    await owen_agent.start(auto_register=True)
    await preference_agent.start(auto_register=True)

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping agents...")
    finally:
        await preference_agent.stop()
        await opportunity_agent.stop()
        await owen_agent.stop()
        await solver_agent.stop()
        await coordinator_agent.stop()
        await counterfactual_agent.stop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    cli_args = parse_args()
    spade.run(
        main(cli_args),
        embedded_xmpp_server=True,
    )
