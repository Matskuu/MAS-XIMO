"""Run the MAS-XIMO multi-agent system."""

from __future__ import annotations

import asyncio
import multiprocessing
from pathlib import Path

import spade

from agents.coordinator_agent import CoordinatorAgent
from agents.counterfactual_agent import CounterfactualAgent
from agents.owen_agent import OwenAgent
from agents.preference_agent import PreferenceAgent
from agents.solver_agent import SolverAgent
from problem_setup import create_problem


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "river_rximo_rpm_background_4obj_300samples_seed1.csv"
)


async def main():
    """Start MAS-XIMO."""
    problem = create_problem()

    solver_agent = SolverAgent(
        "solver@localhost",
        "solver",
        problem=problem,
    )

    counterfactual_agent = (
        CounterfactualAgent(
            "counterfactualagent@localhost",
            "counterfactualagent",
            problem=problem,
            step_fraction=0.10,
            tolerance=1e-6,
        )
    )

    coordinator_agent = CoordinatorAgent(
        "coordinator@localhost",
        "coordinator",
    )

    owen_agent = OwenAgent(
        "owenagent@localhost",
        "owenagent",
        problem=problem,
        background_data_path=DATA_PATH,
        surrogate_type="gaussian_process",
        seed=1,
        background_size=100,
        coalition_advantage_threshold=0.05,
    )

    preference_agent = PreferenceAgent(
        "preferenceagent@localhost",
        "preferenceagent",
        problem=problem,
    )

    # Start the coordinator first so that it is ready
    # to receive messages from the other agents.
    await coordinator_agent.start(
        auto_register=True
    )

    await counterfactual_agent.start(
        auto_register=True
    )

    await solver_agent.start(
        auto_register=True
    )

    await owen_agent.start(
        auto_register=True
    )

    await preference_agent.start(
        auto_register=True
    )

    try:
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("Stopping agents...")

    finally:
        await preference_agent.stop()
        await owen_agent.stop()
        await solver_agent.stop()
        await coordinator_agent.stop()
        await counterfactual_agent.stop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    spade.run(
        main(),
        embedded_xmpp_server=True
    )
