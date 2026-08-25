"""RPM solver agent for MAS-XIMO."""

from __future__ import annotations

import asyncio
import json

import numpy as np

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from explanations.utils import (
    orient_objectives_from_minimize,
    solve_reference_point_with_desdeo_rpm,
)


class SolverAgent(Agent):
    """Solve reference points using DESDEO's reference point method."""

    def __init__(self, jid, password, problem, **kwargs):
        super().__init__(jid, password, **kwargs)

        self.problem = problem

    class ReceiveSolveRequests(CyclicBehaviour):
        """Receive and solve reference-point requests."""

        async def run(self):
            message = await self.receive(timeout=10)

            if message is None:
                return

            try:
                contents = json.loads(message.body)
            except json.JSONDecodeError:
                print("Solver received invalid JSON.")
                return

            if contents.get("type") != "solve_reference_point":
                return

            reference_point = np.asarray(
                contents["reference_point"],
                dtype=float,
            )

            try:
                solution_min = await asyncio.to_thread(
                    solve_reference_point_with_desdeo_rpm,
                    self.agent.problem,
                    reference_point,
                )

            except Exception as error:
                print(
                    "Could not solve reference point "
                    f"{reference_point}: {error}"
                )
                return

            solution_original = orient_objectives_from_minimize(
                self.agent.problem,
                solution_min,
            )

            response = Message(
                to="coordinator@localhost",
                body=json.dumps(
                    {
                        "type": "solution",
                        "reference_point": reference_point.tolist(),
                        "solution_min": solution_min.tolist(),
                        "solution_original": solution_original.tolist(),
                    }
                ),
                metadata={"performative": "inform"},
            )

            await self.send(response)

    async def setup(self):
        """Initialize solver behaviours."""
        print("Solver agent started.")

        self.add_behaviour(
            self.ReceiveSolveRequests(),
            Template(metadata={"performative": "request"}),
        )
