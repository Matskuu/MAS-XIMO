"""Multi-target Owen explanation agent for MAS-XIMO."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from explanations.owen_service import OwenExplanationService


class OwenAgent(Agent):
    """Provide multi-target Owen explanations."""

    def __init__(
        self,
        jid,
        password,
        problem,
        background_data_path: str | Path,
        *,
        surrogate_type: str = "gaussian_process",
        seed: int = 1,
        background_size: int = 100,
        coalition_advantage_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(jid, password, **kwargs)

        self.problem = problem
        self.background_data_path = Path(background_data_path)

        self.surrogate_type = surrogate_type
        self.seed = seed
        self.background_size = background_size
        self.coalition_advantage_threshold = (
            coalition_advantage_threshold
        )

        self.service: OwenExplanationService | None = None

    class SendReady(OneShotBehaviour):
        """Inform the coordinator that the Owen service is ready."""

        async def run(self):
            message = Message(
                to="coordinator@localhost",
                body=json.dumps(
                    {
                        "type": "owen_ready",
                    }
                ),
                metadata={"performative": "inform"},
            )

            await self.send(message)

    class ReceiveExplanationRequests(CyclicBehaviour):
        """Receive requests for multi-target explanations."""

        async def run(self):
            message = await self.receive(timeout=10)

            if message is None:
                return

            try:
                contents = json.loads(message.body)
            except json.JSONDecodeError:
                print("Owen agent received invalid JSON.")
                return

            if contents.get("type") != "explain":
                return

            try:
                result = await asyncio.to_thread(
                    self.agent.service.explain,
                    reference_point_original=contents["reference_point"],
                    solution_min=contents["solution_min"],
                    targets=contents["targets"],
                )

            except Exception as error:
                print(f"Could not generate Owen explanation: {error}")
                return

            response = Message(
                to="coordinator@localhost",
                body=json.dumps(result),
                metadata={"performative": "inform"},
            )

            await self.send(response)

    async def setup(self):
        """Prepare the Owen explanation service and receive requests."""
        print("Owen agent starting...")

        # Training can be computationally expensive, so do it once.
        self.service = await asyncio.to_thread(
            OwenExplanationService,
            problem=self.problem,
            background_data_path=self.background_data_path,
            surrogate_type=self.surrogate_type,
            seed=self.seed,
            background_size=self.background_size,
            coalition_advantage_threshold=(
                self.coalition_advantage_threshold
            ),
        )

        print("Owen agent started.")

        self.add_behaviour(
            self.ReceiveExplanationRequests(),
            Template(metadata={"performative": "request"}),
        )

        self.add_behaviour(
            self.SendReady()
        )
