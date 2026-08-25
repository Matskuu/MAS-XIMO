"""Coordinator agent for MAS-XIMO."""

from __future__ import annotations

import json

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template


class CoordinatorAgent(Agent):
    """Coordinate communication between the MAS-XIMO agents."""

    def __init__(self, jid, password, **kwargs):
        super().__init__(jid, password, **kwargs)

        self.reference_point: list[float] | None = None
        self.solution_min: list[float] | None = None
        self.targets: list[str] | None = None

        self.explanation_requested = False

    async def request_solution(self, behaviour) -> None:
        """Send the current reference point to the solver agent."""
        if self.reference_point is None:
            return

        message = Message(
            to="solver@localhost",
            body=json.dumps(
                {
                    "type": "solve_reference_point",
                    "reference_point": self.reference_point,
                }
            ),
            metadata={"performative": "request"},
        )

        await behaviour.send(message)

    async def request_explanation_if_ready(
        self,
        behaviour,
    ) -> None:
        """Request an Owen explanation when all data are available."""
        if self.explanation_requested:
            return

        if (
            self.reference_point is None
            or self.solution_min is None
            or not self.targets
        ):
            return

        message = Message(
            to="owenagent@localhost",
            body=json.dumps(
                {
                    "type": "explain",
                    "reference_point": self.reference_point,
                    "solution_min": self.solution_min,
                    "targets": self.targets,
                }
            ),
            metadata={"performative": "request"},
        )

        await behaviour.send(message)

        self.explanation_requested = True

    class ReceiveMessages(CyclicBehaviour):
        """Receive information from the other agents."""

        async def run(self):
            message = await self.receive(timeout=10)

            if message is None:
                return

            try:
                contents = json.loads(message.body)
            except json.JSONDecodeError:
                print(
                    "Coordinator received a message with invalid JSON "
                    f"from {message.sender}."
                )
                return

            message_type = contents.get("type")

            if message_type == "reference_point":
                self.agent.reference_point = contents["reference_point"]

                self.agent.solution_min = None
                self.agent.targets = None
                self.agent.explanation_requested = False

                await self.agent.request_solution(self)

            elif message_type == "targets":
                self.agent.targets = contents["targets"]
                self.agent.explanation_requested = False

                await self.agent.request_explanation_if_ready(self)

            elif message_type == "solution":
                self.agent.solution_min = contents["solution_min"]

                # Forward the solution to the preference agent so the DM can
                # see what RPM produced.
                response = Message(
                    to="preferenceagent@localhost",
                    body=json.dumps(
                        {
                            "type": "solution",
                            "reference_point": self.agent.reference_point,
                            "solution_original": contents[
                                "solution_original"
                            ],
                        }
                    ),
                    metadata={"performative": "inform"},
                )

                await self.send(response)

                await self.agent.request_explanation_if_ready(self)

            elif message_type == "owen_explanation":
                self.agent.explanation_requested = False

                response = Message(
                    to="preferenceagent@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"},
                )

                await self.send(response)

            elif message_type == "owen_ready":
                print("Owen agent is ready.")

            else:
                print(
                    "Coordinator received an unknown message type "
                    f"'{message_type}' from {message.sender}."
                )

    async def setup(self):
        """Initialize coordinator behaviours."""
        print("Coordinator agent started.")

        template = Template()

        self.add_behaviour(
            self.ReceiveMessages(),
            template,
        )
