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
        self.solution_original: list[float] | None = None
        self.solution_min: list[float] | None = None
        self.targets: list[str] | None = None

        self.latest_explanation: dict | None = None
        self.latest_counterfactual: dict | None = None

        self.explanation_requested = False
        self.counterfactual_requested = False
        self.opportunity_requested = False

    async def request_solution(
        self,
        behaviour,
    ) -> None:
        """Send the current reference point to the solver agent."""
        if self.reference_point is None:
            return

        message = Message(
            to="solver@localhost",
            body=json.dumps(
                {
                    "type": "solve_reference_point",
                    "reference_point": self.reference_point,
                    "purpose": "decision",
                }
            ),
            metadata={"performative": "request"},
        )

        await behaviour.send(message)

    async def request_opportunity_analysis(
        self,
        behaviour,
        validation: dict,
    ) -> None:
        """Request analysis of non-target improvement opportunities."""
        if self.opportunity_requested:
            return

        if (
            not self.targets
            or "original_solution_min" not in validation
            or "adjusted_solution_min" not in validation
        ):
            return

        message = Message(
            to="opportunityagent@localhost",
            body=json.dumps(
                {
                    "type": "analyse_opportunities",
                    "targets": self.targets,
                    "original_solution_original": validation[
                        "original_solution_original"
                    ],
                    "adjusted_solution_original": validation[
                        "adjusted_solution_original"
                    ],
                    "original_solution_min": validation[
                        "original_solution_min"
                    ],
                    "adjusted_solution_min": validation[
                        "adjusted_solution_min"
                    ],
                    "counterfactual_outcome": validation.get(
                        "outcome_type"
                    ),
                }
            ),
            metadata={
                "performative": "request"
            },
        )

        await behaviour.send(message)

        self.opportunity_requested = True

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

    async def request_counterfactual_validation(
        self,
        behaviour,
    ) -> None:
        """Request validation of the latest Owen suggestion."""
        if self.counterfactual_requested:
            return

        if (
            self.reference_point is None
            or self.solution_min is None
            or not self.targets
            or self.latest_explanation is None
        ):
            return

        message = Message(
            to="counterfactualagent@localhost",
            body=json.dumps(
                {
                    "type": "validate_counterfactual",
                    "reference_point": self.reference_point,
                    "solution_original": self.solution_original,
                    "solution_min": self.solution_min,
                    "targets": self.targets,

                    "target_members": self.latest_explanation.get(
                        "target_members",
                        [],
                    ),
                    "actionable_target_members": (
                        self.latest_explanation.get(
                            "actionable_target_members",
                            [],
                        )
                    ),
                    "non_actionable_target_members": (
                        self.latest_explanation.get(
                            "non_actionable_target_members",
                            [],
                        )
                    ),

                    "rival_members": self.latest_explanation.get(
                        "rival_members",
                        [],
                    ),
                    "selected_action_rival_members": (
                        self.latest_explanation.get(
                            "selected_action_rival_members",
                            [],
                        )
                    ),
                    "actionable_rival_members": (
                        self.latest_explanation.get(
                            "actionable_rival_members",
                            [],
                        )
                    ),
                    "non_actionable_rival_members": (
                        self.latest_explanation.get(
                            "non_actionable_rival_members",
                            [],
                        )
                    ),
                }
            ),
            metadata={"performative": "request"},
        )

        await behaviour.send(message)

        self.counterfactual_requested = True

    async def request_counterfactual_solution(
        self,
        behaviour,
        contents: dict,
    ) -> None:
        """Send a counterfactual reference point to the solver agent."""
        message = Message(
            to="solver@localhost",
            body=json.dumps(
                {
                    "type": "solve_reference_point",
                    "reference_point": contents["reference_point"],
                    "purpose": "counterfactual",
                    "request_id": contents["request_id"],
                }
            ),
            metadata={"performative": "request"},
        )

        await behaviour.send(message)

    async def forward_counterfactual_solution(
        self,
        behaviour,
        contents: dict,
    ) -> None:
        """Forward a counterfactual RPM solution to the validation agent."""
        message = Message(
            to="counterfactualagent@localhost",
            body=json.dumps(
                {
                    "type": "counterfactual_solution",
                    "request_id": contents["request_id"],
                    "reference_point": contents["reference_point"],
                    "solution_min": contents["solution_min"],
                    "solution_original": contents["solution_original"],
                }
            ),
            metadata={"performative": "inform"},
        )

        await behaviour.send(message)

    async def forward_complete_explanation(
        self,
        behaviour,
        opportunity_analysis: dict,
    ) -> None:
        """Send explanation, validation, and opportunities to the DM."""
        if (
            self.latest_explanation is None
            or self.latest_counterfactual is None
        ):
            print(
                "Coordinator cannot construct the complete "
                "explanation because required information is missing."
            )
            return

        combined = {
            **self.latest_explanation,
            "counterfactual": (
                self.latest_counterfactual
            ),
            "opportunities": (
                opportunity_analysis
            ),
        }

        response = Message(
            to="preferenceagent@localhost",
            body=json.dumps(combined),
            metadata={
                "performative": "inform"
            },
        )

        await behaviour.send(response)

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

                self.agent.solution_original = None
                self.agent.solution_min = None
                self.agent.targets = None
                self.agent.latest_explanation = None
                self.agent.latest_counterfactual = None

                self.agent.explanation_requested = False
                self.agent.counterfactual_requested = False
                self.agent.opportunity_requested = False

                await self.agent.request_solution(self)

            elif message_type == "targets":
                self.agent.targets = contents["targets"]

                self.agent.latest_explanation = None
                self.agent.latest_counterfactual = None

                self.agent.explanation_requested = False
                self.agent.counterfactual_requested = False
                self.agent.opportunity_requested = False

                await self.agent.request_explanation_if_ready(self)

            elif message_type == "solution":
                purpose = contents.get(
                    "purpose",
                    "decision",
                )

                if purpose == "counterfactual":
                    await self.agent.forward_counterfactual_solution(
                        self,
                        contents,
                    )
                    return

                self.agent.solution_original = contents["solution_original"]
                self.agent.solution_min = contents["solution_min"]

                # Forward the current decision solution to the preference agent.
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

                self.agent.latest_explanation = contents

                await self.agent.request_counterfactual_validation(
                    self
                )

            elif message_type == "counterfactual_reference_point":
                await self.agent.request_counterfactual_solution(
                    self,
                    contents,
                )

            elif message_type == "counterfactual_validation":
                self.agent.counterfactual_requested = False

                self.agent.latest_counterfactual = contents

                await self.agent.request_opportunity_analysis(
                    self,
                    contents,
                )

            elif message_type == "opportunity_analysis":
                self.agent.opportunity_requested = False

                await self.agent.forward_complete_explanation(
                    self,
                    contents,
                )

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
