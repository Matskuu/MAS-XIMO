"""Counterfactual validation agent for MAS-XIMO."""

from __future__ import annotations

import json

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from explanations.counterfactual_validation import (
    build_counterfactual_reference_point,
    evaluate_counterfactual_solution,
    outcome_to_dict,
)
from explanations.utils import get_rximo_symbols


class CounterfactualAgent(Agent):
    """Validate Owen-based preference-change suggestions.

    The agent constructs a counterfactual reference point from an R-XIMO
    suggestion, asks the solver agent to solve it, and determines whether the
    selected target objectives improve together.
    """

    def __init__(
        self,
        jid,
        password,
        problem,
        *,
        step_fraction: float = 0.10,
        tolerance: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            jid,
            password,
            **kwargs,
        )

        self.problem = problem

        self.step_fraction = step_fraction
        self.tolerance = tolerance

        self.input_symbols, _ = (
            get_rximo_symbols(problem)
        )

        self.pending_requests: dict[
            str,
            dict,
        ] = {}

        self.request_counter = 0

    def next_request_id(self) -> str:
        """Return a unique counterfactual request identifier."""
        self.request_counter += 1

        return (
            f"counterfactual_"
            f"{self.request_counter}"
        )

    class ReceiveValidationRequests(
        CyclicBehaviour
    ):
        """Receive requests to validate Owen suggestions."""

        async def run(self):
            message = await self.receive(
                timeout=10
            )

            if message is None:
                return

            try:
                contents = json.loads(
                    message.body
                )

            except json.JSONDecodeError:
                print(
                    "Counterfactual agent received "
                    "invalid JSON."
                )
                return

            if (
                contents.get("type")
                != "validate_counterfactual"
            ):
                return

            request_id = (
                self.agent.next_request_id()
            )

            try:
                adjusted_reference_point = (
                    build_counterfactual_reference_point(
                        problem=self.agent.problem,
                        reference_point_original=(
                            contents[
                                "reference_point"
                            ]
                        ),
                        input_symbols=(
                            self.agent.input_symbols
                        ),
                        target_members=(
                            contents.get(
                                "actionable_target_members",
                                [],
                            )
                        ),
                        rival_members=(
                            contents.get(
                                "actionable_rival_members",
                                [],
                            )
                        ),
                        step_fraction=(
                            self.agent.step_fraction
                        ),
                    )
                )

            except Exception as error:
                print(
                    "Could not construct "
                    "counterfactual reference point: "
                    f"{error}"
                )
                return

            self.agent.pending_requests[
                request_id
            ] = {
                "targets": contents["targets"],
                "original_solution_original": (
                    contents["solution_original"]
                ),
                "original_solution_min": (
                    contents["solution_min"]
                ),
                "adjusted_reference_point": (
                    adjusted_reference_point
                ),
            }

            request = Message(
                to="coordinator@localhost",
                body=json.dumps(
                    {
                        "type": (
                            "counterfactual_reference_point"
                        ),
                        "request_id": request_id,
                        "reference_point": (
                            adjusted_reference_point.tolist()
                        ),
                    }
                ),
                metadata={
                    "performative": "request"
                },
            )

            await self.send(request)

    class ReceiveCounterfactualSolutions(
        CyclicBehaviour
    ):
        """Receive counterfactual RPM solutions."""

        async def run(self):
            message = await self.receive(
                timeout=10
            )

            if message is None:
                return

            try:
                contents = json.loads(
                    message.body
                )

            except json.JSONDecodeError:
                print(
                    "Counterfactual agent received "
                    "invalid JSON."
                )
                return

            if (
                contents.get("type")
                != "counterfactual_solution"
            ):
                return

            request_id = contents.get(
                "request_id"
            )

            pending = (
                self.agent.pending_requests.pop(
                    request_id,
                    None,
                )
            )

            if pending is None:
                print(
                    "Counterfactual agent received "
                    "an unknown request ID: "
                    f"{request_id}"
                )
                return

            try:
                outcome = (
                    evaluate_counterfactual_solution(
                        problem=self.agent.problem,
                        original_solution_min=(
                            pending[
                                "original_solution_min"
                            ]
                        ),
                        adjusted_solution_min=(
                            contents[
                                "solution_min"
                            ]
                        ),
                        target_symbols=(
                            pending["targets"]
                        ),
                        adjusted_reference_point=(
                            pending[
                                "adjusted_reference_point"
                            ]
                        ),
                        tolerance=(
                            self.agent.tolerance
                        ),
                    )
                )

            except Exception as error:
                print(
                    "Could not evaluate "
                    "counterfactual solution: "
                    f"{error}"
                )
                return

            result = outcome_to_dict(outcome)

            result.update(
                {
                    "type": (
                        "counterfactual_validation"
                    ),
                    "request_id": request_id,
                    "original_solution_original": (
                        pending[
                            "original_solution_original"
                        ]
                    ),
                    "adjusted_solution_original": (
                        contents[
                            "solution_original"
                        ]
                    ),
                    "original_solution_min": (
                        pending[
                            "original_solution_min"
                        ]
                    ),
                    "adjusted_solution_min": (
                        contents[
                            "solution_min"
                        ]
                    ),
                }
            )

            response = Message(
                to="coordinator@localhost",
                body=json.dumps(result),
                metadata={
                    "performative": "inform"
                },
            )

            await self.send(response)

    async def setup(self):
        """Initialize counterfactual-agent behaviours."""
        print(
            "Counterfactual agent started."
        )

        self.add_behaviour(
            self.ReceiveValidationRequests(),
            Template(
                metadata={
                    "performative": "request"
                }
            ),
        )

        self.add_behaviour(
            self.ReceiveCounterfactualSolutions(),
            Template(
                metadata={
                    "performative": "inform"
                }
            ),
        )
