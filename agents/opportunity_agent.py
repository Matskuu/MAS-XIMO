"""Additional improvement opportunity agent for MAS-XIMO."""

from __future__ import annotations

import json

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

from explanations.utils import get_objective_symbols


class OpportunityAgent(Agent):
    """Identify beneficial changes in non-target objectives.

    The opportunity agent examines a validated counterfactual RPM result and
    identifies objectives that were not selected as targets but also improved
    under the tested preference change.

    Opportunities are reported only when the counterfactual did not worsen any
    selected target objective. The agent does not modify the decision maker's
    target set.
    """

    def __init__(
        self,
        jid,
        password,
        problem,
        *,
        tolerance: float = 1e-6,
        **kwargs,
    ):
        super().__init__(
            jid,
            password,
            **kwargs,
        )

        self.problem = problem
        self.objective_symbols = get_objective_symbols(problem)
        self.tolerance = tolerance

    def analyse_opportunities(
        self,
        original_solution_min: list[float],
        adjusted_solution_min: list[float],
        original_solution_original: list[float],
        adjusted_solution_original: list[float],
        targets: list[str],
        counterfactual_outcome: str,
    ) -> dict:
        """Identify improved non-target objectives.

        Parameters
        ----------
        original_solution_min
            Original RPM solution in the common minimization orientation.
        adjusted_solution_min
            Counterfactual RPM solution in the common minimization orientation.
        targets
            Objective symbols selected by the decision maker.
        counterfactual_outcome
            Outcome returned by the counterfactual validation.

        Returns
        -------
        dict
            Serializable description of detected improvement opportunities.
        """
        target_set = set(targets)

        # Additional improvements are considered compatible only if the tested
        # preference change respected the selected target objectives.
        target_compatible = counterfactual_outcome in {
            "joint_improvement",
            "partial_improvement",
        }

        opportunities = []

        if target_compatible:
            for index, symbol in enumerate(
                self.objective_symbols
            ):
                if symbol in target_set:
                    continue

                original = float(
                    original_solution_min[index]
                )
                adjusted = float(
                    adjusted_solution_min[index]
                )

                # All solutions are already in the common minimization
                # orientation, so a smaller value is an improvement.
                improvement = original - adjusted

                if improvement > self.tolerance:
                    opportunities.append(
                        {
                            "symbol": symbol,
                            "improvement_min": improvement,
                            "original_value": float(
                                original_solution_original[index]
                            ),
                            "adjusted_value": float(
                                adjusted_solution_original[index]
                            ),
                        }
                    )

        return {
            "type": "opportunity_analysis",
            "target_compatible": target_compatible,
            "opportunities": opportunities,
        }

    class ReceiveOpportunityRequests(
        CyclicBehaviour
    ):
        """Receive requests to inspect validated counterfactual results."""

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
                    "Opportunity agent received "
                    "invalid JSON."
                )
                return

            if (
                contents.get("type")
                != "analyse_opportunities"
            ):
                return

            try:
                result = (
                    self.agent.analyse_opportunities(
                        original_solution_min=contents[
                            "original_solution_min"
                        ],
                        adjusted_solution_min=contents[
                            "adjusted_solution_min"
                        ],
                        original_solution_original=contents[
                            "original_solution_original"
                        ],
                        adjusted_solution_original=contents[
                            "adjusted_solution_original"
                        ],
                        targets=contents[
                            "targets"
                        ],
                        counterfactual_outcome=contents[
                            "counterfactual_outcome"
                        ],
                    )
                )

            except Exception as error:
                print(
                    "Could not analyse additional "
                    "improvement opportunities: "
                    f"{error}"
                )
                return

            response = Message(
                to="coordinator@localhost",
                body=json.dumps(result),
                metadata={
                    "performative": "inform"
                },
            )

            await self.send(response)

    async def setup(self):
        """Initialize opportunity-agent behaviours."""
        print(
            "Opportunity agent started."
        )

        self.add_behaviour(
            self.ReceiveOpportunityRequests(),
            Template(
                metadata={
                    "performative": "request"
                }
            ),
        )
