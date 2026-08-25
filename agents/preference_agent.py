"""Decision-maker preference agent for MAS-XIMO."""

from __future__ import annotations

import asyncio
import json

from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from explanations.utils import get_objective_symbols


class PreferenceAgent(Agent):
    """Handle interaction between the decision maker and MAS-XIMO.

    The preference agent collects reference points and target objectives from
    the decision maker and forwards them to the coordinator. It also presents
    RPM solutions and multi-target Owen explanations returned by the system.
    """

    def __init__(
        self,
        jid,
        password,
        problem,
        **kwargs,
    ):
        super().__init__(jid, password, **kwargs)

        self.problem = problem
        self.objective_symbols = get_objective_symbols(problem)

        self.reference_point: list[float] | None = None
        self.solution: list[float] | None = None
        self.targets: list[str] | None = None

        # Store the latest complete explanation so that the concise suggestion
        # can be shown by default and details can be requested afterwards.
        self.latest_explanation: dict | None = None

        self.waiting_for_solution = False
        self.waiting_for_explanation = False

    async def get_input(self, prompt: str) -> str:
        """Read terminal input without blocking SPADE's event loop."""
        return await asyncio.to_thread(input, prompt)

    def print_objective_information(self) -> None:
        """Print objective symbols, names, directions, and ranges."""
        print("\nObjectives")
        print("-" * 60)

        for index, objective in enumerate(
            self.problem.objectives,
            start=1,
        ):
            symbol = str(objective.symbol)
            name = getattr(objective, "name", symbol)

            ideal = float(objective.ideal)
            nadir = float(objective.nadir)

            maximize = bool(
                getattr(
                    objective,
                    "maximize",
                    False,
                )
            )

            direction = "maximize" if maximize else "minimize"

            print(
                f"{index}. {symbol} "
                f"({name}, {direction})"
            )
            print(
                f"   ideal = {ideal:.6f}, "
                f"nadir = {nadir:.6f}"
            )

        print()

    class InitialInteraction(OneShotBehaviour):
        """Start the first decision-maker interaction."""

        async def run(self):
            print()
            print(
                "Interactive multi-target Owen explanation loop"
            )
            print("=" * 60)
            print(
                "Reference points are entered in the original "
                "objective orientation."
            )
            print(
                "Multiple target objectives may be selected."
            )

            self.agent.print_objective_information()

            self.agent.add_behaviour(
                self.agent.AskReferencePoint()
            )

    class AskReferencePoint(OneShotBehaviour):
        """Ask the decision maker for a reference point."""

        async def run(self):
            reference_point = []

            print("\nEnter a reference point.")
            print(
                "Provide one value for each objective in "
                "the order shown above."
            )
            print(
                "You may also enter 'ideal' or 'nadir' "
                "for an individual component."
            )
            print()

            for objective in self.agent.problem.objectives:
                symbol = str(objective.symbol)
                name = getattr(
                    objective,
                    "name",
                    symbol,
                )

                ideal = float(objective.ideal)
                nadir = float(objective.nadir)

                lower = min(ideal, nadir)
                upper = max(ideal, nadir)

                while True:
                    value = await self.agent.get_input(
                        f"{symbol} ({name}) "
                        f"[{lower:.6f}, {upper:.6f}]: "
                    )

                    value = value.strip()

                    if value.lower() == "ideal":
                        reference_point.append(ideal)
                        break

                    if value.lower() == "nadir":
                        reference_point.append(nadir)
                        break

                    try:
                        numeric_value = float(value)
                    except ValueError:
                        print(
                            "Please enter a numerical value, "
                            "'ideal', or 'nadir'."
                        )
                        continue

                    if not lower <= numeric_value <= upper:
                        print(
                            "Value must be within the ideal-nadir "
                            f"range [{lower:.6f}, {upper:.6f}]."
                        )
                        continue

                    reference_point.append(
                        numeric_value
                    )
                    break

            self.agent.reference_point = (
                reference_point
            )

            # A new reference point starts a new interaction state.
            self.agent.solution = None
            self.agent.targets = None
            self.agent.latest_explanation = None

            message = Message(
                to="coordinator@localhost",
                body=json.dumps(
                    {
                        "type": "reference_point",
                        "reference_point": reference_point,
                    }
                ),
                metadata={
                    "performative": "inform"
                },
            )

            await self.send(message)

            self.agent.waiting_for_solution = True

            print(
                "\nReference point sent. "
                "Solving with RPM..."
            )

    class AskTargets(OneShotBehaviour):
        """Ask the decision maker for one or more target objectives."""

        async def run(self):
            available = self.agent.objective_symbols

            while True:
                value = await self.agent.get_input(
                    "\nProvide one or more target objectives "
                    "to improve "
                    f"(e.g. {available[0]} or "
                    f"{available[0]} {available[-1]}): "
                )

                requested = value.split()

                if not requested:
                    print(
                        "At least one target objective "
                        "must be selected."
                    )
                    continue

                invalid = [
                    target
                    for target in requested
                    if target not in available
                ]

                if invalid:
                    print(
                        "Unknown target objective(s): "
                        + ", ".join(invalid)
                    )
                    print(
                        "Available objectives: "
                        + ", ".join(available)
                    )
                    continue

                # Preserve the DM's order while removing duplicates.
                targets = list(
                    dict.fromkeys(requested)
                )

                break

            self.agent.targets = targets

            # The old explanation no longer corresponds to the selected
            # target set.
            self.agent.latest_explanation = None

            message = Message(
                to="coordinator@localhost",
                body=json.dumps(
                    {
                        "type": "targets",
                        "targets": targets,
                    }
                ),
                metadata={
                    "performative": "inform"
                },
            )

            await self.send(message)

            self.agent.waiting_for_explanation = True

            print(
                "\nTarget objective(s) selected: "
                + ", ".join(targets)
            )
            print(
                "Generating explanation..."
            )

    class ShowNextOptions(OneShotBehaviour):
        """Ask how the decision maker wants to continue."""

        async def run(self):
            while True:
                print()
                print("Choose how to continue:")
                print(
                    "0 = provide a new reference point"
                )
                print(
                    "1 = choose different target objectives"
                )
                print(
                    "2 = show detailed explanation"
                )
                print(
                    "3 = finish"
                )

                choice = await self.agent.get_input(
                    "> "
                )

                choice = choice.strip()

                if choice == "0":
                    self.agent.reference_point = None
                    self.agent.solution = None
                    self.agent.targets = None
                    self.agent.latest_explanation = None

                    self.agent.add_behaviour(
                        self.agent.AskReferencePoint()
                    )
                    return

                if choice == "1":
                    self.agent.add_behaviour(
                        self.agent.AskTargets()
                    )
                    return

                if choice == "2":
                    if (
                        self.agent.latest_explanation
                        is None
                    ):
                        print(
                            "\nNo detailed explanation "
                            "is currently available."
                        )
                    else:
                        self.agent.display_detailed_explanation(
                            self.agent.latest_explanation
                        )

                    # Remain inside the same loop so that the DM
                    # returns to the continuation menu afterwards.
                    continue

                if choice == "3":
                    print(
                        "\nDecision-making session finished."
                    )
                    return

                print("Invalid option.")

    class ReceiveMessages(CyclicBehaviour):
        """Receive RPM solutions and Owen explanations."""

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
                    "Preference agent received "
                    "invalid JSON."
                )
                return

            message_type = contents.get(
                "type"
            )

            if message_type == "solution":
                self.agent.waiting_for_solution = (
                    False
                )

                solution = contents[
                    "solution_original"
                ]

                self.agent.solution = solution

                print()
                print(
                    "RPM solution objective values"
                )
                print("-" * 40)

                for symbol, value in zip(
                    self.agent.objective_symbols,
                    solution,
                    strict=True,
                ):
                    print(
                        f"{symbol} = "
                        f"{float(value):.6f}"
                    )

                print("-" * 40)

                # Once the solution is available, ask what the DM
                # wants explained.
                self.agent.add_behaviour(
                    self.agent.AskTargets()
                )

            elif (
                message_type
                == "owen_explanation"
            ):
                self.agent.waiting_for_explanation = (
                    False
                )

                # Keep the complete result available for optional
                # detailed inspection.
                self.agent.latest_explanation = contents

                # Show only the concise actionable suggestion by
                # default.
                self.agent.display_suggestion(
                    contents
                )

                self.agent.add_behaviour(
                    self.agent.ShowNextOptions()
                )

            else:
                print(
                    "Preference agent received an "
                    "unknown message type "
                    f"'{message_type}'."
                )

    def display_suggestion(
        self,
        contents: dict,
    ) -> None:
        """Display the concise Owen/R-XIMO recommendation."""
        print()
        print(
            "SUGGESTION"
        )
        print("=" * 60)

        targets = contents.get(
            "targets",
            [],
        )

        if targets:
            print(
                "Target objective(s): "
                + ", ".join(targets)
            )
            print()

        suggestion = contents.get(
            "suggestion"
        )

        if suggestion:
            print(suggestion)
        else:
            print(
                "No suggestion was generated."
            )

        print()

    def display_detailed_explanation(
        self,
        contents: dict,
    ) -> None:
        """Display detailed Owen and R-XIMO explanation information."""
        print()
        print(
            "DETAILED OWEN EXPLANATION"
        )
        print("=" * 60)

        targets = contents.get(
            "targets",
            [],
        )

        if targets:
            print(
                "Target objective(s): "
                + ", ".join(targets)
            )

        case_number = contents.get(
            "case_number"
        )
        case_name = contents.get(
            "case_name"
        )

        if case_number is not None:
            if case_name:
                print(
                    f"R-XIMO case: "
                    f"{case_number} "
                    f"({case_name})"
                )
            else:
                print(
                    f"R-XIMO case: "
                    f"{case_number}"
                )

        status = contents.get(
            "reference_point_status"
        )

        if status is not None:
            print(
                "Reference point status: "
                f"{status}"
            )

        explanation = contents.get(
            "explanation"
        )

        if explanation:
            print()
            print("Explanation")
            print("-" * 60)
            print(explanation)

        owen_values = contents.get(
            "owen_values"
        )

        if owen_values:
            print()
            print("Owen values")
            print("-" * 60)

            for row in owen_values:
                symbol = (
                    row.get("input")
                    or row.get("symbol")
                    or row.get("feature")
                    or row.get("input_symbol")
                )

                value = (
                    row.get("owen_value")
                    if "owen_value" in row
                    else row.get("value")
                )

                if (
                    symbol is not None
                    and value is not None
                ):
                    print(
                        f"{symbol}: "
                        f"{float(value):+.6f}"
                    )
                else:
                    print(row)

        print()

    async def setup(self):
        """Initialize preference-agent behaviours."""
        print("Preference agent started.")

        self.add_behaviour(
            self.ReceiveMessages(),
            Template(
                metadata={
                    "performative": "inform"
                }
            ),
        )

        self.add_behaviour(
            self.InitialInteraction()
        )
