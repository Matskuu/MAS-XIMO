"""Decision-maker preference agent for MAS-XIMO."""

from __future__ import annotations

import asyncio
import json

from explanations.utils import get_objective_symbols
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template


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
                    "\nSelect one or more objectives "
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
                "Generating explanation and validating the suggestion with RPM..."
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

            elif message_type == "owen_explanation":
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

    def format_symbol_list(
        self,
        symbols: list[str],
    ) -> str:
        """Format objective symbols for DM-facing text."""
        if len(symbols) == 1:
            return symbols[0]

        if len(symbols) == 2:
            return (
                f"{symbols[0]} and "
                f"{symbols[1]}"
            )

        return (
            ", ".join(symbols[:-1])
            + f", and {symbols[-1]}"
        )

    def display_suggestion(
        self,
        contents: dict,
    ) -> None:
        """Display the concise Owen/R-XIMO recommendation and validation."""
        print()
        print("SUGGESTION")
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

        counterfactual = contents.get(
            "counterfactual"
        )

        if counterfactual:
            print()
            print("Counterfactual check")
            print("-" * 60)

            outcome = counterfactual.get(
                "outcome_type"
            )

            improved = counterfactual.get(
                "improved_targets",
                [],
            )

            worsened = counterfactual.get(
                "worsened_targets",
                [],
            )

            unchanged = counterfactual.get(
                "unchanged_targets",
                [],
            )

            if outcome == "joint_improvement":
                if len(targets) == 1:
                    print(
                        f"The suggested preference change improved "
                        f"{targets[0]} in the RPM check."
                    )
                else:
                    print(
                        "The suggested preference change improved "
                        "all selected target objectives together "
                        "in the RPM check."
                    )

            elif outcome == "partial_improvement":
                print(
                    "The suggested preference change improved "
                    "some selected targets without worsening "
                    "the others."
                )

                if improved:
                    print(
                        "Improved: "
                        + ", ".join(improved)
                    )

                if unchanged:
                    print(
                        "Unchanged: "
                        + ", ".join(unchanged)
                    )

            elif outcome == "target_conflict":
                print(
                    "The suggested preference change did not "
                    "improve all selected targets together."
                )

                if improved:
                    print(
                        "Improved: "
                        + ", ".join(improved)
                    )

                if worsened:
                    print(
                        "Worsened: "
                        + ", ".join(worsened)
                    )

            elif outcome == "no_improvement":
                print(
                    "The suggested preference change did not "
                    "improve the selected target objectives."
                )

                if worsened:
                    print(
                        "Worsened: "
                        + ", ".join(worsened)
                    )

                if unchanged:
                    print(
                        "Unchanged: "
                        + ", ".join(unchanged)
                    )

            elif outcome == "negligible_change":
                changes = counterfactual.get(
                    "changes",
                    [],
                )

                if len(changes) == 1:
                    symbol = changes[0].get(
                        "symbol",
                        "the selected target",
                    )

                    print(
                        "The tested preference change produced "
                        f"no meaningful change in {symbol}."
                    )
                else:
                    print(
                        "The tested preference change produced "
                        "no meaningful change in the selected targets."
                    )

        opportunity_result = contents.get(
            "opportunities"
        )

        if opportunity_result:
            opportunities = (
                opportunity_result.get(
                    "opportunities",
                    [],
                )
            )

            if opportunities:
                symbols = [
                    opportunity["symbol"]
                    for opportunity in opportunities
                ]

                target_text = (
                    "selected target"
                    if len(targets) == 1
                    else "selected targets"
                )

                print()
                print("Additional opportunity")
                print("-" * 60)
                print(
                    f"In addition to the {target_text}, "
                    f"{self.format_symbol_list(symbols)} also improved "
                    "under the tested preference change."
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

        actionable_targets = contents.get(
            "actionable_target_members",
            [],
        )
        non_actionable_targets = contents.get(
            "non_actionable_target_members",
            [],
        )
        actionable_rivals = contents.get(
            "actionable_rival_members",
            [],
        )
        rival_members = contents.get(
            "rival_members",
            [],
        )

        def display_members(
            members: list[str],
        ) -> str:
            """Format machine-facing symbols for DM-facing output."""
            cleaned = [
                member
                .removeprefix("r_")
                .removeprefix("s_")
                for member in members
            ]

            if not cleaned:
                return "none"

            if len(cleaned) == 1:
                return cleaned[0]

            return (
                "{"
                + ", ".join(cleaned)
                + "}"
            )

        if (
            actionable_targets
            or non_actionable_targets
            or rival_members
        ):
            print()
            print("Aspiration-level action")
            print("-" * 60)

            if non_actionable_targets:
                target_text = display_members(
                    non_actionable_targets
                )

                if len(non_actionable_targets) == 1:
                    print(
                        f"The aspiration level for {target_text} is "
                        "already at or beyond its ideal value, so no "
                        "further improvement of this aspiration is suggested."
                    )
                else:
                    print(
                        f"The aspiration levels for {target_text} are "
                        "already at or beyond their ideal values, so no "
                        "further improvement of these aspirations is suggested."
                    )

            if actionable_targets:
                target_text = display_members(
                    actionable_targets
                )

                if len(actionable_targets) == 1:
                    print(
                        f"The aspiration level for {target_text} "
                        "can still be improved."
                    )
                else:
                    print(
                        f"The aspiration levels for {target_text} "
                        "can still be improved."
                    )

            if rival_members and not actionable_rivals:
                rival_text = display_members(
                    rival_members
                )

                if len(rival_members) == 1:
                    print(
                        f"The relevant external aspiration level for "
                        f"{rival_text} is already at its nadir value, "
                        "so it cannot be impaired further."
                    )
                else:
                    print(
                        f"The relevant external aspiration levels for "
                        f"{rival_text} are already at their nadir values, "
                        "so they cannot be impaired further."
                    )

            elif actionable_rivals:
                rival_text = display_members(
                    actionable_rivals
                )

                if len(actionable_rivals) == 1:
                    print(
                        f"The aspiration level for {rival_text} "
                        "can be impaired."
                    )
                else:
                    print(
                        f"The aspiration levels for {rival_text} "
                        "can be impaired."
                    )

        
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

        counterfactual = contents.get(
            "counterfactual"
        )

        if counterfactual:
            print()
            print("Counterfactual validation")
            print("-" * 60)

            outcome = counterfactual.get(
                "outcome_type"
            )

            if outcome is not None:
                print(
                    f"Outcome: {outcome}"
                )

            changes = counterfactual.get(
                "changes",
                [],
            )

            if changes:
                print()
                print("Target comparison:")

                for change in changes:
                    print(
                        f"{change['symbol']}: "
                        f"{float(change['original_value']):.6f} "
                        "-> "
                        f"{float(change['adjusted_value']):.6f} "
                        f"({change['status']})"
                    )

            adjusted_reference_point = (
                counterfactual.get(
                    "adjusted_reference_point"
                )
            )

            if adjusted_reference_point is not None:
                print()
                print(
                    "Counterfactual reference point: "
                    f"{adjusted_reference_point}"
                )

        opportunity_result = contents.get(
            "opportunities"
        )

        if opportunity_result:
            opportunities = (
                opportunity_result.get(
                    "opportunities",
                    [],
                )
            )

            print()
            print("Additional objective opportunities")
            print("-" * 60)

            if opportunities:
                for opportunity in opportunities:
                    print(
                        f"{opportunity['symbol']}: "
                        f"{float(opportunity['original_value']):.6f} "
                        "-> "
                        f"{float(opportunity['adjusted_value']):.6f} "
                        "(improved)"
                    )
            else:
                print(
                    "No additional compatible non-target "
                    "improvements were identified."
                )

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
