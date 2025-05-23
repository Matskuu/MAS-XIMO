import asyncio
import json
import keyboard
import random
import re
import spade
import string

import numpy as np
import polars as pl

from pathlib import Path
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from desdeo.explanations import ShapExplainer, generate_biased_mean_data
from desdeo.mcdm import rpm_solve_solutions
from desdeo.problem import Problem
from desdeo.problem.testproblems import pareto_navigator_test_problem
from desdeo.utopia_stuff.utopia_problem_old import utopia_problem_old


shap_model = None

PROBLEM_ENUM = {
    "utopia_problem_old": utopia_problem_old()[0],
    "pareto_navigator_test_problem": pareto_navigator_test_problem()
}

class Solver(Agent):
    def __init__(self, jid, password, solver, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.problem = None
        self.reference_point = None
        self.solution = None
        self.solver = solver

    class SendSolution(OneShotBehaviour):
        async def run(self):
            print("Solver sending a solution...")
            contents = {
                "solution": self.agent.solution,
                "reference_point": self.agent.reference_point
            }
            msg = Message(
                to="explanationgatherer@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print(f"Solution and reference point sent to explanation gatherer: {contents}")
            
            contents = {"solution": self.agent.solution}
            msg = Message(
                to="preferenceagent@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print(f"Solution sent to preference agent: {self.agent.solution}")

    class Solve(OneShotBehaviour):
        async def run(self):
            print("Solver solving the problem...")
            self.agent.solution = self.agent.solver(self.agent.problem, self.agent.reference_point)[0].optimal_objectives # rpm returns a list of two things
            self.agent.add_behaviour(self.agent.SendSolution())

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print("Solver waiting for messages.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "problem" in contents:
                    problem = PROBLEM_ENUM[contents["problem"]]
                    print(f"Solver received the problem: {contents["problem"]}.")
                    self.agent.problem = problem
                    if self.agent.reference_point: # assuming we are solving the same problem for which the reference point is given
                        self.agent.add_behaviour(self.agent.Solve())
                    # TODO: maybe request for a reference point as well once the problem is received?
                if "reference_point" in contents:
                    print(f"Solver received reference point: {contents["reference_point"]}.")
                    self.agent.reference_point = contents["reference_point"]
                    if self.agent.problem:
                        self.agent.add_behaviour(self.agent.Solve())
                    else:
                        self.agent.add_behaviour(self.agent.SendRequests(request_content="problem"))
    
    async def setup(self):
        print("Solver started.")
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))

class PreferenceAgent(Agent):
    def __init__(self, jid, password, problem_name, max_iterations, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.can_send_target = False
        self.received_solution = False
        self.problem = PROBLEM_ENUM[problem_name]
        self.problem_name = problem_name
        self.objective_symbols = [obj.symbol for obj in self.problem.objectives]
        self.problem_ideal = self.problem.get_ideal_point()
        self.problem_nadir = self.problem.get_nadir_point()
        self.reference_point = {}
        self.solution = None
        self.iteration = 0
        self.max_iterations = max_iterations
        self.explanation = None
        self.examples = None

    class ReceiveInformMessages(CyclicBehaviour):
        def get_updated_reference_point(self, explanation: str, previous_reference_point):
            previous = previous_reference_point
            if previous and self.agent.solution:
                explanation_split = re.split(r"[ ,]+", explanation)
                objective_to_improve = explanation_split[explanation_split.index("improve") + 2]
                objective_to_impair = explanation_split[explanation_split.index("impair") + 2]
                #amount_to_impair = float(re.sub(r"\D+$", "", explanation_split[explanation_split.index("by") + 1]))
                #print(objective_to_improve, objective_to_impair, amount_to_impair)
                # because the problem is maximization, when this is not known it should be checked
                new_reference_point = self.agent.solution.copy()
                amount_to_improve = (self.agent.problem.get_ideal_point()[objective_to_improve] - self.agent.solution[objective_to_improve]) / 2
                if objective_to_impair != "None":
                    amount_to_impair = (self.agent.solution[objective_to_impair] - self.agent.problem.get_nadir_point()[objective_to_impair]) / 2
                    new_reference_point[objective_to_impair] = new_reference_point[objective_to_impair] - amount_to_impair
                new_reference_point[objective_to_improve] = new_reference_point[objective_to_improve] + amount_to_improve
                #print(f"New reference point: {new_reference_point}")
                return new_reference_point

        async def run(self):
            #print("Preference agent waiting for messages.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                # this could have been named more appropriately
                # basically this is meant to be done the first time when sending a reference point and then a target
                # this is done like this (at least it was initially) to make sure everyone is ready for this data
                if "ready for target" in contents:
                    self.agent.can_send_target = True
                    #self.agent.add_behaviour(self.agent.SendData("problem"))
                    # initially reference point is not the ideal so that the reference point components stay in between ideal and nadir
                    self.agent.add_behaviour(self.agent.SendData(content_type="reference point"))
                    if self.agent.received_solution:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                        self.agent.received_solution = False
                elif "solution" in contents:
                    self.agent.received_solution = True
                    self.agent.solution = contents["solution"]
                    if self.agent.iteration >= self.agent.max_iterations:
                        print("Preference agent has reached its preset max number of iterations. Ending the solution process.")
                        print(f"{self.agent.solution} was chosen as the final solution.")
                        return
                    if self.agent.can_send_target:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                        self.agent.received_solution = False
                elif "explanation" and "examples" in contents:
                    print("Preference agent received an explanation.")
                    self.agent.explanation = contents["explanation"]
                    self.agent.examples = contents["examples"]
                    print(self.agent.explanation)
                    print("-----------------------------------------")
                    for pair in contents["examples"]:
                        print(f"Reference point: {pair[0]}")
                        print(f"Solution: {pair[1]}")
                        print("-----------------------------------------")
                    self.agent.add_behaviour(self.agent.SendData("reference point"))

    class ReceiveRequests(CyclicBehaviour):
        async def run(self):
            #print("Preference agent waiting for requests.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "target":
                    self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                if msg.body == "problem":
                    self.agent.add_behaviour(self.agent.SendData(content_type="problem"))

    class SendData(OneShotBehaviour):
        def __init__(self, content_type: str):
            super().__init__()
            self.content_type = content_type

        async def run(self):
            if self.content_type == "target":
                target = input("Provide an objective to improve: ")
                if target not in self.agent.objective_symbols:
                    # TODO: should do some iterating here if invalid target
                    print("Target not valid!")
                    target = "f_1"
                contents = {"target": target}
                msg = Message(
                    to="explanationgatherer@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                print("Preference agent sending the target...")
                await self.send(msg)
                print(f"Target sent: {target}.")
            elif self.content_type == "problem":
                print("Preference agent sending the problem...")
                #problem_name = input("Provide the problem name: ")
                recipients = ["solver@localhost", "explanationgatherer@localhost"]
                for recipient in recipients:
                    contents = {"problem": self.agent.problem_name}
                    msg = Message(
                        to=recipient,
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
                print(f"The problem sent: {self.agent.problem.name}.")
            elif self.content_type == "reference point":
                for symbol in self.agent.objective_symbols:
                    # TODO: check the reference point component values and if not between ideal and nadir, ask for a component value
                    value = input(f"Provide a value for objective {symbol}: ")
                    if value == "ideal":
                        self.agent.reference_point[symbol] = self.agent.problem_ideal[symbol]
                    elif value == "nadir":
                        self.agent.reference_point[symbol] = self.agent.problem_nadir[symbol]
                    elif not value.isnumeric():
                        # TODO: should do something more fancy here
                        self.agent.reference_point[symbol] = self.agent.problem_ideal[symbol]
                    else:
                        self.agent.reference_point[symbol] = float(value)
                contents = {"reference_point": self.agent.reference_point}
                msg = Message(
                    to="solver@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                print("Preference agent sending a reference point...")
                await self.send(msg)
                print(f"A reference point sent: {self.agent.reference_point}.")
                self.agent.iteration = self.agent.iteration + 1
    
    async def setup(self):
        print("Preference agent started.")
        receive_inform_messages = self.ReceiveInformMessages()
        self.add_behaviour(receive_inform_messages, Template(metadata={"performative": "inform"}))
        receive_requests = self.ReceiveRequests()
        self.add_behaviour(receive_requests, Template(metadata={"performative": "request"}))

class SHAPAgent(Agent):
    def __init__(self, jid, password, df: pl.DataFrame, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.reference_point = None
        self.solution = None
        self.df = df
        self.shaps = None
        self.problem = None
        self.objective_symbols = None

    class SendInformMessages(OneShotBehaviour):
        def __init__(self, content_type: str):
            super().__init__()
            self.content_type = content_type

        async def run(self):
            if self.content_type == "shaps":
                msg = Message(
                    to="explanationgatherer@localhost",
                    body=json.dumps({"shaps": self.agent.shaps.tolist()}),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)

                # TODO: alternatively send the shaps straight to the explainers instead
                """contents = {
                    "reference_point": self.agent.reference_point,
                    "solution": self.agent.solution,
                    "shaps": self.agent.shaps.tolist()
                }
                for symbol in self.agent.objective_symbols:
                    clean_symbol = symbol.translate(str.maketrans('', '', string.punctuation))
                    msg = Message(
                        to=f"explainer{clean_symbol}@localhost",
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)"""

    class GenerateSHAPs(OneShotBehaviour):
        async def run(self):
            target = [value for _, value in self.agent.reference_point.items()]
            background_subset = generate_biased_mean_data(self.agent.df[["f_1", "f_2", "f_3"]].to_numpy(), target, solver="GUROBI")
            shap_model.setup(background_data=pl.DataFrame(self.agent.df[background_subset]))
            self.agent.shaps = shap_model.explain_input(pl.DataFrame({"z_1": target[0], "z_2": target[1], "z_3": target[2]})).values[0].T

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "reference_point" and "solution" in contents:
                    print(f"SHAP agent received a solution and reference point: {contents}")
                    self.agent.reference_point = contents["reference_point"]
                    self.agent.solution = contents["solution"]
                    generate_shaps = self.agent.GenerateSHAPs()
                    self.agent.add_behaviour(generate_shaps)
                    await generate_shaps.join()
                    self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps"))
                elif "problem" in contents:
                    problem = PROBLEM_ENUM[contents["problem"]]
                    print(f"Explanation gatherer received the problem: {problem.name}")
                    self.agent.problem = problem
                    self.agent.n_objectives = len(problem.objectives)
                    self.agent.objective_symbols = [obj.symbol for obj in problem.objectives]

    async def setup(self):
        receive_inform_messages = self.ReceiveInformMessages()
        self.add_behaviour(receive_inform_messages, Template(metadata={"performative": "request"}))

class ExplanationGatherer(Agent):
    def __init__(self, jid, password, df: pl.DataFrame, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.n_objectives = None
        self.problem = None
        self.objective_symbols = []
        self.target = None
        self.df = df
        self.reference_point = None
        self.solution = None
        self.shaps = None
        self.explanation = None

    class GenerateShaps(OneShotBehaviour):
        async def run(self):
            target = [value for _, value in self.agent.reference_point.items()]
            background_subset = generate_biased_mean_data(self.agent.df[["f_1", "f_2", "f_3"]].to_numpy(), target, solver="GUROBI")
            shap_model.setup(background_data=pl.DataFrame(self.agent.df[background_subset]))
            self.agent.shaps = shap_model.explain_input(pl.DataFrame({"z_1": target[0], "z_2": target[1], "z_3": target[2]})).values[0].T

    class SendInformMessages(OneShotBehaviour):
        def __init__(self, content_type: str):
            super().__init__()
            self.content_type = content_type

        async def run(self):
            if self.content_type == "shaps":
                contents = {
                    "reference_point": self.agent.reference_point,
                    "solution": self.agent.solution,
                    "shaps": self.agent.shaps.tolist()
                }
                for symbol in self.agent.objective_symbols:
                    clean_symbol = symbol.translate(str.maketrans('', '', string.punctuation))
                    msg = Message(
                        to=f"explainer{clean_symbol}@localhost",
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
            elif self.content_type == "explanation":
                contents = {
                    "explanation": self.agent.explanation,
                    "examples": self.agent.examples
                }
                msg = Message(
                    to="preferenceagent@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)

    class SendRequests(OneShotBehaviour):
        def __init__(self, request_content: str):
            super().__init__()
            self.request_content = request_content

        async def run(self):
            print(f"Explanation gatherer sending a request for {self.request_content}...")
            if self.request_content == "problem":
                msg = Message(
                    to = "preferenceagent@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                print("Explanation gatherer requested the problem.")
            elif self.request_content == "examples":
                clean_symbol = self.agent.target.translate(str.maketrans('', '', string.punctuation))
                msg = Message(
                    to = f"explainer{clean_symbol}@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                print("Explanation gatherer requested examples from the target explainer.")
    
    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print("Explanation gatherer ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "problem" in contents:
                    problem = PROBLEM_ENUM[contents["problem"]]
                    print(f"Explanation gatherer received the problem: {problem.name}")
                    self.agent.problem = problem
                    self.agent.n_objectives = len(problem.objectives)
                    self.agent.objective_symbols = [obj.symbol for obj in problem.objectives]
                    self.agent.add_behaviour(self.agent.CreateExplainers())
                elif "target" in contents:
                    print(f"Explanation gatherer received the target: {contents["target"]}")
                    #self.agent.target = contents["target"].translate(str.maketrans('', '', string.punctuation))
                    self.agent.target = contents["target"]
                    """generate_explanations = self.agent.GenerateExplanations()
                    self.agent.add_behaviour(generate_explanations)
                    await generate_explanations.join()"""
                    #self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps"))
                    self.agent.add_behaviour(self.agent.SendRequests(request_content="examples"))
                    #self.agent.add_behaviour(self.agent.SendRequests("explanation"))
                elif "reference_point" and "solution" in contents:
                    print(f"Explanation gatherer received a solution and reference point: {contents}")
                    self.agent.reference_point = contents["reference_point"]
                    self.agent.solution = contents["solution"]
                    generate_shaps = self.agent.GenerateShaps()
                    self.agent.add_behaviour(generate_shaps)
                    await generate_shaps.join()
                    self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps"))
                elif "examples" and "explanation" in contents:
                    #print(f"Explanation gatherer received examples: \n  {contents["examples"]}")
                    self.agent.examples = contents["examples"]
                    self.agent.explanation = contents["explanation"]
                    self.agent.add_behaviour(self.agent.SendInformMessages(content_type="explanation"))
                elif "shaps" in contents:
                    self.agent.shaps = contents["shaps"]
                    self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps"))


    class CreateExplainers(OneShotBehaviour):
        async def run(self):
            print(f"Explanation gatherer creating {self.agent.n_objectives} explainers...")
            for i in range(self.agent.n_objectives):
                objective = self.agent.objective_symbols[i]
                clean_symbol = objective.translate(str.maketrans('', '', string.punctuation))
                explainer = Explainer(f"explainer{clean_symbol}@localhost", "explainer", objective=objective, df=self.agent.df, problem=self.agent.problem)
                await explainer.start(auto_register=True)
            msg = Message(
                to="preferenceagent@localhost",
                body=json.dumps("ready for target"),
                metadata={"performative": "inform"}
            )
            await self.send(msg)

    async def setup(self):
        print("Explanation gatherer started.")
        self.add_behaviour(self.SendRequests(request_content="problem"))
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))


class Explainer(Agent):
    def __init__(self, jid, password, objective: str, df: pl.DataFrame, problem, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.objective = objective
        self.df = df
        self.problem = problem
        self.objective_symbols = [obj.symbol for obj in problem.objectives]
        self.can_send_examples = False
        self.has_new_data = False
        self.ready_to_send_examples = False
        self.solution = None
        self.reference_point = None
        self.examples = None
        self.explanation = None

    class ReceiveRequests(CyclicBehaviour):
        async def run(self):
            #print(f"{self.agent.jid.username} ready to receive requests.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "examples":
                    self.agent.can_send_examples = True
                    if self.agent.ready_to_send_examples:
                        self.agent.add_behaviour(self.agent.SendExamples())
                        self.agent.can_send_examples = False

    class GenerateExamples(OneShotBehaviour):
        def __init__(self, number_of_examples: int):
            super().__init__()
            self.number_of_examples = number_of_examples

        async def run(self):
            examples = []
            shaps = self.agent.shaps
            # TODO: make these problem specific
            shaps_dict = {}
            if len(shaps) == len(self.agent.objective_symbols):
                shaps_dict = {
                    self.agent.objective_symbols[i]: shaps[i]
                    for i in range(len(shaps))
                }
            shaps_for_this_agent = shaps_dict[self.agent.objective]
            if len(self.agent.objective_symbols) == len(shaps_for_this_agent):
                shaps_for_this_agent_dict = {
                    self.agent.objective_symbols[i]: shaps_for_this_agent[i]
                    for i in range(len(shaps_for_this_agent))
                }
            # initialize these with no default value
            to_improve = None
            to_impair = None
            max_effect = -float("inf")
            min_effect = float("inf")
            for key, value in shaps_for_this_agent_dict.items():
                if value > max_effect and key != self.agent.objective:
                #if value > max_effect:
                    max_effect = value
                    #to_improve = key
                    to_impair = key
                #if value < min_effect and key != self.agent.objective:
                if value < min_effect:
                    min_effect = value
                    #to_impair = key
                    to_improve = key
            # here trying to not worsen the value of the objective by improving a component in the reference point that has a negative effect on the target
            # though it is possible that the target has a negative effect on itself as well
            if min_effect >= 0:
                to_improve = self.agent.objective
            if max_effect < 0:
                to_impair = None

            explanation = f"To get better value for objective {self.agent.objective}, try to improve objective {to_improve} and impair objective {to_impair} in the reference point."
            self.agent.explanation = explanation

            # assuming that some component gets improved every time for now
            # TODO: should the basis be the original reference point from the DM or the solution?
            for i in range(self.number_of_examples):
                new_reference_point = self.agent.solution.copy()
                amount_to_improve = (self.agent.problem.get_ideal_point()[to_improve] - self.agent.solution[to_improve]) / (10/(i+1))
                new_reference_point[to_improve] = new_reference_point[to_improve] + amount_to_improve
                if to_impair:
                    amount_to_impair = (self.agent.solution[to_impair] - self.agent.problem.get_nadir_point()[to_impair]) / (10/(i+1))
                    new_reference_point[to_impair] = new_reference_point[to_impair] - amount_to_impair
                solution = rpm_solve_solutions(self.agent.problem, new_reference_point)[0].optimal_objectives
                examples.append((new_reference_point, solution))
            self.agent.examples = examples
            self.agent.ready_to_send_examples = True
            print(f"{self.agent.jid.username} ready to send examples.")
            #print(examples)
    
    class SendExamples(OneShotBehaviour):
        async def run(self):
            print(f"{self.agent.jid.username} sending examples...")
            contents = {
                "explanation": self.agent.explanation,
                "examples": self.agent.examples
            }
            # TODO: should this be sent straight to the preference agent instead?
            msg = Message(
                to="explanationgatherer@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print("Examples sent.")
            self.agent.ready_to_send_examples = False
            self.agent.has_new_data = False

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print(f"{self.agent.jid.username} ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "shaps" and "solution" and "reference_point" in contents:
                    #print(f"{self.agent.jid.username} received data: {contents}")
                    self.agent.solution = contents["solution"]
                    self.agent.reference_point = contents["reference_point"]
                    self.agent.shaps = contents["shaps"]
                    self.agent.has_new_data = True
                    self.agent.add_behaviour(self.agent.GenerateExamples(number_of_examples=5))
                    if self.agent.can_send_examples:
                        self.agent.add_behaviour(self.agent.SendExamples())
                        self.agent.can_send_examples = False

    async def setup(self):
        print(f"{self.jid.username} started.")
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))
        receiveRequests = self.ReceiveRequests()
        self.add_behaviour(receiveRequests, Template(metadata={"performative": "request"}))

async def main(df: pl.DataFrame):
    # initialize and start the solver
    solverAgent = Solver("solver@localhost", "solver", solver=rpm_solve_solutions)
    await solverAgent.start(auto_register=True)

    # initialize and start a preference agent
    preferenceAgent = PreferenceAgent("preferenceagent@localhost", "preferenceagent", problem_name="utopia_problem_old", max_iterations=5)
    await preferenceAgent.start(auto_register=True)

    # initialize and start an explanation gatherer
    explanationGatherer = ExplanationGatherer("explanationgatherer@localhost", "preference", df=df)
    await explanationGatherer.start(auto_register=True)
    #explanationGatherer.web.start(hostname="127.0.0.1", port="10000")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping agents...")
        await solverAgent.stop()
        await explanationGatherer.stop()
        await preferenceAgent.stop()

def sample_input_space_to_file(n_samples: int, problem_name: str = None, file_name: str = None):
    problem = PROBLEM_ENUM[problem_name] if problem_name else PROBLEM_ENUM["utopia_problem_old"]
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    n_samples = n_samples

    obj_symbols = [obj.symbol for obj in problem.objectives]

    bounds = {
        symbol: (nadir[symbol], ideal[symbol])
        for symbol in obj_symbols
    }

    inputs = {
        key: np.random.uniform(low, high, size=n_samples)
        for key, (low, high) in bounds.items()
    }

    input_dicts = [
        {key: inputs[key][i] for key in bounds}
        for i in range(n_samples)
    ]

    outputs = []

    for i in input_dicts:
        d = rpm_solve_solutions(problem, i)[0].optimal_objectives
        for key, value in i.items():
            d[key.replace("f", "z")] = value
        outputs.append(d)

    if file_name:
        with Path.open(f"{file_name}.json", "w") as file:
            json.dump(outputs, file)
    else:
        with Path.open("sample.json", "w") as file:
            json.dump(outputs, file)

if __name__ == "__main__":
    # run the multi-agent system with three objectives (i.e., three explainers)

    # this seems like something the explanation gatherer should do as soon as it gets the problem
    #sample_input_space_to_file(n_samples=20, problem_name="utopia_problem_old")
    
    outputs = []

    with Path.open("sample.json", "r") as file:
        outputs = json.load(file)

    df = pl.DataFrame(outputs)

    shap_model = ShapExplainer(problem_data=df, input_symbols=["z_1", "z_2", "z_3"], output_symbols=["f_1", "f_2", "f_3"])

    spade.run(main(df=df))