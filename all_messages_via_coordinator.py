"""A multi-agent system version in which the explainers send the example reference points to the coordinator
who then sends them to the solver and in return gets the solution that is sent to the correct explainer by the
coordinator."""

import asyncio
import json
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
from desdeo.utopia_stuff.utopia_problem_old import utopia_problem_old
from desdeo.problem.testproblems import dtlz2

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import time

file = open("mas_log.txt", mode="w")

executor = ThreadPoolExecutor(max_workers=1)

class Solver(Agent):
    def __init__(self, jid, password, solver, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.problem = None
        self.reference_point = None
        self.solution = None
        self.solver = solver

    class SendSolution(OneShotBehaviour):
        def __init__(self, reference_point, solution, sender: str = None):
            super().__init__()
            self.sender = sender
            self.reference_point = reference_point
            self.solution = solution

        async def run(self):
            #print("Solver sending a solution...")
            contents = {
                "solution": self.solution,
                "reference_point": self.reference_point
            }
            msg = Message(
                to="coordinator@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            if self.sender:
                msg.sender = self.sender
            await self.send(msg)
            file.write(f"Solver sent message {msg} at {time.time()}\n")
            #print(f"Solution and reference point sent to coordinator: {contents}")

    class Solve(OneShotBehaviour):
        def __init__(self, reference_point, sender: str = None):
            super().__init__()
            self.sender = sender
            self.reference_point = reference_point

        async def run(self):
            #print("Solver solving the problem...")
            #print(self.reference_point)
            await asyncio.sleep(0.01)
            file.write(f"Time before solved for {self.reference_point} is {time.time()}\n")
            #solution = self.agent.solver(self.agent.problem, self.reference_point)[0].optimal_objectives # rpm returns a list of two things
            loop = asyncio.get_running_loop()
            solver = self.agent.solver
            problem = self.agent.problem.model_copy(deep=True)
            reference_point = self.reference_point.copy()
            try:
                result = await loop.run_in_executor(
                    executor,
                    solver,
                    problem,
                    reference_point
                )
            except RuntimeError as e:
                print("Caught RuntimeError in executor:", e)
                return
            solution = result[0].optimal_objectives
            file.write(f"Time after solved for {self.reference_point} is {time.time()}\n")
            if self.sender:
                self.agent.add_behaviour(self.agent.SendSolution(sender=self.sender, reference_point=self.reference_point, solution=solution))
            else:
                self.agent.add_behaviour(self.agent.SendSolution(reference_point=self.reference_point, solution=solution))

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print("Solver waiting for messages.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "problem" in contents:
                    problem = Problem.model_validate_json(contents["problem"])
                    #print(f"Solver received the problem: {problem.name}.")
                    self.agent.problem = problem
                    if self.agent.reference_point: # assuming we are solving the same problem for which the reference point is given
                        self.agent.add_behaviour(self.agent.Solve())
                    # TODO: maybe request for a reference point as well once the problem is received?
                if "reference_point" in contents:
                    #print(f"Solver received reference point: {contents["reference_point"]}.")
                    #self.agent.reference_point = contents["reference_point"]
                    if self.agent.problem:
                        file.write(f"Solver received message {msg} at {time.time()}\n")
                        self.agent.add_behaviour(self.agent.Solve(sender=msg.sender.full, reference_point=contents["reference_point"]))
                    else:
                        self.agent.add_behaviour(self.agent.SendRequests(request_content="problem"))
    
    async def setup(self):
        print("Solver started.")
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))

class PreferenceAgent(Agent):
    def __init__(self, jid, password, problem: Problem, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.can_send_target = False
        self.received_solution = False
        self.problem = problem
        self.objective_symbols = [obj.symbol for obj in self.problem.objectives]
        self.problem_ideal = self.problem.get_ideal_point()
        self.problem_nadir = self.problem.get_nadir_point()
        self.reference_point = {}
        self.solution = None
        self.explanation = None
        self.examples = None
        self.target = None
        self.explanation_type = None

    async def get_input(self, prompt):
        print(prompt, end='', flush=True)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, input)

    class ShowOptions(OneShotBehaviour):
        async def run(self):
            while True:
                choice = await self.agent.get_input("Choose an option to proceed:\n0 = provide a new reference point\n1 = choose a different target\n2 = choose different explanation type\n3 = choose current solution as the final solution\n")
                if choice == "0":
                    self.agent.add_behaviour(self.agent.SendData("reference point"))
                    break
                elif choice == "1":
                    self.agent.add_behaviour(self.agent.SendData("target"))
                    break
                elif choice == "2":
                    while True:
                        explanation_type = await self.agent.get_input("What type of explanation would you like (0 = R-XIMO suggestion, 1 = R-XIMO suggestion + examples)? ")
                        if explanation_type == "0" or "1":
                            break
                        else:
                            print("Invalid input.")
                    self.agent.explanation_type = explanation_type
                    self.agent.add_behaviour(self.agent.ShowDifferentExplanations())
                    break
                else:
                    print("Option not valid.")

    class ShowDifferentExplanations(OneShotBehaviour):
        async def run(self):
            if self.agent.explanation_type == "0":
                print(self.agent.explanation)
                print(f"Original solution: {self.agent.solution}")
            elif self.agent.explanation_type == "1":
                print(self.agent.explanation)
                print(f"Original solution: {self.agent.solution}")
                print("-----------------------------------------")
                for reference_point, solution in self.agent.examples:
                    print(f"Reference point: {reference_point}")
                    print(f"Solution: {solution}")
                    for symbol in solution:
                        print(f"{self.agent.problem.get_objective(symbol).name}: {solution[symbol]}")
                    print("-----------------------------------------")
            # TODO: what if instead of defaulting to reference point, we give the option to choose a different target, different explanations etc.?
            self.agent.add_behaviour(self.agent.ShowOptions())

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
                amount_to_improve = (self.agent.problem_ideal[objective_to_improve] - self.agent.solution[objective_to_improve]) / 2
                if objective_to_impair != "None":
                    amount_to_impair = (self.agent.solution[objective_to_impair] - self.agent.problem_nadir[objective_to_impair]) / 2
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
                    self.agent.add_behaviour(self.agent.SendData(content_type="reference point"))
                    if self.agent.received_solution:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                        self.agent.received_solution = False
                elif "solution" in contents:
                    print(f"Solution found based on the reference point: {contents["solution"]}")
                    file.write(f"Preference agent received solution {contents["solution"]} at {time.time()}\n")
                    self.agent.received_solution = True
                    self.agent.solution = contents["solution"]
                    if self.agent.can_send_target:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                        self.agent.received_solution = False
                elif all(key in contents for key in ("explanation", "examples")):
                    print("Preference agent received an explanation.")
                    self.agent.explanation = contents["explanation"]
                    examples = contents["examples"]
                    examples_ordered = sorted(examples, key=lambda d: d[1][self.agent.target])
                    self.agent.examples = examples_ordered
                    if self.agent.explanation_type == "0":
                        print(self.agent.explanation)
                        print(f"Original solution: {self.agent.solution}")
                    elif self.agent.explanation_type == "1":
                        print(self.agent.explanation)
                        print(f"Original solution: {self.agent.solution}")
                        print("-----------------------------------------")
                        for reference_point, solution in self.agent.examples:
                            print(f"Reference point: {reference_point}")
                            print(f"Solution: {solution}")
                            for symbol in solution:
                                print(f"{self.agent.problem.get_objective(symbol).name}: {solution[symbol]}")
                            print("-----------------------------------------")
                    # TODO: what if instead of defaulting to reference point, we give the option to choose a different target, different explanations etc.?
                    self.agent.add_behaviour(self.agent.ShowOptions())
                    #self.agent.add_behaviour(self.agent.SendData("reference point"))

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

        async def get_input(self, prompt):
            print(prompt, end='', flush=True)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, input)

        async def run(self):
            if self.content_type == "target":
                while True:
                    await asyncio.sleep(0.01)
                    file.write(f"WAITING FOR A TARGET at {time.time()}\n")
                    #target = input(f"Provide an objective to improve among {self.agent.objective_symbols}: ")
                    # This is not blocking but also the prompt goes away because of the other prints
                    # In other words, get rid of the console prints and this solution works fine from the console
                    target = await self.get_input(f"Provide an objective to improve among {self.agent.objective_symbols}: ")
                    if target in self.agent.objective_symbols:
                        break
                    else:
                        print(f"Target not valid. Please enter one of the objectives: {self.agent.objective_symbols}.")
                self.agent.target = target
                contents = {"target": target}
                file.write(f"TARGET RECEIVED at {time.time()}\n")
                while True:
                    await asyncio.sleep(0.01)
                    file.write(f"WAITING FOR EXPLANATION TYPE at {time.time()}\n")
                    explanation_type = await self.get_input("What type of explanation would you like (0 = R-XIMO suggestion, 1 = R-XIMO suggestion + examples)? ")
                    if explanation_type == "0" or "1":
                        break
                    else:
                        print("Invalid input.")
                self.agent.explanation_type = explanation_type
                file.write(f"EXPLANATION TYPE RECEIVED at {time.time()}\n")
                #contents["explanation type"] = explanation_type
                msg = Message(
                    to="coordinator@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                #print("Preference agent sending the target...")
                await self.send(msg)
                file.write(f"TARGET SENT AT {time.time()}")
                #print(f"Target sent: {target} at {time.time()}.")
            elif self.content_type == "problem":
                #print("Preference agent sending the problem...")
                #problem_name = input("Provide the problem name: ")
                problem_json = self.agent.problem.model_dump_json(indent=4)
                contents = {"problem": problem_json}
                msg = Message(
                    to="coordinator@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                #print(f"The problem sent: {self.agent.problem.name}.")
            elif self.content_type == "reference point":
                for symbol in self.agent.objective_symbols:
                    objective = self.agent.problem.get_objective(symbol)
                    maximize = objective.maximize
                    if maximize:
                        range = [self.agent.problem_nadir[symbol], self.agent.problem_ideal[symbol]]
                    else:
                        range = [self.agent.problem_ideal[symbol], self.agent.problem_nadir[symbol]]
                    while True:
                        try:
                            #value = input(f"Provide a value for objective {symbol} ({objective.name}) within range {range}: ")
                            value = await self.get_input(f"Provide a value for objective {symbol} ({objective.name}) within range {range}: ")
                            if value == "ideal":
                                self.agent.reference_point[symbol] = self.agent.problem_ideal[symbol]
                                break
                            elif value == "nadir":
                                self.agent.reference_point[symbol] = self.agent.problem_nadir[symbol]
                                break
                            else:
                                if not maximize and self.agent.problem_ideal[symbol] <= float(value) <= self.agent.problem_nadir[symbol]:
                                    self.agent.reference_point[symbol] = float(value)
                                    break
                                elif maximize and self.agent.problem_ideal[symbol] >= float(value) >= self.agent.problem_nadir[symbol]:
                                    self.agent.reference_point[symbol] = float(value)
                                    break
                                else:
                                    raise ValueError()
                        except ValueError:
                            print(f"Value for reference point component {symbol} not valid!. Please enter a numerical value within range {range}, ideal or nadir.")
                contents = {"reference_point": self.agent.reference_point}
                msg = Message(
                    to="coordinator@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                #print("Preference agent sending a reference point...")
                await self.send(msg)
                file.write(f"Preference agent sent reference point {self.agent.reference_point} at {time.time()}\n")
                #print(f"A reference point sent: {self.agent.reference_point}.")
    
    async def setup(self):
        print("Preference agent started.")
        receive_inform_messages = self.ReceiveInformMessages()
        self.add_behaviour(receive_inform_messages, Template(metadata={"performative": "inform"}))
        receive_requests = self.ReceiveRequests()
        self.add_behaviour(receive_requests, Template(metadata={"performative": "request"}))

class SHAPAgent(Agent):
    def __init__(self, jid, password, sample_file: str = None, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.reference_point = None
        self.solution = None
        self.df = None
        self.sample_file = sample_file
        self.shap_model = None
        self.shaps = None
        self.problem = None
        self.objective_symbols = None

    class SendInformMessages(OneShotBehaviour):
        def __init__(self, content_type: str, receiver: str):
            super().__init__()
            self.content_type = content_type
            self.receiver = receiver

        async def run(self):
            if self.content_type == "shaps":
                msg = Message(
                    to=self.receiver,
                    body=json.dumps({"shaps": self.agent.shaps.tolist()}),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)

    class SampleReferenceSpace(OneShotBehaviour):
        async def run(self):
            problem = self.agent.problem
            ideal = problem.get_ideal_point()
            nadir = problem.get_nadir_point()
            # TODO: make this an argument
            n_samples = 20

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
                # TODO: allow other solvers/interactive methods?
                d = rpm_solve_solutions(problem, i)[0].optimal_objectives
                for key, value in i.items():
                    d[key.replace("f", "z")] = value
                outputs.append(d)
            
            self.agent.df = pl.DataFrame(outputs)

            fs = [symbol for symbol in self.agent.objective_symbols]
            zs = [f"z_{i+1}" for i in range(len(self.agent.objective_symbols))]

            self.agent.shap_model = ShapExplainer(problem_data=self.agent.df, input_symbols=zs, output_symbols=fs)

    class GenerateSHAPs(OneShotBehaviour):
        def __init__(self, reference_point):
            super().__init__()
            self.reference_point = reference_point

        async def run(self):
            await asyncio.sleep(1)
            file.write(f"STARTED TO GENERATE SHAPS AT {time.time()}\n")
            target = [value for _, value in self.reference_point.items()]
            # TODO: this solver should also be given as an argument, maybe pair it with the problem?
            #background_subset = generate_biased_mean_data(self.agent.df[["f_1", "f_2", "f_3"]].to_numpy(), target, solver="GUROBI")
            loop = asyncio.get_running_loop()
            fs = [symbol for symbol in self.agent.objective_symbols]
            zs = [f"z_{i+1}" for i in range(len(self.agent.objective_symbols))]
            data = self.agent.df[fs].to_numpy()
            try:
                background_subset = await loop.run_in_executor(
                    executor,
                    partial(
                        generate_biased_mean_data,
                        data,
                        target,
                        min_size=5,
                        max_size=20,
                        #solver="GUROBI"
                    )
                )
            except RuntimeError as e:
                print("Caught RuntimeError in executor:", e)
                return
            self.agent.shap_model.setup(background_data=pl.DataFrame(self.agent.df[background_subset]))
            z_dict = {}
            for i in range(len(zs)):
                z_dict[zs[i]] = target[i]
            self.agent.shaps = self.agent.shap_model.explain_input(pl.DataFrame(z_dict)).values[0].T
            file.write(f"FINISHED GENERATING SHAPS AT {time.time()}\n")
            self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps", receiver="coordinator@localhost"))

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "reference_point" in contents:
                    file.write(f"SHAP agent received reference point {contents["reference_point"]} at {time.time()}\n")
                    self.agent.reference_point = contents["reference_point"]
                    generate_shaps = self.agent.GenerateSHAPs(reference_point=contents["reference_point"])
                    self.agent.add_behaviour(generate_shaps)
                    #await generate_shaps.join() # just making sure the SHAPs are ready to be sent
                elif "problem" in contents:
                    problem = Problem.model_validate_json(contents["problem"])
                    #print(f"SHAP agent received the problem: {problem.name}")
                    self.agent.problem = problem
                    self.agent.n_objectives = len(problem.objectives)
                    self.agent.objective_symbols = [obj.symbol for obj in problem.objectives]
                    if self.agent.sample_file:
                        outputs = []
                        with Path.open("sample.json", "r") as f:
                            outputs = json.load(f)
                        self.agent.df = pl.DataFrame(outputs)
                        self.agent.shap_model = ShapExplainer(problem_data=self.agent.df, input_symbols=["z_1", "z_2", "z_3"], output_symbols=["f_1", "f_2", "f_3"])
                    else:
                        self.agent.add_behaviour(self.agent.SampleReferenceSpace())
                    

    async def setup(self):
        receive_inform_messages = self.ReceiveInformMessages()
        self.add_behaviour(receive_inform_messages, Template(metadata={"performative": "inform"}))
        print("SHAP agent started.")

class CoordinatorAgent(Agent):
    def __init__(self, jid, password, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.n_objectives = None
        self.problem = None
        self.problem_name = None
        self.objective_symbols = []
        self.target = None
        #self.explanation_type = None
        self.reference_point = None
        self.solution = None
        self.shaps = None
        self.explanation = None
        self.received_solution = False
        self.received_shaps = False

    class SendInformMessages(OneShotBehaviour):
        def __init__(self, content_type: str, sender: str = None, contents = None):
            super().__init__()
            self.content_type = content_type
            self.sender = sender
            self.contents = contents

        async def run(self):
            if self.content_type == "shaps":
                contents = {
                    "reference_point": self.agent.reference_point,
                    "solution": self.agent.solution,
                    "shaps": self.agent.shaps # coming as a list from SHAP agent
                }
                for symbol in self.agent.objective_symbols:
                    clean_symbol = symbol.translate(str.maketrans('', '', string.punctuation))
                    msg = Message(
                        to=f"explainer{clean_symbol}@localhost",
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
                self.agent.received_solution = False
                self.agent.received_shaps = False
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
            elif self.content_type == "problem":
                recipients = ["shapagent@localhost", "solver@localhost"]
                for recipient in recipients:
                    msg = Message(
                        to=recipient,
                        body=json.dumps({"problem": self.agent.problem_name}),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
            elif self.content_type == "solution":
                contents = {
                    "solution": self.agent.solution
                }
                msg = Message(
                    to="preferenceagent@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
            elif self.content_type == "reference point":
                contents = {
                    "reference_point": self.agent.reference_point
                }
                if self.sender:
                    msg = Message(
                        to="solver@localhost",
                        sender=self.sender,
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
                recipients = ["solver@localhost", "shapagent@localhost"]
                for recipient in recipients:
                    msg = Message(
                        to=recipient,
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
                    file.write(f"Coordinator sent message {msg} at {time.time()}\n")
            elif self.content_type == "example reference point":
                contents = {
                    "reference_point": self.contents["reference_point"]
                }
                msg = Message(
                    to="solver@localhost",
                    sender=self.sender,
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                file.write(f"Coordinator sent message {msg} at {time.time()}\n")
            elif self.content_type == "example solution":
                contents = {
                    "reference_point": self.contents["reference_point"],
                    "solution": self.contents["solution"]
                }
                msg = Message(
                    to=self.sender,
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)

    class SendRequests(OneShotBehaviour):
        # One use case for these requests would be that if, for some reason, some information has not been sent to or received by
        # the coordinator, the coordinator can request this information that should be stored on the corresponding agent
        def __init__(self, request_content: str):
            super().__init__()
            self.request_content = request_content

        async def run(self):
            #print(f"coordinator sending a request for {self.request_content}...")
            if self.request_content == "problem":
                msg = Message(
                    to = "preferenceagent@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                #print("coordinator requested the problem.")
            elif self.request_content == "examples":
                clean_symbol = self.agent.target.translate(str.maketrans('', '', string.punctuation))
                msg = Message(
                    to = f"explainer{clean_symbol}@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                #print("coordinator requested examples from the target explainer.")
    
    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print("coordinator ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "problem" in contents:
                    problem = Problem.model_validate_json(contents["problem"])
                    #print(f"coordinator received the problem: {problem.name}")
                    self.agent.add_behaviour(self.agent.SendInformMessages(content_type="problem"))
                    self.agent.problem = problem
                    self.agent.problem_name = contents["problem"]
                    self.agent.n_objectives = len(problem.objectives)
                    self.agent.objective_symbols = [obj.symbol for obj in problem.objectives]
                    self.agent.add_behaviour(self.agent.CreateExplainers())
                elif "target" in contents:
                    #print(f"coordinator received the target: {contents["target"]}")
                    self.agent.target = contents["target"]
                    #self.agent.explanation_type = contents["explanation type"]
                    self.agent.add_behaviour(self.agent.SendRequests(request_content="examples"))
                elif all(key in contents for key in ("reference_point", "solution")):
                    if "explainer" in msg.sender.full:
                        file.write(f"Coordinator received message {msg} at {time.time()}\n")
                        self.agent.add_behaviour(self.agent.SendInformMessages(content_type="example solution", sender=msg.sender.full, contents=contents))
                    else:
                        self.agent.solution = contents["solution"]
                        self.agent.received_solution = True
                        self.agent.add_behaviour(self.agent.SendInformMessages(content_type="solution"))
                        if self.agent.received_shaps:
                            self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps"))
                elif "reference_point" in contents:
                    if "explainer" in msg.sender.full:
                        file.write(f"Coordinator received message {msg} at {time.time()}\n")
                        #print(f"Coordinator received message {msg} at {time.time()}")
                        self.agent.add_behaviour(self.agent.SendInformMessages(content_type="example reference point", sender=msg.sender.full, contents=contents))
                    else:
                        file.write(f"Coordinator received message {msg} at {time.time()}\n")
                        self.agent.reference_point = contents["reference_point"]
                        self.agent.add_behaviour(self.agent.SendInformMessages(content_type="reference point"))
                elif "shaps" in contents:
                    #print("coordinator received SHAPs.")
                    self.agent.shaps = contents["shaps"]
                    self.agent.received_shaps = True
                    if self.agent.received_solution:
                        self.agent.add_behaviour(self.agent.SendInformMessages(content_type="shaps"))
                elif all(key in contents for key in ("examples", "explanation")):
                    #print(f"coordinator received examples: \n  {contents["examples"]}")
                    self.agent.examples = contents["examples"]
                    self.agent.explanation = contents["explanation"]
                    self.agent.add_behaviour(self.agent.SendInformMessages(content_type="explanation"))

    class CreateExplainers(OneShotBehaviour):
        async def run(self):
            #print(f"coordinator creating {self.agent.n_objectives} explainers...")
            for i in range(self.agent.n_objectives):
                objective = self.agent.objective_symbols[i]
                clean_symbol = objective.translate(str.maketrans('', '', string.punctuation))
                explainer = Explainer(f"explainer{clean_symbol}@localhost", "explainer", objective=objective, problem=self.agent.problem)
                await explainer.start(auto_register=True)
            msg = Message(
                to="preferenceagent@localhost",
                body=json.dumps("ready for target"),
                metadata={"performative": "inform"}
            )
            await self.send(msg)

    async def setup(self):
        print("Coordinator started.")
        self.add_behaviour(self.SendRequests(request_content="problem"))
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))


class Explainer(Agent):
    def __init__(self, jid, password, objective: str, problem: Problem, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.objective = objective
        self.problem = problem
        self.objective_symbols = [obj.symbol for obj in problem.objectives]
        self.problem_ideal = self.problem.get_ideal_point()
        self.problem_nadir = self.problem.get_nadir_point()
        self.can_send_explanations = False
        self.has_new_data = False
        self.ready_to_send_explanations = False # TODO: should probably get rid of these
        self.has_sent_explanations = False
        self.solution = None
        self.reference_point = None
        self.examples = None
        self.example_pairs = []
        self.explanation = None
        self.number_of_examples = 3
        self.iteration = 1
        self.max_iterations = 10 # TODO: make this an argument

    class ReceiveRequests(CyclicBehaviour):
        async def run(self):
            #print(f"{self.agent.jid.username} ready to receive requests.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "examples":
                    self.agent.can_send_explanations = True
                    if self.agent.ready_to_send_explanations:
                        self.agent.add_behaviour(self.agent.SendExamples(receiver=msg.sender.full))

    # TODO: separate these into two behaviours?
    class GenerateExamples(OneShotBehaviour):
        def __init__(self, number_of_examples: int):
            super().__init__()
            self.number_of_examples = number_of_examples

        async def run(self):
            self.agent.number_of_examples = self.number_of_examples
            shaps = self.agent.shaps
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
            # TODO: check if the objectives are min or max (actually, the current solution should work both ways)
            max_iterations = 10
            self.agent.iteration = 1
            self.agent.example_pairs = []
            i = 1
            while len(self.agent.example_pairs) < self.number_of_examples and i < self.agent.max_iterations:
                await asyncio.sleep(0)
                new_reference_point = self.agent.solution.copy()
                """if self.agent.reference_point == self.agent.problem_ideal or self.agent.reference_point == self.agent.problem_nadir:
                    new_reference_point = self.agent.solution.copy()
                else:
                    new_reference_point = self.agent.reference_point.copy()"""
                amount_to_improve = (self.agent.problem_ideal[to_improve] - new_reference_point[to_improve]) / (3*max_iterations/(i))
                new_reference_point[to_improve] = new_reference_point[to_improve] + amount_to_improve
                if to_impair:
                    amount_to_impair = (new_reference_point[to_impair] - self.agent.problem_nadir[to_impair]) / (3*max_iterations/(i))
                    new_reference_point[to_impair] = new_reference_point[to_impair] - amount_to_impair
                msg = Message(
                    to="coordinator@localhost",
                    body=json.dumps({"reference_point": new_reference_point}),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                file.write(f"{self.agent.jid} sent message {msg} at {time.time()}\n")
                #print(f"{self.agent.jid} sent message {msg} at {time.time()}")
                i = i + 1
            #self.agent.ready_to_send_explanations = True
            #print(f"{self.agent.jid.username} ready to send examples.")
            #print(examples)
    
    class SendExamples(OneShotBehaviour):
        def __init__(self, receiver: str):
            super().__init__()
            self.receiver = receiver

        async def run(self):
            #print(f"{self.agent.jid.username} sending examples...")
            contents = {
                "explanation": self.agent.explanation,
                "examples": self.agent.example_pairs
            }
            # TODO: should this be sent straight to the preference agent instead?
            msg = Message(
                to=self.receiver,
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            #print("Examples sent.")
            #self.agent.ready_to_send_explanations = False
            self.agent.has_new_data = False
            self.agent.can_send_explanations = False
            self.agent.has_sent_explanations = True
            self.agent.iteration = 1

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print(f"{self.agent.jid.username} ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if all(key in contents for key in ("shaps", "solution", "reference_point")):
                    #print(f"{self.agent.jid.username} received data: {contents}")
                    self.agent.solution = contents["solution"]
                    self.agent.reference_point = contents["reference_point"]
                    self.agent.shaps = contents["shaps"]
                    self.agent.has_new_data = True
                    self.agent.ready_to_send_explanations = False
                    self.agent.has_sent_explanations = False
                    self.agent.add_behaviour(self.agent.GenerateExamples(number_of_examples=3))
                    #if self.agent.can_send_explanations:
                        #self.agent.add_behaviour(self.agent.SendExamples(receiver=msg.sender.full))
                elif all(key in contents for key in ("reference_point", "solution")):
                    self.agent.iteration = self.agent.iteration + 1
                    unique = True
                    for (_, solution) in self.agent.example_pairs:
                        for symbol in self.agent.objective_symbols:
                            if np.isclose(contents["solution"][symbol], solution[symbol]):
                                unique = False
                                break
                    if unique and (contents["solution"][self.agent.objective] > self.agent.solution[self.agent.objective]) and len(self.agent.example_pairs) < self.agent.number_of_examples:
                        self.agent.example_pairs.append((contents["reference_point"], contents["solution"]))
                        if len(self.agent.example_pairs) >= self.agent.number_of_examples or self.agent.iteration >= self.agent.max_iterations:
                            self.agent.ready_to_send_explanations = True
                            if self.agent.can_send_explanations and not self.agent.has_sent_explanations:
                                self.agent.add_behaviour(self.agent.SendExamples(receiver=msg.sender.full))
                    elif self.agent.iteration >= self.agent.max_iterations:
                        self.agent.ready_to_send_explanations = True
                        if self.agent.can_send_explanations and not self.agent.has_sent_explanations:
                            self.agent.add_behaviour(self.agent.SendExamples(receiver=msg.sender.full))

    async def setup(self):
        print(f"{self.jid.username} started.")
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))
        receiveRequests = self.ReceiveRequests()
        self.add_behaviour(receiveRequests, Template(metadata={"performative": "request"}))

async def main():
    # initialize and start the solver
    solver_agent = Solver("solver@localhost", "solver", solver=rpm_solve_solutions)
    await solver_agent.start(auto_register=True)

    # initialize and start a SHAP agent
    shap_agent = SHAPAgent("shapagent@localhost", "shapagent", sample_file="sample.json")
    #shap_agent = SHAPAgent("shapagent@localhost", "shapagent", sample_file="test_sample.json")
    #shap_agent = SHAPAgent("shapagent@localhost", "shapagent")
    await shap_agent.start(auto_register=True)

    # initialize and start a preference agent
    preference_agent = PreferenceAgent("preferenceagent@localhost", "preferenceagent", problem=utopia_problem_old()[0])
    #preference_agent = PreferenceAgent("preferenceagent@localhost", "preferenceagent", problem=dtlz2(3,3))
    await preference_agent.start(auto_register=True)

    # initialize and start a coordinator agent
    coordinator_agent = CoordinatorAgent("coordinator@localhost", "coordinator")
    await coordinator_agent.start(auto_register=True)
    #coordinator_agent.web.start(hostname="127.0.0.1", port="10000")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        file.close()
        print("Stopping agents...")
        await solver_agent.stop()
        await coordinator_agent.stop()
        await preference_agent.stop()

if __name__ == "__main__":
    spade.run(main())