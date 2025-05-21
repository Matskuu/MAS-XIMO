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
from desdeo.utopia_stuff.utopia_problem_old import utopia_problem_old


shap_model = None

PROBLEM_ENUM = {
    "utopia_problem_old": utopia_problem_old()[0]
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
            contents = {"solution": self.agent.solution}
            msg = Message(
                to="preferenceagent@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print(f"Solution sent to preference agent: {self.agent.solution}")

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
    def __init__(self, jid, password, problem, max_iterations, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.can_send_target = False
        self.received_solution = False
        self.problem = problem
        self.objective_symbols = [obj.symbol for obj in problem.objectives]
        self.problem_ideal = problem.get_ideal_point()
        self.problem_nadir = problem.get_nadir_point()
        self.reference_point = {}
        self.solution = None
        self.iteration = 0
        self.max_iterations = max_iterations

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
                new_reference_point = self.agent.solution
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
                elif "explanation" in contents:
                    print("Preference agent received an explanation.")
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
                print("Preference agent thinking of an objective to improve...")
                await asyncio.sleep(5)
                print("Preference agent sending the target...")
                await self.send(msg)
                print(f"Target sent: {target}.")
            elif self.content_type == "problem":
                print("Preference agent sending the problem...")
                #problem_name = input("Provide the problem name: ")
                recipients = ["solver@localhost", "explanationgatherer@localhost"]
                for recipient in recipients:
                    contents = {"problem": "utopia_problem_old"}
                    msg = Message(
                        to=recipient,
                        body=json.dumps(contents),
                        metadata={"performative": "inform"}
                    )
                    await self.send(msg)
                print(f"The problem sent: {self.agent.problem.name}.")
            elif self.content_type == "reference point":
                if self.agent.iteration == 0:
                    print("Problem has been set up. Press C to start the solution process.")
                    while True:
                        if keyboard.is_pressed("c"):
                            break
                        await asyncio.sleep(0.1)
                    print("Starting the solution process...")
                for symbol in self.agent.objective_symbols:
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
                print("Preference agent choosing a reference point...")
                await asyncio.sleep(5) #Preference agent thinking
                print("Preference agent sending a reference point...")
                await self.send(msg)
                print(f"A reference point sent: {self.agent.reference_point}.")
                self.agent.iteration = self.agent.iteration + 1
    
    async def setup(self):
        print("Preference agent started.")
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))
        receiveRequests = self.ReceiveRequests()
        self.add_behaviour(receiveRequests, Template(metadata={"performative": "request"}))

class ExplanationGatherer(Agent):
    def __init__(self, jid, password, df: pl.DataFrame, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.n_objectives = None
        self.problem = None
        self.objective_symbols = []
        self.target = 0
        self.df = df

    class SendExplanations(OneShotBehaviour):
        def __init__(self, explanation):
            super().__init__()
            self.explanation = explanation

        async def run(self):
            #print(f"Explanation gatherer sending the explanation {self.explanation} to Preference agent...")
            msg = Message(
                to = "preferenceagent@localhost",
                body = json.dumps({"explanation": self.explanation}),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print("Explanation gatherer sent the explanation to Preference agent.")

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
            elif self.request_content == "explanation":
                msg = Message(
                    to = f"explainer{self.agent.target}@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                print("Explanation gatherer requested an explanation from the target explainer.")
    
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
                    self.agent.add_behaviour(self.agent.CreateExplainers(objective_symbols=self.agent.objective_symbols))
                elif "target" in contents:
                    print(f"Explanation gatherer received the target: {contents["target"]}")
                    self.agent.target = contents["target"].translate(str.maketrans('', '', string.punctuation))
                    self.agent.add_behaviour(self.agent.SendRequests("explanation"))
                elif "explanation" in contents:
                    print(f"Explanation gatherer received explanations: \n    {contents["explanation"]}")
                    self.agent.add_behaviour(self.agent.SendExplanations(contents["explanation"]))
                elif "solution" in contents:
                    print(f"Explanation gatherer received a solution and reference point: {contents}")
                    for i in range(len(self.agent.objective_symbols)):
                        objective = self.agent.objective_symbols[i]
                        clean_symbol = objective.translate(str.maketrans('', '', string.punctuation))
                        msg_to_send = Message(
                            to=f"explainer{clean_symbol}@localhost",
                            body=msg.body,
                            metadata={"performative": "inform"}
                        )
                        await self.send(msg_to_send)
                    print("Explanation gatherer shared the solution and reference point with the explainers.")

    class CreateExplainers(OneShotBehaviour):
        def __init__(self, objective_symbols, **kwargs):
            super().__init__(**kwargs)
            self.objective_symbols = objective_symbols

        async def run(self):
            print(f"Explanation gatherer creating {self.agent.n_objectives} explainers...")
            for i in range(self.agent.n_objectives):
                objective = self.objective_symbols[i]
                clean_symbol = objective.translate(str.maketrans('', '', string.punctuation))
                explainer = Explainer(f"explainer{clean_symbol}@localhost", "explainer", objective=objective, df=self.agent.df)
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
    def __init__(self, jid, password, objective: str, df: pl.DataFrame, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.objective = objective
        self.df = df
        self.can_send_explanation = False
        self.has_new_data = False
        self.solution = None
        self.reference_point = None

    class SendExplanations(OneShotBehaviour):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def generate_explanations(self, reference_point: dict):
            target = [value for _, value in reference_point.items()]
            background_subset = generate_biased_mean_data(self.agent.df[["f_1", "f_2", "f_3"]].to_numpy(), target, solver="GUROBI")
            shap_model.setup(background_data=pl.DataFrame(self.agent.df[background_subset]))
            shaps = shap_model.explain_input(pl.DataFrame({"z_1": target[0], "z_2": target[1], "z_3": target[2]})).values[0].T
            shaps_dict = {
                "f_1": shaps[0],
                "f_2": shaps[1],
                "f_3": shaps[2]
            }
            shaps_for_this_agent = shaps_dict[self.agent.objective]
            shaps_for_this_agent_dict = {
                "f_1": shaps_for_this_agent[0],
                "f_2": shaps_for_this_agent[1],
                "f_3": shaps_for_this_agent[2]
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
            #print(self.agent.objective, shaps_for_this_agent_dict, to_improve, to_impair)
            explanation = f"To get better value for objective {self.agent.objective}, try to improve objective {to_improve} and impair objective {to_impair} in the reference point."
            return explanation

        async def run(self):
            explanation = self.generate_explanations(self.agent.reference_point)
            print(f"{self.agent.jid.username} sending explanations...")
            contents = {
                "explanation": explanation
            }
            msg = Message(
                to="explanationgatherer@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )

            await self.send(msg)
            print("Explanations sent.")

            self.agent.has_new_data = False
            self.agent.can_send_explanation = False

    class ReceiveRequests(CyclicBehaviour):
        async def run(self):
            #print(f"{self.agent.jid.username} ready to receive requests.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "explanation":
                    self.agent.can_send_explanation = True
                    if self.agent.has_new_data:
                        self.agent.add_behaviour(self.agent.SendExplanations())
                        self.agent.can_send_explanation = False

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print(f"{self.agent.jid.username} ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "solution" and "reference_point" in contents:
                    #print(f"{self.agent.jid.username} received data: {contents}")
                    self.agent.solution = contents["solution"]
                    self.agent.reference_point = contents["reference_point"]
                    self.agent.has_new_data = True
                    if self.agent.can_send_explanation:
                        self.agent.add_behaviour(self.agent.SendExplanations())
                        self.agent.has_new_data = False

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
    preferenceAgent = PreferenceAgent("preferenceagent@localhost", "preferenceagent", problem=PROBLEM_ENUM["utopia_problem_old"], max_iterations=5)
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

def sample_input_space_to_file(n_samples: int):
    problem = PROBLEM_ENUM["utopia_problem_old"]
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    n_samples = n_samples

    bounds = {
        "f_1": (nadir["f_1"], ideal["f_1"]),
        "f_2": (nadir["f_2"], ideal["f_2"]),
        "f_3": (nadir["f_3"], ideal["f_3"])
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

    with Path.open("sample.json", "w") as file:
        json.dump(outputs, file)

if __name__ == "__main__":
    # run the multi-agent system with three objectives (i.e., three explainers)

    # this seems like something the explanation gatherer should do as soon as it gets the problem
    #sample_input_space_to_file(n_samples=20)
    
    outputs = []

    with Path.open("sample.json", "r") as file:
        outputs = json.load(file)

    df = pl.DataFrame(outputs)

    shap_model = ShapExplainer(problem_data=df, input_symbols=["z_1", "z_2", "z_3"], output_symbols=["f_1", "f_2", "f_3"])

    spade.run(main(df=df))