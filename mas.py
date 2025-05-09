import asyncio
import json
import random
import re
import spade

from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template

from desdeo.utopia_stuff.utopia_problem_old import utopia_problem_old


problem = {
    "ideal": {"1": 2, "2": 3, "3": 4}
}

def solver(problem, reference_point):
    return {"1": 1, "2": 2, "3": 3}

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
                to="forestowner@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print(f"Solution sent to forest owner: {self.agent.solution}")

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
            self.agent.solution = self.agent.solver(self.agent.problem, self.agent.reference_point)
            self.agent.add_behaviour(self.agent.SendSolution())

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print("Solver waiting for messages.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "problem" in contents:
                    print(f"Solver received the problem: {contents["problem"]}.")
                    self.agent.problem = contents["problem"]
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

class ForestOwner(Agent):
    def __init__(self, jid, password, problem, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.n_objectives = 3
        self.can_send_target = False
        self.received_solution = False
        self.problem = problem
        self.reference_point = None

    class ReceiveInformMessages(CyclicBehaviour):
        def get_updated_reference_point(self, explanation: str, previous_reference_point):
            previous = previous_reference_point
            if previous:
                explanation_split = re.split(r"[ ,]+", explanation)
                objective_to_improve = int(explanation_split[explanation_split.index("improve") + 2])
                objective_to_impair = int(explanation_split[explanation_split.index("impair") + 2])
                amount_to_impair = float(re.sub(r"\D+$", "", explanation_split[explanation_split.index("by") + 1]))
                print(objective_to_improve, objective_to_impair, amount_to_impair)
                new_reference_point = previous
                new_reference_point["1"] = new_reference_point["1"] + amount_to_impair
                print(new_reference_point)
                return new_reference_point

        async def run(self):
            #print("Forest owner waiting for messages.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "ready for target" in contents:
                    self.agent.can_send_target = True
                    self.agent.add_behaviour(self.agent.SendData("problem"))
                    self.agent.add_behaviour(self.agent.SendData("reference point", content=self.agent.problem["ideal"])) # initially reference point is the ideal
                    if self.agent.received_solution:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                        self.agent.received_solution = False
                elif "solution" in contents:
                    self.agent.received_solution = True
                    if self.agent.can_send_target:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                        self.agent.received_solution = False
                elif "explanation" in contents:
                    print("Forest owner received an explanation.")
                    reference_point = self.get_updated_reference_point(contents["explanation"], self.agent.reference_point)
                    self.agent.add_behaviour(self.agent.SendData("reference point", content=reference_point))


    class ReceiveRequests(CyclicBehaviour):
        async def run(self):
            #print("Forest owner waiting for requests.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "target":
                    self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                elif msg.body == "number of objectives":
                    self.agent.add_behaviour(self.agent.SendData(content_type="number of objectives"))

    class SendData(OneShotBehaviour):
        def __init__(self, content_type: str, content = None):
            super().__init__()
            self.content_type = content_type
            self.content = content

        async def run(self):
            if self.content_type == "target":
                target = random.randint(0, self.agent.n_objectives - 1)
                contents = {"target": target}
                msg = Message(
                    to="explanationgatherer@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                print("Forest owner thinking of an objective to improve...")
                await asyncio.sleep(2)
                print("Forest owner sending the target...")
                await self.send(msg)
                print(f"Target sent: {target}.")
            elif self.content_type == "number of objectives":
                print("Forest owner sending the number of objectives...")
                n_objectives = self.agent.n_objectives
                contents = {"n_objectives": n_objectives}
                msg = Message(
                    to="explanationgatherer@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                print(f"Number of objectives sent: {n_objectives}.")
            elif self.content_type == "problem":
                print("Forest owner sending the problem...")
                contents = {"problem": self.agent.problem}
                msg = Message(
                    to="solver@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                print(f"The problem sent: {self.agent.problem}.")
            elif self.content_type == "reference point" and self.content:
                self.agent.reference_point = self.content
                contents = {"reference_point": self.content}
                msg = Message(
                    to="solver@localhost",
                    body=json.dumps(contents),
                    metadata={"performative": "inform"}
                )
                print("Forest owner choosing a reference point...")
                await asyncio.sleep(5) #forest owner thinking
                print("Forest owner sending a reference point...")
                await self.send(msg)
                print(f"A reference point sent: {self.agent.problem["ideal"]}.")
    
    async def setup(self):
        print("Forest owner started.")
        self.n_objectives = random.randint(2, 5)
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))
        receiveRequests = self.ReceiveRequests()
        self.add_behaviour(receiveRequests, Template(metadata={"performative": "request"}))

class ExplanationGatherer(Agent):
    def __init__(self, jid, password, n_objectives, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.n_objectives = n_objectives
        self.target = 0

    class SendExplanations(OneShotBehaviour):
        def __init__(self, explanation):
            super().__init__()
            self.explanation = explanation

        async def run(self):
            print(f"Explanation gatherer sending the explanation {self.explanation} to forest owner...")
            msg = Message(
                to = "forestowner@localhost",
                body = json.dumps({"explanation": self.explanation}),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print("Explanation gatherer sent the explanation to forest owner.")

            #self.agent.add_behaviour(self.agent.SendRequests("target"))

    class SendRequests(OneShotBehaviour):
        def __init__(self, request_content: str):
            super().__init__()
            self.request_content = request_content

        async def run(self):
            print(f"Explanation gatherer sending a request for {self.request_content}...")
            if self.request_content == "target":
                msg = Message(
                    to = "forestowner@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                print("Explanaton gatherer requested a target.")
            elif self.request_content == "number of objectives":
                msg = Message(
                    to = "forestowner@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                print("Explanaton gatherer requested a number of objectives.")
            elif self.request_content == "explanation":
                msg = Message(
                    to = f"explanationagent{self.agent.target}@localhost",
                    body = self.request_content,
                    metadata={"performative": "request"}
                )
                await self.send(msg)
                print("Explanaton gatherer requested an explanation from the target explainer.")
    
    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            #print("Explanation gatherer ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "target" in contents:
                    print(f"Explanation gatherer received the target: {contents["target"]}")
                    self.agent.target = contents["target"]
                    self.agent.add_behaviour(self.agent.SendRequests("explanation"))
                elif "explanation" in contents:
                    print(f"Explanation gatherer received explanations: \n    {contents["explanation"]}")
                    self.agent.add_behaviour(self.agent.SendExplanations(contents["explanation"]))
                elif "n_objectives" in contents:
                    print(f"Explanation gatherer received the number of objectives : {contents["n_objectives"]}.")
                    self.agent.n_objectives = contents["n_objectives"]
                    self.agent.add_behaviour(self.agent.CreateExplainers(n_objectives=contents["n_objectives"]))
                elif "solution" in contents:
                    print(f"Explanation gatherer received a solution and reference point: {contents}")
                    for i in range(self.agent.n_objectives):
                        msg_to_send = Message(
                            to=f"explanationagent{i}@localhost",
                            body=msg.body,
                            metadata={"performative": "inform"}
                        )
                        await self.send(msg_to_send)
                    print("Explanation gatherer shared the solution and reference point with the explainers.")

            #else:
                #print("Explanation gatherer has not received a target after 10 seconds.")
                #self.kill()

    class CreateExplainers(OneShotBehaviour):
        def __init__(self, n_objectives, **kwargs):
            super().__init__(**kwargs)
            self.n_objectives = n_objectives

        async def run(self):
            print(f"Explanation gatherer creating {self.n_objectives} explainers...")
            for i in range(self.n_objectives):
                explanationAgent = ExplanationAgent(f"explanationagent{i}@localhost", "explainer", objective=i, n_objectives=self.n_objectives)
                await explanationAgent.start(auto_register=True)
            msg = Message(
                to="forestowner@localhost",
                body=json.dumps("ready for target"),
                metadata={"performative": "inform"}
            )
            await self.send(msg)

    async def setup(self):
        print("Explanation gatherer started.")
        self.add_behaviour(self.SendRequests(request_content="number of objectives"))
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))


class ExplanationAgent(Agent):
    def __init__(self, jid, password, objective: int, n_objectives: int, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.objective = objective
        self.rival = (objective + 1) % n_objectives
        self.can_send_explanation = False
        self.has_new_data = False
        self.solution = None
        self.reference_point = None

    class SendExplanations(OneShotBehaviour):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        async def run(self):
            print(f"{self.agent.jid.username} sending explanations...")
            contents = {
                "explanation": f"""These are the explanations based on this data: {self.agent.solution, self.agent.reference_point}.
                To improve objective {self.agent.objective}, you need to impair objective {self.agent.rival} by {self.agent.solution["1"] / 10}."""
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
                    print(f"{self.agent.jid.username} received data: {contents}")
                    self.agent.solution = contents["solution"]
                    self.agent.reference_point = contents["reference_point"]
                    self.agent.has_new_data = True
                    if self.agent.can_send_explanation:
                        self.agent.add_behaviour(self.agent.SendExplanations())
                        self.agent.has_new_data = False
            """else:
                print(f"{self.agent.jid.username} has not received a message after 10 seconds.")
                #self.kill()"""

    async def setup(self):
        print(f"{self.jid.username} started.")
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))
        receiveRequests = self.ReceiveRequests()
        self.add_behaviour(receiveRequests, Template(metadata={"performative": "request"}))

async def main(n_objectives: int):
    # initialize and start the solver
    solverAgent = Solver("solver@localhost", "solver", solver=solver)
    await solverAgent.start(auto_register=True)

    # initialize and start a forest owner
    forestOwner = ForestOwner("forestowner@localhost", "forestowner", problem=problem)
    await forestOwner.start(auto_register=True)

    # initialize and start an explanation gatherer
    explanationGatherer = ExplanationGatherer("explanationgatherer@localhost", "preference", n_objectives=n_objectives)
    await explanationGatherer.start(auto_register=True)
    #explanationGatherer.web.start(hostname="127.0.0.1", port="10000")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping agents...")
        await explanationGatherer.stop()
        await forestOwner.stop()

if __name__ == "__main__":
    # run the multi-agent system with three objectives (i.e., three explainers)
    # NOTE: the number of objectives should come from the problem
    spade.run(main(n_objectives=3))