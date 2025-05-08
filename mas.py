import asyncio
import json
import random
import spade
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message
from spade.template import Template


class ForestOwner(Agent):
    def __init__(self, jid, password, port = 5222, verify_security = False):
        super().__init__(jid, password, port, verify_security)
        self.n_objectives = 3
        self.can_send_target = False

    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            print("Forest owner waiting for messages.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "ready for target":
                    self.agent.can_send_target = True
                    # TODO: do this when solver has sent a solution
                    if self.agent.can_send_target:
                        self.agent.add_behaviour(self.agent.SendData(content_type="target"))

    class ReceiveRequests(CyclicBehaviour):
        async def run(self):
            print("Forest owner waiting for requests.")
            msg = await self.receive(timeout=10)
            if msg:
                if msg.body == "target":
                    self.agent.add_behaviour(self.agent.SendData(content_type="target"))
                elif msg.body == "number of objectives":
                    self.agent.add_behaviour(self.agent.SendData(content_type="number of objectives"))

    class SendData(OneShotBehaviour):
        def __init__(self, content_type: str):
            super().__init__()
            self.content_type = content_type

        async def run(self):
            if self.content_type == "target":
                print("Forest owner sending the target...")
                target = random.randint(0, self.agent.n_objectives - 1)
                contents = {"target": target}
                msg = Message(
                    to="explanationgatherer@localhost",
                    body=f"{json.dumps(contents)}",
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                print(f"Target sent: {target}.")
            elif self.content_type == "number of objectives":
                print("Forest owner sending the number of objectives...")
                n_objectives = self.agent.n_objectives
                contents = {"n_objectives": n_objectives}
                msg = Message(
                    to="explanationgatherer@localhost",
                    body=f"{json.dumps(contents)}",
                    metadata={"performative": "inform"}
                )
                await self.send(msg)
                print(f"Number of objectives sent: {n_objectives}.")
    
    async def setup(self):
        print("Forest owner started.")
        self.n_objectives = random.randint(2, 5)
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))
        receiveRequests = self.ReceiveRequests()
        self.add_behaviour(receiveRequests, Template(metadata={"performative": "request"}))
        #sendTarget = self.SendTarget()
        #self.add_behaviour(sendTarget)

class ExplanationGatherer(Agent):
    def __init__(self, jid, password, n_objectives, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.n_objectives = n_objectives

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
    
    class ReceiveInformMessages(CyclicBehaviour):
        async def run(self):
            print("Explanation gatherer ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                contents = json.loads(msg.body)
                if "target" in contents:
                    print(f"Explanation gatherer received the target: {contents["target"]}")
                    self.agent.add_behaviour(self.agent.EmployTarget(contents["target"]))
                elif "explanation" in contents:
                    print(f"Explanation gatherer received explanations: \n    {contents["explanation"]}")
                elif "n_objectives" in contents:
                    print(f"Explanation gatherer received the number of objectives : {contents["n_objectives"]}.")
                    self.agent.add_behaviour(self.agent.CreateExplainers(n_objectives=contents["n_objectives"]))
            #else:
                #print("Explanation gatherer has not received a target after 10 seconds.")
                #self.kill()

    class EmployTarget(OneShotBehaviour):
        def __init__(self, target, **kwargs):
            super().__init__(**kwargs)
            self.target = target

        async def run(self):
            #await self.agent.createExplainers.join() # this has to wait until all explainers are created
            print("Explanation gatherer employing the target...")
            data = {"CO2 level": 10}
            msg = Message(
                to=f"explanationagent{self.target}@localhost",
                body=json.dumps(data),
                metadata={"performative": "inform"}
            )
            await self.send(msg)
            print("Target employed.")

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
                body="ready for target",
                metadata={"performative": "inform"}
            )
            await self.send(msg)

    async def setup(self):
        print("Explanation gatherer started.")
        self.add_behaviour(self.SendRequests(request_content="number of objectives"))
        #self.createExplainers = self.CreateExplainers(n_objectives=self.n_objectives)
        #self.add_behaviour(self.createExplainers)
        #self.add_behaviour(self.SendRequests(request_content="target"))
        receiveInformMessages = self.ReceiveInformMessages()
        self.add_behaviour(receiveInformMessages, Template(metadata={"performative": "inform"}))


class ExplanationAgent(Agent):
    def __init__(self, jid, password, objective: int, n_objectives: int, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.objective = objective
        self.n_objectives = n_objectives
        self.rival = (objective + 1) % n_objectives

    class SendExplanations(OneShotBehaviour):
        def __init__(self, data, **kwargs):
            super().__init__(**kwargs)
            self.data = data

        async def run(self):
            print(f"{self.agent.jid.username} sending explanations...")
            contents = {
                "explanation": f"These are the explanations based on this data: {self.data}.\n    To improve objective {self.agent.objective}, you need to impair objective {self.agent.rival} by {self.data["CO2 level"] + self.agent.objective}."
            }
            msg = Message(
                to="explanationgatherer@localhost",
                body=json.dumps(contents),
                metadata={"performative": "inform"}
            )

            await self.send(msg)
            print("Explanations sent.")

            self.exit_code = "Explanations sent"

    class ReceiveData(CyclicBehaviour):
        def __init__(self):
            super().__init__()

        async def run(self):
            print(f"{self.agent.jid.username} ready to receive data.")
            msg = await self.receive(timeout=10)
            if msg:
                data = json.loads(msg.body)
                print(f"{self.agent.jid.username} received data: {data}")
                self.agent.add_behaviour(self.agent.SendExplanations(data))
            """else:
                print(f"{self.agent.jid.username} has not received a message after 10 seconds.")
                #self.kill()"""

    async def setup(self):
        print(f"{self.jid.username} started.")
        receiveData = self.ReceiveData()
        self.add_behaviour(receiveData, Template(metadata={"performative": "inform"}))

async def main(n_objectives: int):
    # initialize and start a forest owner
    forestOwner = ForestOwner("forestowner@localhost", "forestowner")
    await forestOwner.start(auto_register=True)

    # initialize and start an explanation gatherer
    explanationGatherer = ExplanationGatherer("explanationgatherer@localhost", "preference", n_objectives=n_objectives)
    await explanationGatherer.start(auto_register=True)
    explanationGatherer.web.start(hostname="127.0.0.1", port="10000")

    # wait for the explainers to be created and started
    #await explanationGatherer.createExplainers.join()

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