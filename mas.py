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
    class SendTarget(OneShotBehaviour):
        async def run(self):
            print("Forest owner sending the target...")
            target = random.randint(0, 2)
            msg = Message(
                to="explanationgatherer@localhost",
                body=f"{target}",
                metadata={"performative": "target_preference"}
            )

            await self.send(msg)
            print(f"Target sent: {target}.")
    
    async def setup(self):
        print("Forest owner started.")
        sendTarget = self.SendTarget()
        self.add_behaviour(sendTarget)

class ExplanationGatherer(Agent):
    def __init__(self, jid, password, n_objectives, port = 5222, verify_security = False, **kwargs):
        super().__init__(jid, password, port, verify_security, **kwargs)
        self.n_objectives = n_objectives

    class ReceiveExplanations(CyclicBehaviour):
        async def run(self):
            print("Explanation gatherer ready to receive explanations.")
            msg = await self.receive(timeout=10)
            if msg:
                print(f"Explanation gatherer received explanations: \n    {msg.body}")
            else:
                print("Explanation gatherer has not received explanations after 10 seconds.")
                self.kill()

    class ReceiveTarget(CyclicBehaviour):
        async def run(self):
            print("Explanation gatherer ready to receive target.")
            msg = await self.receive(timeout=10)
            if msg:
                print("Explanation gatherer received the target: {}".format(msg.body))
                self.agent.add_behaviour(ExplanationGatherer.EmployTarget(msg.body))
            else:
                print("Explanation gatherer has not received a target after 10 seconds.")
                self.kill()

    class EmployTarget(OneShotBehaviour):
        def __init__(self, target, **kwargs):
            super().__init__(**kwargs)
            self.target = target

        async def run(self):
            print("Explanation gatherer employing the target...")
            data = {"something": 10}
            msg = Message(
                to=f"explanationagent{self.target}@localhost",
                body=json.dumps(data),
                metadata={"performative": "data"}
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

    async def setup(self):
        print("Explanation gatherer started.")
        self.createExplainers = self.CreateExplainers(n_objectives=self.n_objectives)
        self.add_behaviour(self.createExplainers)
        receiveExplanations = self.ReceiveExplanations()
        self.add_behaviour(receiveExplanations, Template(metadata={"performative": "explanations"}))
        receiveTarget = self.ReceiveTarget()
        self.add_behaviour(receiveTarget, Template(metadata={"performative": "target_preference"}))


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
            msg = Message(
                to="explanationgatherer@localhost",
                body=f"These are the explanations based on this data: {self.data}.\n    To improve objective {self.agent.objective}, you need to impair objective {self.agent.rival} by {self.data["something"] + self.agent.objective}.",
                metadata={"performative": "explanations"}
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
                self.agent.add_behaviour(ExplanationAgent.SendExplanations(data))
            else:
                print(f"{self.agent.jid.username} has not received a message after 10 seconds.")
                self.kill()

    async def setup(self):
        print(f"{self.jid.username} started.")
        receiveData = self.ReceiveData()
        self.add_behaviour(receiveData, Template(metadata={"performative": "data"}))

async def main(n_objectives: int):
    # initialize and start an explanation gatherer
    explanationGatherer = ExplanationGatherer("explanationgatherer@localhost", "preference", n_objectives=n_objectives)
    await explanationGatherer.start(auto_register=True)

    # wait for the explainers to be created and started
    await explanationGatherer.createExplainers.join()

    # initialize and start a forest owner
    forestOwner = ForestOwner("forestowner@localhost", "forestowner")
    await forestOwner.start(auto_register=True)

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