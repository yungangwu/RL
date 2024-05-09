from ddz.agent.agent import Agent
from ddz.game.utils import *

class AgentRecord(Agent):
    def __init__(self):
        super().__init__()

    def set_record_actions(self, actions):
        self.records = actions
        self.record_index = 0

    def get_action(self):
        cards = self.records[self.record_index]
        self.record_index = self.record_index + 1
        return cards
