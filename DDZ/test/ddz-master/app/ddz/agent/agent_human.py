from ddz.agent.agent import Agent
from ddz.game.utils import *

class AgentHuman(Agent):
    def __init__(self):
        super(AgentHuman, self).__init__()

    def handout_cards(self, cards):
        self.actions.append(cards.copy())
        return cards.copy()

    def reset(self, cards):
        self.actions = []

    def accept_bonus_cards(self, cards):
        raise RuntimeError("human agent can not accept bonus")


    def end(self):
        total_card_count_list = [20, 17, 17]
        total_card_count = total_card_count_list[self.position]
        handout_count = 0
        for action in self.actions:
            handout_count = handout_count + len(action)
        return handout_count >= total_card_count

    def get_legal_actions(self):
        raise RuntimeError("human agent can not get legal actions")
        