from ddz.environment.env import Env
from ddz.game.utils import *

class EnvRecord(Env):
    def __init__(self):
        super(EnvRecord, self).__init__()

    def reset(self, initial_cards, actions):
        self.played_cards = []
        for i in range(NUM_AGENT):
            self.agents[i].reset(initial_cards[i])
            self.agents[i].set_record_actions(actions[i::3])
        self.winner = None
        self.round = 0


    
