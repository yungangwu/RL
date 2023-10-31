
from ddz.game.utils import *
import numpy as np


class PolicyRandom(object):

    def evaluate(self, agent):
        legal_actions = agent.get_legal_actions()
        if legal_actions:
            action_index = np.random.choice(len(legal_actions))
            action = legal_actions[action_index]
            if DEBUG:
                print("random policy evaluate action:",action)
        else:
            action = []
            if DEBUG:
                print("random policy evaluate action: pass")
        return action
