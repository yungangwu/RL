from ddz.environment.env import Env
from ddz.game.utils import *

class EnvPredict(Env):
    def __init__(self):
        super(EnvPredict, self).__init__()

    def reset(self, initial_cards):
        self.played_cards = []
        for i in range(NUM_AGENT):
            self.agents[i].reset(initial_cards[i])
        self.winner = None
        self.round = 0

    def step(self, position, cards):
        self.round = position
        agent = self.agents[position]
        agent.handout_cards(cards)
        self.played_cards.extend(cards.copy())
        if DEBUG:
            if cards:
                print("agent [{}] handout: {}".format(
                    position, cards_to_str(cards)))
            else:
                print("agent [{}] handout: pass".format(position))
        done = agent.end()
        if done:
            if position != 0:
                self.winner = self.get_farmer()
            else:
                self.winner = self.get_landlord()
        self.round = (self.round + 1) % 3
        return done
    


    
