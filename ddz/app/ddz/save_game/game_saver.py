from ddz.game.utils import *
from ddz.agent.agent_mcts import AgentMCTS
import numpy as np

class GameSaver(object):
    def __init__(self):
        super().__init__()
        self.mcts = []
        self.records = []

    def save_mcts_record(self, agents, winner):
        for n, agent in enumerate(agents):
            if isinstance(agent, AgentMCTS):
                ms, ma = zip(*agent.memories)
                is_win = 1 if (n in winner) else -1
                self.mcts.append([n, ms, ma, is_win])
                

    def save_game_record(self, agents):
        initial_cards = [None for _ in range(NUM_AGENT)]
        for n, agent in enumerate(agents):
            initial_cards[n] = agent.initial_cards.copy()
        actions = []
        done = False
        index = 0
        while not done:
            for agent in agents:
                if index < len(agent.actions):
                    action = agent.actions[index]
                    actions.append(action)
                else:
                    done = True
                    break
            index = index + 1
        # for cards in initial_cards:
        #     print("initial cards:", cards_to_str(cards))
        # for action in actions:
        #     print("actions:", cards_to_str(action))
        self.records.append([initial_cards, actions])


    def save_mcts(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            np.save(f, self.mcts)

    def save_game(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open(filepath, 'wb') as f:
            np.save(f, self.records)