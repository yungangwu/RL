from utils import *
from state import State
import random

class Env(object):
    def __init__(self):
        super().__init__()
        self.agents = [None for i in range(NUM_AGENT)]
        
    def set_agent(self, agent, position):
        self.agents[position] = agent
        agent.set_position(position)
        agent.set_env(self)

    def get_state(self, position):
        state = State(position)
        state.copy_from_env(self)
        return state

    def get_all_cards(self):
        all_cards = [(x + 1, y + 1) for x in range(NUM_CARD_TYPE) for y in range(NUM_REGULAR_CARD_VALUE)]
        all_cards.extend([(1, SMALL_JOKER), (1, BIG_JOKER)])
        return all_cards

    def reset(self, seed=None):
        self.played_cards = []
        all_cards = self.get_all_cards()
        if seed:
            random.seed(seed)
        else:
            random.seed()
        random.shuffle(all_cards)
        self.bonus_cards = all_cards[:3]
        player_cards = all_cards[3:]
        for i in range(NUM_AGENT):
            self.agents[i].reset(player_cards[17*i:17*(i+1)])
        self.winner = None
        self.agents[0].accept_bonus_cards(self.bonus_cards)
        self.round = 0

    def get_agent(self, position):
        position = (position + NUM_AGENT) % NUM_AGENT
        return self.agents[position]

    def step(self, position, cards):
        self.round = position
        agent = self.agents[position]
        if DEBUG:
            print("agent [{}] handcards: {}".format(
                position, cards_to_str(agent.handcards)))
        handout_cards = agent.handout_cards(cards)
        self.played_cards.extend(handout_cards)
        if DEBUG:
            if handout_cards:
                print("agent [{}] handout: {}".format(
                    position, cards_to_str(handout_cards)))
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

    def get_farmer(self):
        return [1, 2]

    def get_landlord(self):
        return [0]

    def get_remain_cards(self):
        all_cards = self.get_all_cards()
        return [x for x in all_cards if x not in self.played_cards]

    def get_winner(self):
        return self.winner 

    def mcts_enabled(self):
        remain_card_nums = [20, 17, 17]
        for i, agent in enumerate(self.agents):
            count = 0
            actions = agent.get_history_actions()
            for action in actions:
                count = count + len(action)
            if remain_card_nums[i] - count <= 15:
                return True
        return False

    def end(self):
        return self.winner != None

    def get_follow_cards(self, position):
        for i in range(2):
            agent = self.get_agent(position - i - 1)
            cards = agent.get_last_action()
            if cards:
                return cards
        return None
           
        
