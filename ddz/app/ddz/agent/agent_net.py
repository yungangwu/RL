from ddz.agent.agent import Agent
from ddz.game.utils import *
from ddz.policy.policy_manager import policy_manager
from ddz.game.move import move_manager
import random
import numpy as np


class AgentNet(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        self._kwargs = kwargs.copy()
        self._temperature = self._kwargs['temperature'] if 'temperature' in self._kwargs else 0
        self._minor_net = self._kwargs['minor'] if 'minor' in self._kwargs else 'net'

    def set_position(self, position):
        super(AgentNet, self).set_position(position)
        name = "play_net_{}_{}".format(position, self._kwargs['model_path'])
        self.brain = policy_manager.get_policy(name, "play_net", position=position, **self._kwargs)
        if self._minor_net == 'rules':
            self.minor_brain = policy_manager.get_policy("minor_rule", "minor_rule")
        elif self._minor_net == 'net':
            minor_net_name = "minor_net_{}".format(self._kwargs['model_path'])
            self.minor_brain = policy_manager.get_policy(minor_net_name, "minor_net", **self._kwargs)
        else:
            raise RuntimeError("no minor policy is set.")
        print("agent [{}] net set temperature:{}".format(self.position, self._temperature))

    def get_action(self):
        state = self.env.get_state(self.position)
        # s.print()
        s = state.get_policy_input()
        a = self.brain.evaluate(self, s, self._temperature)
        action_type, action_index = decode_action(a)
        move = move_manager.get_move_by_id(a)
        cards = move.get_action_cards()
        minor_card_count = get_request_minor_card_count(
            action_type, action_index)
        if minor_card_count:
            for card_count in minor_card_count:
                minor_cards = self.get_minor_cards(state, card_count, cards)
                cards.extend(minor_cards)
        return self.get_card_from_handcards(cards)


    def get_minor_cards(self, state, card_count, major_cards):
        return self.minor_brain.evaluate(state, card_count, major_cards)
