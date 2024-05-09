from ddz.agent.agent import Agent
from ddz.game.utils import *
from ddz.policy.policy_manager import policy_manager
from ddz.game.move import move_manager

class AgentRandom(Agent):
    def __init__(self):
        super().__init__()
        self.brain = policy_manager.get_policy("random", "random")
        self.minor_brain = policy_manager.get_policy("minor_rule", "minor_rule")

    def get_action(self):
        action = self.brain.evaluate(self)
        action_type, action_index = decode_action(action)
        if action_type != Action_Pass:
            move = move_manager.get_move_by_id(action)
            cards = move.get_action_cards()
            minor_card_count = get_request_minor_card_count(
                action_type, action_index)
            if minor_card_count:
                for card_count in minor_card_count:
                    minor_cards = self.minor_brain.evaluate(self, card_count, cards)
                    cards.extend(minor_cards)
            return self.get_card_from_handcards(cards)
        return []