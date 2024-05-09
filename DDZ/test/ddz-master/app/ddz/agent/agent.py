

from ddz.game.utils import *
from ddz.policy.policy_registry import make
from ddz.game.move import move_manager, get_legal_actions


class Agent(object):
    def __init__(self):
        super().__init__()

    def set_position(self, position):
        self.position = position

    def set_env(self, env):
        self.env = env

    def reset(self, cards):
        self.initial_cards = cards.copy()
        self.handcards = cards.copy()
        self.actions = []
        if DEBUG:
            print("agent [{}]'s initial card:{}".format(
                self.position, cards_to_str(cards)))


    def accept_bonus_cards(self, cards):
        self.handcards.extend(cards)
        self.initial_cards.extend(cards)
        if DEBUG:
            print("agent [{}] accept bonus cards:{}".format(
                self.position, cards_to_str(self.handcards)))

    def get_action(self):
        pass

    def get_minor_cards(self, state, card_count, major_cards):
        pass

    def get_card_from_handcards(self, cards):
        handout_cards = []
        card_size = len(cards)
        cards_copy = cards.copy()
        handcards = self.handcards.copy()
        while cards_copy:
            card_value = cards_copy.pop()
            for card in handcards:
                if card[1] == card_value:
                    handout_cards.append(card)
                    handcards.remove(card)
                    break
        if DEBUG and len(handout_cards) != card_size:
            raise RuntimeError("can not play cards, handcards:{}, handout cards:{}, required card num is {}, but found only {}".format(
                               cards_to_str(self.handcards), cards_value_to_str(cards), card_size, len(handout_cards)))
        handout_cards.sort(key=lambda k: k[1])
        return handout_cards

    def handout_cards(self, cards):
        for card in cards:
            self.handcards.remove(card)
        handout_cards = cards.copy()
        self.actions.append(handout_cards)
        return handout_cards

    def get_history_actions(self):
        return self.actions.copy()

    def get_last_action(self):
        if self.actions:
            return self.actions[-1].copy()
        return []

    def end(self):
        return not self.handcards

    def get_legal_actions(self):
        follow_cards = self.env.get_follow_cards(self.position)
        if follow_cards:
            follow_action_id = move_manager.get_move_by_cards(get_card_values(follow_cards))
        else:
            follow_action_id = None
        return get_legal_actions(self.handcards, follow_action_id)
