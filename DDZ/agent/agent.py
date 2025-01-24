from util.util import *


class Agent:
    def __init__(self) -> None:
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