
from ddz.game.utils import *
import numpy as np

def get_candidate_minor_cards(handcards, minor_card_size, major_cards):
    handcard_values = [x[1] for x in handcards]
    value_counter = Counter(handcard_values)
    candidate_cards = [
        x for x in value_counter if value_counter[x] >= minor_card_size and x not in major_cards]
    return candidate_cards


class MinorPolicyRandom(object):

    def evaluate(self, agent, card_count, major_cards):
        candidate_minor_cards = get_candidate_minor_cards(agent.handcards, card_count, major_cards)
        if candidate_minor_cards:
            minor_card = np.random.choice(np.array(candidate_minor_cards))
            minor_card = [minor_card for _ in range(card_count)]
            if DEBUG:
                print("minor random policy evaluate action:" +
                      cards_value_to_str(minor_card))
        else:
            raise ValueError("no minor card found.")
        return minor_card
