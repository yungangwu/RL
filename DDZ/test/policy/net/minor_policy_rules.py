
from ddz.policy.minor_cards_rules import *
import numpy as np



class MinorPolicyRules(object):

    def evaluate(self, agent, card_count, major_cards):
        minor_card = get_minor_cards_by_rules(agent.handcards, card_count, major_cards)
        if minor_card:
            minor_card = [minor_card for _ in range(card_count)]
            if DEBUG:
                print("minor rule policy evaluate action:" +
                      cards_value_to_str(minor_card))
        else:
            raise ValueError("no minor card found.")
        return minor_card
