
from utils import *
    

def get_minor_cards_by_rules(handcards, minor_card_size, major_cards):
    handcard_values = [x[1] for x in handcards]
    value_counter = Counter(handcard_values)
    candidate_cards = {
        k:v for k, v in value_counter.items() if v >= minor_card_size and k not in major_cards}
    if candidate_cards:
        candidate_cards = sorted(candidate_cards.items(), key = lambda kv:(kv[1], kv[0]))
        return candidate_cards[0][0]
    else:
        return None
    
