
import enum

CARD_FEATURES_SIZE = 112

CARD_RANK = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', '🃏']

CARD_RANK_TO_INDEX = {k: i for i, k in enumerate(CARD_RANK)}

CARD_SUIT = ['♠️', '♥️', '♣︎', '♦️']

CARD_SUIT_TO_INDEX = {k: i for i, k in enumerate(CARD_SUIT)}

GUI_RANK = CARD_RANK_TO_INDEX['🃏']
A_RANK = CARD_RANK_TO_INDEX['A']

CARD_SCORE = [15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0]

MAX_DESK = 24

MAX_TAIL = CARD_RANK_TO_INDEX['K'] + 1
# print(f'max tail:{MAX_TAIL}')
