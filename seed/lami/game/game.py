import collections
import random
import numpy as np
import functools
from lami.game import Player
from lami.types import CARD_RANK, CARD_SUIT, CARD_SCORE, GUI_RANK, MAX_DESK, MAX_TAIL
from lami.utils import card_to_matrix, cards_to_str
from lami.game.move import Move


num_rank = len(CARD_RANK)
num_suit = len(CARD_SUIT)
num_set = 2

init_card_per_player = 20
max_desk_card_seq = MAX_DESK

