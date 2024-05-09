import numpy as np

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from util import MoveType


def choose_cards(next_moves_type, next_moves, last_move_type, last_move,
                 hand_cards, mode, RL, agent, game, player_id, action):
    if mode == 'random':
        return choose_random(next_moves_type, next_moves, last_move_type)
    elif mode == 'rl':
        pass


def choose_random(next_moves_type, next_moves, last_move_type):
    if len(next_moves) == 0:
        return MoveType.yaobuqi, []
    else:
        if last_move_type == MoveType.start:
            r_max = len(next_moves)
        else:
            r_max = len(next_moves) + 1
        r = np.random.randint(0, r_max)
        if r == len(next_moves):
            return MoveType.yaobuqi, []

    return next_moves_type[r], next_moves[r]
