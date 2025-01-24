import random
from random import choice

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from env.env_util import Game, Player
from util import MoveType


class DouDizhu:
    def __init__(self, agents, rl):
        self.game = Game(agents, rl)
        self.bomb_num = 0

    def reset(self, landlord_id):
        self.game.game_start(landlord_id)
        return self.get_next_state(landlord_id)

    def get_next_state(self, cur_player_idx):
        cur_player: Player = self.game.players[cur_player_idx]
        cur_hands = cur_player.hand_cards[:]
        last_move_type = self.game.last_move_type
        last_move = self.game.last_move

        up_player_idx, down_player_idx = self.get_up_down_player_idx(
            cur_player_idx)
        up_player_cards = self.game.players[up_player_idx].hand_cards
        down_player_cards = self.game.players[down_player_idx].hand_cards

        up_player_cards_num = len(up_player_cards)
        down_player_cards_num = len(down_player_cards)

        landlord_id = self.game.landlord_id

        res = {
            'cur_hands': cur_hands,
            'last_move_type': last_move_type,
            'last_move': last_move,
            'up_player_cards_num': up_player_cards_num,
            'down_player_cards_num': down_player_cards_num,
            'landlord_id': landlord_id,
            'bomb_num': self.bomb_num,
            'desk_record': self.game.play_records.desk_record[:],
            'cur_player_idx': cur_player_idx,
        }

        return res

    def step(self, cur_player: int, action):
        cur_game_player: Player = self.game.players[cur_player]
        cur_move_type, cur_move, self.game.game_end, self.yaobuqi = cur_game_player.play(
            self.game.last_move_type, self.game.last_move,
            self.game.play_records, action)

        if cur_move_type == MoveType.bomb:
            self.bomb_num += 1

        self.game.last_move_type, self.game.last_move = cur_move_type, cur_move
        if self.yaobuqi:
            self.game.yaobuqis.append(cur_player)
        else:
            self.game.yaobuqis = []

        if len(self.game.yaobuqis) == 2:
            self.game.yaobuqis = []
            self.game.last_move_type = self.game.last_move = MoveType.start

        done = False
        if self.game.game_end:
            self.game.play_records.winner = cur_player
            done = True

        _, next_player = self.get_up_down_player_idx(cur_player)
        if next_player == 0:
            self.game.play_round += 1

        next_state = self.get_next_state(next_player)
        reward = self.get_reward(cur_player, done)
        return next_state, reward, done, next_player, cur_move_type, cur_move

    def get_reward(self, cur_player, done):
        reward = [0, 0, 0]
        if done:
            reward[cur_player] = 1
            return reward
        else:
            return reward

    def get_action(self, cur_player_id):
        next_moves_type, next_moves = self.game.get_next_moves(cur_player_id)
        return (next_moves_type, next_moves)

    def get_up_down_player_idx(self, cur_player_idx):
        up_player_idx = cur_player_idx - 1
        down_player_idx = cur_player_idx + 1

        if up_player_idx < 0:
            up_player_idx = 2

        if down_player_idx > 2:
            down_player_idx = 0

        return up_player_idx, down_player_idx