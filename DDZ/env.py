import random
from random import choice
from env_util import Game, Player
from util import MoveType


class DouDizhu:
    def __init__(self, agents, rl):
        self.game = Game(agents, rl)

    def reset(self, landlord_id):
        self.game.game_start(landlord_id)
        return self.get_next_state(landlord_id)

    def get_next_state(self, cur_player_idx):
        cur_player: Player = self.game.players[cur_player_idx]
        cur_hands = cur_player.hand_cards
        last_move_type = self.game.last_move_type
        last_move = self.game.last_move
        return (cur_hands, last_move_type, last_move, cur_player_idx)

    def step(self, cur_player: int, action):
        cur_game_player: Player = self.game.players[cur_player]
        cur_move_type, cur_move, self.game.game_end, self.yaobuqi = cur_game_player.play(
            self.game.last_move_type, self.game.last_move,
            self.game.play_records, action)

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

        next_player = cur_player + 1
        if next_player > 2:
            self.game.play_round += 1
            next_player = 0

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
