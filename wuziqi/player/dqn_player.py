import numpy as np

from player.base_player import Player
from env.gomoku import GomokuEnv
from model.model_dqn import DQNPolicyValue

class DQNPlayer(Player):
    def __init__(self, strategy, is_selfplay) -> None:
        self.strategy: DQNPolicyValue = strategy
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self, strategy):
        self.strategy = strategy

    def get_action(self, board: GomokuEnv):
        sensible_moves = board.availables
        state = board.current_state()
        if len(sensible_moves) > 0:
            if self._is_selfplay:
                move = self.strategy.get_move(sensible_moves, state)
            else:
                move = np.random.choice(sensible_moves)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self, ) -> str:
        return "DQN {}".format(self.player)