import numpy as np

from strategy.mcts import MCTS, policy_value_fn
from config.config import *

class MCTS_Pure:

    def __init__(self) -> None:
        self.mcts = MCTS(policy_value_fn, c_puct)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self) -> str:
        return "MCTS {}".format(self.player)

class MCTSPlayer(MCTS_Pure):
    def __init__(self, policy_value_function, is_selfplay=0) -> None:
        super(MCTSPlayer, self).__init__()
        self.mcts = MCTS(policy_value_function, c_puct)
        self._is_selfplay = is_selfplay

    def get_action(self, env, return_prob=0):
        sensible_moves = env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board_width * board_height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temperature)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    acts,
                    p=noise_eps * probs + (1 - noise_eps) * np.random.dirichlet(
                        dirichlet_alpha * np.ones(len(probs))
                    )
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move

        else:
            print("WARNING: the board is full")