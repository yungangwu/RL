import numpy as np

class DQNStrategy:
    def __init__(self, epsilon, dqn, target_dqn) -> None:
        self.epsilon = epsilon

        self.dqn = dqn
        self.target_dqn = target_dqn

    def get_move(self, board):
        sensible_moves = board.availables
        q_values = self.dqn(board).squeeze().detach().numpy()
        q_values[~sensible_moves] = -np.inf
        if np.random.uniform() < self.epsilon:
            move = np.argmax(q_values)
        else:
            move = np.random.choice(np.flatnonzero(sensible_moves))

        return move
