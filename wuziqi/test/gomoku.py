import numpy as np
from config import *

class Gomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, move):
        # logger.info(f"""{type(move)},{move}""")
        row_index, col_index = move
        if self.board[row_index, col_index] == 0:
            self.board[row_index, col_index] = self.current_player
            if self.check_winner(move):
                return self.current_player
            self.current_player = 3 - self.current_player
            return 0
        return -1

    def check_winner(self, move):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1)]
        for dx, dy in directions:
            count = 1
            for i in range(1, 5):
                x, y = move[0] + i * dx, move[1] + i * dy
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == self.current_player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                x, y = move[0] - i * dx, move[1] - i * dy
                if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == self.current_player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False
