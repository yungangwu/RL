import numpy as np


class GameState:
    def __init__(self, board_size) -> None:
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.board_size = board_size
        self.move_history = []
        self.state_size = board_size * board_size
        self.action_size = board_size * board_size
        self.state_data = np.zeros((3, board_size, board_size))

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size),
                              dtype=np.int8)
        self.move_history = []
        return self.get_state()

    def step(self, action, player):
        '''
        在给定位置放置当前落子方的棋子，并更新游戏状态。
        '''
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row][col] != 0:
            raise ValueError("非法落子位置！")

        self.board[row][col] = player.value
        self.move_history.append(action)
        winner = self.check_game_winner()
        done = self.check_game_over()
        reward = self.get_reward(player, row, col)
        return self.get_state(), reward, done, winner

    def get_legal_moves(self):
        zero_mask = self.board == 0
        legal_moves = np.where(zero_mask, True, False).flatten()
        return legal_moves

    def get_reward(self, player, row, col):
        if self.check_game_over():
            if self.check_game_winner() == player.value:
                return 100

        # 检查四子连珠
        if self.check_num_in_a_row(player, row, col, 4):
            return 10

        # 检查三子连珠
        if self.check_num_in_a_row(player, row, col, 3):
            return 1
        return 0

    def check_game_over(self):
        if not np.all(self.board) and not self.check_game_winner():
            return 0
        return 1

    def check_game_winner(self):
        '''
        检查游戏是否结束，如果是则返回胜者，1或-1，否则返回None
        '''
        row, cols = np.where(self.board != 0)
        for r, c in zip(row, cols):
            if self.check_five_in_board(r, c):
                return self.board[r][c]
        return None

    def check_five_in_board(self, row, col):
        '''
        检查在给定的位置是否有五子连珠
        '''
        player = self.board[row][col]
        if player == 0:
            return False

        # 检查水平方向
        if col + 4 < self.board_size and np.all(self.board[row, col:col +
                                                           5] == player):
            return True

        # 检查竖直方向
        if row + 4 < self.board_size and np.all(self.board[row:row + 5,
                                                           col] == player):
            return True

        # 检查正对角线方向
        if row + 4 < self.board_size and col + 4 < self.board_size and np.all(
                np.diag(self.board[row:row + 5, col:col + 5] == player)):
            return True

        # 检查反对角线方向
        if row + 4 < self.board_size and col + 4 < self.board_size and np.all(
                np.diag(self.board[row:row + 5, col - 4:col + 1] == player)):
            return True

        return False

    # def get_available_actions(self, state):
    #     empty_cells = np.where(state.board == 0)
    #     return empty_cells

    def get_state(self):
        board_size = self.board_size
        board = self.board
        state_data = self.state_data
        for i in range(board_size):
            for j in range(board_size):
                if board[i][j] == 1:
                    # 白子
                    state_data[0][i][j] = 255
                    state_data[1][i][j] = 255
                    state_data[2][i][j] = 255
                elif board[i][j] == -1:
                    # 黑子
                    state_data[0][i][j] = 0
                    state_data[1][i][j] = 0
                    state_data[2][i][j] = 0
                else:
                    # 空位
                    state_data[0][i][j] = 128
                    state_data[1][i][j] = 128
                    state_data[2][i][j] = 128
        return state_data

    def check_three_in_a_row(self, player, row, col):
        """判断在 (row, col) 位置下了 player 的棋子后，是否连成了三个"""
        # 判断横向是否连成了三个
        if col + 2 < self.board.shape[1] and np.all(self.board[row, col:col +
                                                               3] == player):
            return True

        if col - 2 > 0 and np.all(self.board[row, col - 2:col + 1] == player):
            return True

        # 判断纵向是否连成了三个
        if row + 2 < self.board.shape[0] and np.all(self.board[row:row + 3,
                                                               col] == player):
            return True

        if row - 2 > 0 and np.all(self.board[row - 2:row + 1, col] == player):
            return True

        # 判断左上到右下斜向是否连成了三个
        if row + 2 < self.board.shape[0] and col + 2 < self.board.shape[
                1] and np.all(
                    np.diag(self.board[row:row + 3, col:col + 3] == player)):
            return True

        if row - 2 > 0 and col - 2 > 0 and np.all(
                np.diag(self.board[row - 2:row + 1,
                                   col - 2:col + 1] == player)):
            return True
        # 判断右上到左下斜向是否连成了三个
        if row + 2 < self.board.shape[0] and col - 2 >= 0 and np.all(
                np.diag(self.board[row:row + 3, col - 2:col + 1] == player)):
            return True

        # 左下到右上
        if row - 2 > 0 and col + 2 >= self.board.shape[1] and np.all(
                np.diag(self.board[row - 2:row + 1, col:col + 3] == player)):
            return True
        return False

    def check_num_in_a_row(self, player, row, col, num):
        """判断在 (row, col) 位置下了 player 的棋子后，是否连成了三个"""
        temp_num = num - 1
        # 判断横向是否连成了三个
        if col + temp_num < self.board.shape[1] and np.all(
                self.board[row, col:col + num] == player):
            return True

        if col - temp_num > 0 and np.all(self.board[row, col - temp_num:col +
                                                    1] == player):
            return True

        # 判断纵向是否连成了三个
        if row + temp_num < self.board.shape[0] and np.all(
                self.board[row:row + num, col] == player):
            return True

        if row - temp_num > 0 and np.all(self.board[row - temp_num:row + 1,
                                                    col] == player):
            return True

        # 判断左上到右下斜向是否连成了三个
        if row + temp_num < self.board.shape[0] and col + temp_num < self.board.shape[1] and np.all(np.diag(self.board[row:row + num, col:col + num] == player)):
            return True

        if row - temp_num > 0 and col - temp_num > 0 and np.all(
                np.diag(self.board[row - temp_num:row + 1,
                                   col - temp_num:col + 1] == player)):
            return True
        # 判断右上到左下斜向是否连成了三个
        if row + temp_num < self.board.shape[
                0] and col - temp_num >= 0 and np.all(
                    np.diag(self.board[row:row + num,
                                       col - temp_num:col + 1] == player)):
            return True

        # 左下到右上
        if row - temp_num > 0 and col + temp_num >= self.board.shape[
                1] and np.all(
                    np.diag(self.board[row - temp_num:row + 1,
                                       col:col + num] == player)):
            return True
        return False