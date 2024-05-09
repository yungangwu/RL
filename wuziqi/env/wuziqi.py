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
        self.move_history.append((row, col))
        winner = self.check_game_winner(player.value, row, col)
        done = self.check_game_over(winner)
        reward = self.get_reward(player.value, row, col)
        return self.get_state(), reward, done, winner

    def get_legal_moves(self):
        zero_mask = self.board == 0
        legal_moves = np.where(zero_mask, True, False).flatten()
        return legal_moves

    def get_reward(self, player, row, col):
        if self.check_num_in_board(player, row, col, 5):
            return 5

        # 检查四子连珠
        if self.check_num_in_board(player, row, col, 4):
            return 2

        # 检查三子连珠
        if self.check_num_in_board(player, row, col, 3):
            return 1

        return 0

    def check_game_over(self, winner):
        if not np.all(self.board) and (winner is None):
            return 0
        return 1

    def check_game_winner(self, player, row, col):
        '''
        检查游戏是否结束，如果是则返回胜者，1或-1，否则返回None
        '''
        if self.check_num_in_board(player, row, col, 5):
            return self.board[row][col]
        return None

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

    def count_consecutive_elements(self, subarray, player):
        consecutive_count = 0
        max_consecutive_count = 0

        for element in subarray:
            if element == player:
                consecutive_count += 1
                max_consecutive_count = max(max_consecutive_count, consecutive_count)
            else:
                consecutive_count = 0

        return max_consecutive_count

    def check_num_in_board(self, player, row, col, num):
        # 获取棋盘的行数和列数
        rows, cols = self.board.shape

        # 检查行
        row_start = max(0, col - num + 1)
        row_end = min(cols, col + num)
        row_player_num = self.count_consecutive_elements(self.board[row, row_start:row_end], player)
        if row_player_num >= num:
            return True

        # 检查列
        col_start = max(0, row - num + 1)
        col_end = min(rows, row + num)
        col_player_num = self.count_consecutive_elements(self.board[col_start:col_end, col], player)
        if col_player_num >= num:
            return True

        # 检查主对角线
        diag_start_row = row
        diag_start_col = col
        while diag_start_row > 0 and diag_start_col > 0:
            diag_start_row -= 1
            diag_start_col -= 1

        diag_end_row = row
        diag_end_col = col
        while diag_end_row < rows and diag_end_col < cols:
            diag_end_row += 1
            diag_end_col += 1

        diag = np.diagonal(self.board[diag_start_row:diag_end_row, diag_start_col:diag_end_col])
        diag_player_num = self.count_consecutive_elements(diag, player)
        if diag_player_num >= num:
            return True

        # 检查副对角线
        rev_board = np.fliplr(self.board)
        rev_row = row
        rev_col = cols - 1 - col

        rev_diag_start_row = rev_row
        rev_diag_start_col = rev_col
        while rev_diag_start_row > 0 and rev_diag_start_col > 0:
            rev_diag_start_row -= 1
            rev_diag_start_col -= 1

        rev_diag_end_row = rev_row
        rev_diag_end_col = rev_col
        while rev_diag_end_row < rows and rev_diag_end_col < cols:
            rev_diag_end_row += 1
            rev_diag_end_col += 1

        rev_diag = np.diagonal(rev_board[rev_diag_start_row:rev_diag_end_row, rev_diag_start_col:rev_diag_end_col])
        rev_diag_player_num = self.count_consecutive_elements(rev_diag, player)
        if rev_diag_player_num >= num:
            return True

        return False
