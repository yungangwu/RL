import numpy as np
from game.game_util import GamePieces

# 构建基础游戏环境，提供step，reset，等接口

class GameState:
    def __init__(self, board_size) -> None:
        # 初始化棋盘为一个board_size x board_size的二维数组，元素类型为np.int8，并全部置为0
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        # 设置棋盘大小
        self.board_size = board_size
        # 初始化移动历史记录为空列表
        self.move_history = []
        # 计算状态大小，即棋盘元素的总数
        self.state_size = board_size * board_size
        # 计算动作大小，即棋盘元素的总数
        self.action_size = board_size * board_size
        # 初始化状态数据为一个3x15x15的三维数组，元素全部置为0
        self.state_data = np.zeros((3, 15, 15))
        self._num_players = 2
        self.players = [GamePieces.WHITE, GamePieces.BLACK]
        self.players_values = [1, -1]

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.move_history = []
        self._round = 0
        chair_id = self.get_chair_id()
        return self.get_state(chair_id), chair_id

    def step(self, action):
        '''
        在给定位置放置当前落子方的棋子，并更新游戏状态。
        '''
        chair_id = self.get_chair_id()
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row][col] != 0:
            raise ValueError("非法落子位置！")

        self.board[row][col] = self.players_values[chair_id]
        self.move_history.append(action)
        self._round += 1

        obs = self.get_state(chair_id)
        reward = self.get_reward(chair_id)
        done = self.check_game_over()
        legal_actions = self.get_legal_actions()
        action_mask = self.action_mask()
        info = {
            'legal_actions': legal_actions,
            'action_mask': action_mask,
            }

        return obs, reward, done, info,

    def action_mask(self):
        mask = self.board != 0
        return mask

    def get_legal_actions(self):
        zero_mask = self.board == 0
        legal_moves = np.where(zero_mask, True, False).flatten()
        return legal_moves

    def get_reward(self, chair_id):
        if self.check_game_over():
            if self.check_game_winner() == self.players_values[chair_id]:
                return 1
            else:
                return -1
        return 0

    def check_game_over(self):
        if not np.all(self.board) and not self.check_game_winner():
            return False
        return True

    def check_game_winner(self):
        '''
        检查游戏是否结束，如果是则返回胜者，1或-1，否则返回None
        '''
        row, cols = np.where(self.board != 0)
        for r, c in zip(row, cols):
            if self.check_five_in_row(r, c):
                return self.board[r][c]
        return None

    def check_five_in_row(self, row, col):
        '''
        检查在给定的位置是否有五子连珠
        '''
        player = self.board[row][col]
        if player == 0:
            return False

        # 检查水平方向
        if col + 4 < 15 and np.all(self.board[row, col:col+5] == player):
            return True

        # 检查竖直方向
        if row + 4 < 15 and np.all(self.board[row:row+5, col] == player):
            return True

        # 检查正对角线方向
        if row + 4 < 15 and col + 4 < 15 and np.all(np.diag(self.board[row:row+5, col:col+5] == player)):
            return True

        # 检查反对角线方向
        if row  + 4 < 15 and col + 4 < 15 and np.all(np.diag(self.board[row:row+5, col-4:col+1] == player)):
            return True

        return False

    def get_state(self, chair_id):
        # state, 我方状态，敌方状态，空位置状态
        cur_state = np.zeros((self.board_size, self.board_size))
        op_state = np.zeros((self.board_size, self.board_size))
        blank_state = np.zeros((self.board_size, self.board_size))

        op_chair_id = self._get_next_chair_id(chair_id)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == chair_id:
                    cur_state[i][j] = 1
                elif self.board[i][j] == op_chair_id:
                    op_state[i][j] = 1
                else:
                    blank_state[i][j] = 1

        ret = {
            'cur_state': cur_state,
            'op_state': op_state,
            'blank_state': blank_state
        }
        return ret

    def get_chair_id(self):
        return self._round % self._num_players

    def _get_next_chair_id(self, chair_id):
        return (chair_id + 1) % self._num_players
