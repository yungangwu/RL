from config import *

# 状态空间为[4, 棋盘宽, 棋盘高]，四个维度分别是当前视角下的位置，对手位置，上次位置以及轮次
class GomokuEnv(gym.Env):
    def __init__(self, start_player=0) -> None:
        self.start_player = start_player

        self.action_space = Discrete((board_width * board_height))
        self.observation_space = Box(0, 1, shape=(4, board_width, board_height))
        self.reward = 0
        self.info = {}
        self.players = [1, 2]

    def step(self, action):
        self.states[action] = self.current_player
        if action in self.availables:
            self.availables.remove(action)

        self.last_move = action

        done, winner = self.game_end()
        reward = 0
        if done:
            if winner == self.current_player:
                reward = 1
            else:
                reward = -1

        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

        # update state
        obs = self.current_state()
        return obs, reward, done, self.info

    def reset(self, ):
        if board_width < n_in_row or board_height < n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(n_in_row))

        self.current_player = self.players[self.start_player]
        self.availables = list(range(board_width * board_height))
        self.states = {}
        self.last_move = -1

        return self.current_state()

    def render(self, mode='human', start_player=0):
        width = board_width
        height = board_height

        p1, p2 = self.players

        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height-1, -1, -1):
            print("{0:4}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = self.states.get(loc, -1)
                if p == p1:
                    print('B'.center(8), end='')
                elif p == p2:
                    print('W'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def has_a_winner(self,):
        states = self.states
        moved = list(set(range(board_width * board_height)) - set(self.availables))
        if len(moved) < n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // board_width
            w = m % board_width
            player = states[m]

            if (w in range(board_width - n_in_row + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n_in_row))) == 1):
                return True, player

            if (h in range(board_height - n_in_row + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n_in_row * board_width, board_width))) == 1):
                return True, player

            if (w in range(board_width - n_in_row + 1) and h in range(board_height - n_in_row + 1) and
                    len(set(
                        states.get(i, -1) for i in range(m, m + n_in_row * (board_width + 1), board_width + 1))) == 1):
                return True, player

            if (w in range(n_in_row - 1, board_width) and h in range(board_height - n_in_row + 1) and
                    len(set(
                        states.get(i, -1) for i in range(m, m + n_in_row * (board_width - 1), board_width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        win, winner = self.has_a_winner()
        if win:
            print("winner is player{}".format(winner))
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def current_state(self):
        square_state = np.zeros((4, board_width, board_height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // board_width, # 当前玩家的棋子位置
                            move_curr % board_height] = 1.0
            square_state[1][move_oppo // board_width, # 对手的棋子位置
                            move_oppo % board_height] = 1.0
            square_state[2][self.last_move // board_width, # 上一步落子的位置
                            self.last_move % board_height] = 1.0

        if len(self.states) % 2 == 0: # 表示当前玩家的颜色，偶数步当前的玩家是先手玩家，矩阵就全为1，奇数就代表后手玩家，矩阵就全为0
            square_state[3][:, :] = 1.0

        return square_state[:, ::-1, :] # 宽度上的反转，gpt说是为了正确表达棋盘的状态

    def start_play(self, player1, player2, start_player=0):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')

        self.reset()
        p1, p2 = self.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        while True:
            player_in_turn = players[self.current_player]
            move = player_in_turn.get_action(self)
            self.step(move)
            end, winner = self.game_end()
            if end:
                return winner

    def start_self_play(self, player):
        self.reset()
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self, return_prob=1)
            states.append(self.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.current_player)
            self.step(move)
            end, winner = self.game_end()
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                player.reset_player()
                return winner, zip(states, mcts_probs, winners_z) # zip函数的作用可以将多个list中的元素，按照位置一一对应的打包成元组

    def location_to_move(self, location):
        if (len(location) != 2):
            return -1

        h = location[0]
        w = location[1]
        move = h * board_width + w
        if (move not in range(board_width * board_width)):
            return -1
        return move

    def move_to_location(self, move):
        h = move // board_width
        w = move % board_width
        return [h, w]