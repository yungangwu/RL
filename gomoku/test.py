from config import *
from env import GomokuEnv
from model import PolicyValueNet
from mcts import MCTSPlayer

class CurPlayer:
    player_id = 0

class Game(object):
    def __init__(self, board):
        self.board = board
        self.cell_size = board_width - 1
        self.chess_size = 50 * self.cell_size

        self.whitex = []
        self.whitey = []
        self.blackx = []
        self.blacky = []

        self.color = "#e4ce9f"
        self.colors = [[self.color] * self.cell_size for _ in range(self.cell_size)]

    def graphic(self, board, player1, player2):
        plt_fig, ax = plt.subplots(facecolor=self.color,
                                   colWidths=[1 / board_width] * self.cell_size,
                                   loc='center'
                                   )
        ax.set_facecolor(self.color)

        # 制作棋盘
        mytable = plt.table(cellColours=self.colors,
                            colWidths=[1 / board_width] * self.cell_size,
                            loc='center'
                            )
        ax.set_aspect('equal')

        cell_height = 1 / board_width
        for pos, cell in mytable.get_celld().item():
            cell.set_height(cell_height)

        mytable.auto_set_font_size(False)
        mytable.set_fontsize(self.cell_size)
        ax.set_xlim([1, board_width * 2 + 1])
        ax.set_ylim([board_height * 2 + 1, 1])
        plt.title("Gomoku")

        plt.axis('off')
        cur_player = CurPlayer()

        while True:
            try:
                if cur_player.player_id == 1:
                    move = player1.get_action(self.board)
                    self.board.step(move)
                    x, y = self.board.move_to_location(move)
                    plt.scatter((y + 1) * 2, (x + 1) * 2, s=self.chess_size, c='white')
                    cur_player.player_id = 0
                elif cur_player.player_id == 0:
                    move = player2.get_action(self.board)
                    self.board.step(move)
                    x, y = self.board.move_to_location(move)
                    plt.scatter((y + 1) * 2, (x + 1) * 2, s=self.chess_size, c='black')
                    cur_player.player_id = 1

                end, winner = self.board.game_end()
                if end:
                    if winner != -1:
                        ax.text(x=board_width, y=(board_height + 1)  * 2 + 0.1,
                                s="Game end. Winner is player {}".format(cur_player.player_id), fontsize=10,
                                color='red', weight='bold',
                                horizontalalignment='center')
                    else:
                        ax.text(x=board_width, y=(board_height + 1) * 2 + 0.1,
                                s="Game end. Tie Round".format(cur_player.player_id), fontsize=10, color='red',
                                weight='bold',
                                horizontalalignment='center')

                    return winner
                display.display(plt.gcf())
                display.clear_output(wait=True)
            except:
                pass

    def start_play(self, player1, player2, start_player=0):
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')

        self.board.reset()
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)

        self.graphic(self.board, player1, player2)

if __name__ == '__main__':
    board = GomokuEnv()
    game = Game(board)
    best_policy = PolicyValueNet(model_file="best_model.pt")
    mcts_player = MCTSPlayer(best_policy.policy_value_fn)
    game.start_play(mcts_player, mcts_player, start_player=0)
