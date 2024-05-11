from base_player import Player

class DQNPlayer(Player):
    def __init__(self, strategy) -> None:
        self.strategy = strategy

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.strategy.get_move(board)
            return move
        else:
            print("WARNING: the board is full")