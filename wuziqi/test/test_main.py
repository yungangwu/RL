from gomoku import Gomoku
from dqn import DQN
from self_play import train_dqn

gomoku = Gomoku()
dqn = DQN(gomoku.board_size)
dqn.to('cuda')
target_dqn = DQN(gomoku.board_size)
target_dqn.to('cuda')

train_dqn(gomoku, dqn, target_dqn)