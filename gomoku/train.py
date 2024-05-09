import numpy as np
import random

from collections import defaultdict
from collections import deque
from env import GomokuEnv
from model import PolicyValueNet
from mcts import MCTSPlayer, MCTS_Pure
from config import *

# 自博弈是什么？谁和谁对打？
# 此处的自博弈双方分别是基于MCTS的神经网络和纯mcts
# mcts怎么做自博弈？

class TrainPipeline:
    def __init__(self):
        self.env = GomokuEnv()

        self.data_buffer = deque()
        self.play_batch_size = 1
        self.best_win_ratio = 0.0

        self.policy_value_net = PolicyValueNet(model_file=restore_model)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      is_selfplay=1)
        self.mcts_infer = mcts_infer
        self.lr_multiplier = lr_multiplier

    def get_equi_data(self, play_data): # 通过旋转和翻转的方式来扩充数据集
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state]) # np.rot90()是一个逆时针90度旋转函数，s是一个二维数组，i表示旋转的次数，
                equi_mcts_prob = np.rot90(np.flipud( # np.flipud()是一个水平翻转函数，将矩阵的行倒序排列
                    mcts_prob.reshape(board_height, board_width)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state]) # np.fliplr()是一个垂直方向翻转
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner)) # 为啥要水平和垂直翻转

        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.env.start_self_play(self.mcts_player) # 开启自博弈，使用mcts player开启，采用mcts来收集数据，训练网络学习mcts的出牌
            play_data = list(play_data)[:] # 对所有的数据复制
            self.episode_len = len(play_data)

            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_updata(self):
        mini_batch = random.sample(self.data_buffer, train_batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(update_epochs):  # 每次更新update_epochs次网络
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > kl_coeff * 4:
                break

            if kl > kl_coeff * 2 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
            elif kl < kl_coeff / 2 and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn)
        pure_mcts_player = MCTS_Pure()
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.env.start_play(current_mcts_player,
                                         pure_mcts_player,
                                         start_player=i % 2)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts: {}, win: {}, loss: {}, tie: {}".format(self.mcts_infer, win_cnt[1], win_cnt[2], win_cnt[-1]))

        return win_ratio

    def run(self):
        win_num = 0
        try:
            for i_step in range(game_batch_num):
                self.collect_selfplay_data(self.play_batch_size) # 收集自博弈数据
                print("batch i: {}, episode_len: {}".format(i_step + 1, self.episode_len))
                if len(self.data_buffer) > train_batch_size: # 数据的收集数量已经大于可以训练的batch size，可以训练了
                    loss, entropy = self.policy_updata()

                if (i_step + 1) % checkpoint_freq == 0: # 每20轮，保留一次
                    print("current self-play batch: {}".format(i_step + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.path.join(model_path, "newest_model.pt"))
                    if win_ratio > self.best_win_ratio: # 训练出来的比最好的模型，赢得次数还多，就保存更新一下
                        win_num += 1
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(os.path.join(model_path, "best_model.pt"))
                        if self.best_win_ratio == 1.0 and self.mcts_infer < 5000:
                            self.mcts_infer += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\requit')

        return win_num