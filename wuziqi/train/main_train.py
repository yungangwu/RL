import random
import os
import time
import numpy as np

from config.config import *
from collections import deque, defaultdict
from env.gomoku import GomokuEnv
from model.model_mcts import PolicyValueNet
from player.mcts_player import MCTSPlayer, MCTS_Pure
from util.buffer import ReplayBuffer

class TrainPipeline:
    def __init__(self) -> None:
        self.env = GomokuEnv()

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.play_batch_size = 1
        self.best_win_ratio = 0.0

        self.policy_value_net = PolicyValueNet(model_file=restore_model)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, is_selfplay=1)
        self.mcts_infer = mcts_infer
        self.lr_multiplier = lr_multiplier

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state]) # np.rot90用于将数组逆时针旋转90度
                equi_mcts_prob = np.rot90(np.flipud( # np.flipud用于将数组沿着水平方向翻转
                    mcts_prob.reshape(board_height, board_width)
                ), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))

        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.env.start_self_play(self.mcts_player)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # agument the data
            play_data = self.get_equi_data(play_data)
            self.replay_buffer.push(play_data)

    def policy_update(self):
        state_batch, action_batch, winner_batch = self.replay_buffer.sample(train_batch_size)
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(update_epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                action_batch,
                winner_batch,
                learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)
            ), axis=1))
            if kl > kl_coeff * 4:
                break

        # adaptively adjust the learning rate
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
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.mcts_infer,
                                                                  win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        '''run the training pipeline'''
        win_num = 0
        try:
            for i_step in range(game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i_step + 1, self.episode_len))
                if len(self.replay_buffer) > train_batch_size:
                    loss, entropy = self.policy_update()

                if (i_step + 1) % checkpoint_freq == 0:
                    print("current self-play batch: {}".format(i_step + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.path.join(model_path, "newest_model.pt"))
                    if win_ratio > self.best_win_ratio:
                        win_num += 1
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(os.path.join(model_path, "best_model.pt"))
                        if self.best_win_ratio == 1.0 and self.mcts_infer < 5000:
                            self.mcts_infer += 1000
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            print('\n\rquit')

        return win_num

if __name__ == "__main__":
    start_t = time.time()
    training_pipeline = TrainPipeline()
    training_pipeline.run()
    print("time cost is {}".format(time.time()-start_t))