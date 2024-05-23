import random
import os
import time
import numpy as np

from config.config import *
from collections import deque, defaultdict
from env.gomoku import GomokuEnv
from model.model_mcts import PolicyValueNet
from model.model_dqn import DQNPolicyValue
from player.mcts_player import MCTSPlayer, MCTS_Pure
from player.dqn_player import DQNPlayer
from util.buffer import ReplayBuffer

class TrainPipeline:
    def __init__(self, ) -> None:
        self.env = GomokuEnv()

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.play_batch_size = 1
        self.best_win_ratio = 0.0

        self.policy_value_net = DQNPolicyValue(model_file='')
        self.dqn_player = DQNPlayer(strategy=self.policy_value_net, is_selfplay=1)

        # self.policy_value_net = PolicyValueNet(model_file=restore_model)
        # self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, is_selfplay=1)
        self.lr_multiplier = lr_multiplier

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, act, winner_z, state_), ..., ...]
        """
        extend_data = []
        for state, act, winner, state_ in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state]) # np.rot90用于将数组逆时针旋转90度
                equi_acts = np.rot90(np.flipud( # np.flipud用于将数组沿着水平方向翻转
                    act.reshape(board_height, board_width)
                ), i)
                equi_state_ = np.array([np.rot90(s, i) for s in state_])
                extend_data.append((equi_state,
                                    np.flipud(equi_acts).flatten(),
                                    winner, equi_state_))
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_acts = np.fliplr(equi_acts)
                equi_state_ = np.array([np.fliplr(s) for s in equi_state_])
                extend_data.append((equi_state,
                                    np.flipud(equi_acts).flatten(),
                                    winner, equi_state_))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        for i in range(n_games):
            winner, play_data = self.env.start_self_play(self.dqn_player)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # agument the data
            play_data = self.get_equi_data(play_data)
            self.replay_buffer.push(play_data)

    def policy_update(self):
        state_batch, action_batch, winner_batch, state_batch_ = self.replay_buffer.sample(train_batch_size)
        for i in range(update_epochs):
            loss = self.policy_value_net.train_step(
                state_batch,
                action_batch,
                winner_batch,
                state_batch_,
                learn_rate * self.lr_multiplier
            )

        return loss

    def policy_evaluate(self, n_games=10, old_model_path='', new_model_path=''):
        if not old_model_path:
            old_model_path = os.path.join(model_path, "best_model.pt") if os.path.join(model_path, "best_model.pt") else ''

        old_strategy = DQNPolicyValue(model_file=old_model_path)
        old_dqn_player = DQNPlayer(strategy=old_strategy, is_selfplay=1)

        if not new_model_path:
            new_model_path = os.path.join(model_path, "newest_model.pt") if os.path.join(model_path, "newest_model.pt") else ''

        new_strategy = DQNPolicyValue(model_file=new_model_path)
        current_dqn_player = DQNPlayer(strategy=new_strategy, is_selfplay=1)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.env.start_play(old_dqn_player,
                                         current_dqn_player,
                                         start_player=i % 2)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("win: {}, lose: {}, tie:{}".format(win_cnt[1], win_cnt[2], win_cnt[-1]))
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
                    loss = self.policy_update()

                if (i_step + 1) % checkpoint_freq == 0:
                    print("current self-play batch: {}".format(i_step + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.path.join(model_path, "newest_model.pt"))
                    self.policy_value_net.save_model(os.path.join(model_path, f"model_{i_step}.pt"))
                    if win_ratio > self.best_win_ratio:
                        win_num += 1
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model(os.path.join(model_path, "best_model.pt"))
                        if self.best_win_ratio == 1.0:
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            print('\n\rquit')

        return win_num

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    set_seed(42)
    start_t = time.time()
    training_pipeline = TrainPipeline()
    # training_pipeline.run()
    # print("time cost is {}".format(time.time()-start_t))
    training_pipeline.policy_evaluate(old_model_path='./path_to_model/best_model.pt', new_model_path='./path_to_model/model_19.pt')