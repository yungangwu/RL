import random
import numpy as np
import wandb
import os
import torch
import sys
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from agent import DQNAgent, RandomAgent
from env.wuziqi import GameState
from util.common import Player


class SelfPlay:
    def __init__(self, env, agent1, agent2):
        self.env = env
        self.agent1 = agent1  # 黑棋AI玩家
        self.agent2 = agent2  # 白棋AI玩家

    def play_one_game(self, log=False):
        state = self.env.get_initial_state()
        players = [self.agent1, self.agent2]
        np.random.shuffle(players)

        while True:
            action = players[0].act(state)
            winner = state.update_state(action)
            if winner is not None:
                if log:
                    print('Game over, Winner is ',
                          'black' if winner == Player.BLACK.value else 'white')
                return winner

            players.reverse()


class ReinforcementLearning:
    def __init__(self, env, agent1, agent2):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2

    def train(self, num_iterations=10000):
        for i in range(num_iterations):
            winner = self.env.play_one_game()
            reward = self.get_reward(winner)
            self.agent1.update(reward)
            self.agent2.update(-reward)

    def get_reward(self, winner):
        if self.agent1.player == Player.BLACK and winner == Player.BLACK:
            return 1.0
        elif self.agent1.player == Player.WHITE and winner == Player.WHITE:
            return 1.0
        else:
            return -1.0


if __name__ == '__main__':
    # wandb.init(project="wuziqi", entity="yungang")
    board_size = 15
    env = GameState(board_size)
    state_size = env.state_size
    action_size = env.action_size
    agent1 = DQNAgent(state_size,
                      action_size,
                      Player.BLACK,
                      buffer_size=4000,
                      batch_size=32,
                      gamma=0.95,
                      lr=0.001)

    agent1.load_model('/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/epoch_3460000')

    agent2 = RandomAgent(env)

    # num_episodes = 10000000
    max_steps = 1000
    num_epochs = 10000000
    save_epoch = 5000
    update_epoch = 200
    for epoch in range(num_epochs):
        print('epoch', epoch)
        history = []

        record = []
        players = [agent1, agent2]
        np.random.shuffle(players)
        state = env.reset()

        for step in range(max_steps):
            legal_moves = env.get_legal_moves()
            action = players[0].act(state, legal_moves)

            next_state, reward, done, winner = env.step(action, players[0].agent_name)
            history.append((state, action, reward, next_state))
            record.append(copy.deepcopy(env.board))

            agent1.push(state, action, reward, next_state, done)

            if done:
                print('reward, winner', reward, winner)
                break

            players.reverse()
            state = next_state

        if agent1.loss:
            print('1loss: ', agent1.loss.item())
            # wandb.log({
            #     "loss": agent1.loss.item(),
            # })

        if len(agent1.memory) > agent1.batch_size:
            experiences = agent1.memory.sample(agent1.batch_size)
            agent1.learn(experiences, agent1.gamma)

            if epoch % 200 == 0:
                agent1.update_epsilon()
                print('执行一次update_epsilon', agent1.epsilon)

            if epoch % update_epoch == 0:
                agent1.update_target_network()

            if epoch % save_epoch == 0:
                save_model_path = '/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/' + f'epoch_{epoch}'
                if not os.path.exists(save_model_path):
                    os.makedirs(save_model_path)
                model_path = save_model_path + '/model.pth'
                agent1.save_model(model_path)

        # if len(agent2.memory) > agent2.batch_size:
        #     experiences = agent2.memory.sample(agent2.batch_size)
        #     agent2.learn(experiences, agent2.gamma)
