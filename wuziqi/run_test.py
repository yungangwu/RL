import torch
import random
import time
import wandb
import os
from collections import deque
from game.wuziqi import GameState
from agent.agent import Agent

board_size = 15
env = GameState(board_size)
state_size = env.state_size
action_size = env.action_size
# 创建DQNAgent实例并加载模型
agent1 = DQNAgent(state_size, action_size, Player.BLACK, buffer_size=2000, batch_size=32, gamma=0.95, lr=0.001)
agent1.load_model('/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/epoch_7620000/model.pth')
agent2 = DQNAgent(state_size, action_size, Player.WHITE, buffer_size=2000, batch_size=32, gamma=0.95, lr=0.001)
agent2.load_model('/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/epoch_5000/model.pth')
# path1 = '/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/epoch_5000/model.pth'
# for k in range(10000,950000,5000):
#     path2 = '/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/epoch_' + f'{k}/model.pth'
#     # path2 = '/home/yg/code/ReinforcementLearning/wuziqi/path_to_model/epoch_' + f'{k+5000}/model.pth'

#     agent1.load_model(path1)
#     agent2.load_model(path2)

#     # 进行测试
#     score = test(agent1, agent2, 1000)
#     # wandb.log("score", score)
#     time.sleep(20)
test(agent1, agent2, 1000)