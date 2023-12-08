import torch
import random
import time
import wandb
import os
from collections import deque
from env.wuziqi import GameState
from train.agent import DQNAgent
from util.common import Player

# 定义测试函数
def test(agent1, agent2, num_games):
    # 创建游戏环境
    env = GameState(15)

    # 初始化得分
    score1, score2 = 0, 0

    # 进行num_games局游戏
    for i in range(num_games):
        # 随机决定先手方
        if random.random() < 0.5:
            cur_agent1, cur_agent2 = agent1, agent2
        else:
            cur_agent1, cur_agent2 = agent2, agent1

        # 初始化游戏状态
        state = env.reset()
        done = 0

        # 对局
        while not done:
            # 由agent1选择动作并执行
            legal_moves = env.get_legal_moves()
            action = cur_agent1.act(state, legal_moves)
            next_state, done, winner = env.step(action, cur_agent1.agent_name)
            # cur_agent1.memorize(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

            # 由agent2选择动作并执行
            legal_moves = env.get_legal_moves()
            action = cur_agent2.act(state, legal_moves)
            next_state, done, winner = env.step(action, cur_agent2.agent_name)
            # cur_agent2.memorize(state, action, reward, next_state, done)
            state = next_state

        # 更新得分
        if winner == Player.BLACK.value:
            score1 += 1
        elif winner == Player.WHITE.value:
            score2 += 1


    # 输出胜负情况和得分
    print("Agent1 wins:", score1)
    print("Agent2 wins:", score2)
    print("Total score:", score1 - score2)
    return score2 - score1

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