import gym
from doubledqn import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

natural_DQN = DoubleDQN(
    n_actions=ACTION_SPACE, n_features=3, e_greedy=0.001, double_q=False
)

double_DQN = DoubleDQN(
    n_actions=ACTION_SPACE, n_features=3, e_greedy=0.001, double_q=True
)

def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4) # 用于将一个离散的动作值映射到一个连续的空间

        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10

        RL.store_transiton(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()
        
        if total_steps - MEMORY_SIZE > 20000:
            break

        observation = observation_
        total_steps += 1
    return RL.q

q_natural = train(natural_DQN)
q_double = train(double_DQN)

# 出对比图
plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('training steps')
plt.grid()
plt.show()