import threading
import numpy as np
import logging
import time

from util.buffer import ReplayBuffer
from model.ppo import PPOPolicyValue
from util.retrowrapper import RetroWrapper

# 查看所有支持的游戏
# retro.data.list_games()

ACTION_DIM = 9
EP_LEN = 1000
MIN_BATCH_SIZE = 128
BUFFER_SIZE = 2000
EPOCHS = 20000
HEIGHT = 224
WIDE = 240

# 配置 logging 模块
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainPipeline:
    def __init__(self, buffer, action_dim) -> None:
        self.replay_buffer = buffer
        self.action_space = np.arange(action_dim)

    # 环境信息收集
    def work(self, ppo: PPOPolicyValue, env):
        for epo in range(EPOCHS):
            logger.info(f'Worker {threading.current_thread().name} - Epo: {epo}')
            state = env.reset()
            print('*' * 30)
            for t in range(EP_LEN):
                logger.info(f'ep: {t}')
                action = ppo.get_action(state, self.action_space)
                state_, reward, done, _ = env.step(action)

                self.replay_buffer.push([(state.transpose(2, 0, 1), action.numpy(), reward, state_.transpose(2, 0, 1))])
                state = state_

                if t == EP_LEN - 1 or done:
                    logger.info(f'Worker {threading.current_thread().name} - Step: {t}, Reward: {reward}')
                    break

def training_loop(ppo, replay_buffer: ReplayBuffer, batch_size=MIN_BATCH_SIZE):
    while True:
        if replay_buffer.len() >= batch_size:
            act_loss = ppo.train_step(replay_buffer, batch_size)
            logger.info(f'Training - Action Loss: {act_loss}')
        else:
            time.sleep(1)  # 避免繁忙等待

def running_train(ppo, replay_buffer, envs, num_works):
    # 启动环境收集线程
    threads = []
    for i in range(num_works):
        print(f'创建环境收集线程{i}')
        pipeline = TrainPipeline(replay_buffer, ACTION_DIM)
        work_thread = threading.Thread(target=pipeline.work, args=(ppo, envs[i]), daemon=True)
        work_thread.start()
        threads.append(work_thread)

    # 启动训练线程
    train_thread = threading.Thread(target=training_loop, args=(ppo, replay_buffer, MIN_BATCH_SIZE), daemon=True)
    train_thread.start()
    # threads.append(train_thread)


if __name__ == '__main__':
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    ppo = PPOPolicyValue(HEIGHT, WIDE, ACTION_DIM)

    game = 'IceClimber-Nes'
    num_works = 4
    envs = [RetroWrapper(game) for _ in range(num_works)]

    print('开始训练过程')
    running_train(ppo, replay_buffer, envs, num_works)