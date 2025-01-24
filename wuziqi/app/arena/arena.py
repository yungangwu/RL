import random
import logging
from game.wuziqi import GameState
from arena.fight_agent import FightAgent, RandomAgent
from util.common import get_logger

logger = get_logger(__name__, logging.DEBUG)


MODEL_FILE_PATH = "./path_to_model/ppo_model_{version}.pth"
def evaluate(config, board_size, version1, version2, seed, num_epoch, device):
    wins = {version1:0, version2:0}
    # 创建游戏环境
    env = GameState(board_size)
    env.set_seed(seed=seed)
    versions = [version1, version2]

    if version1 == 0:
        agent1 = RandomAgent()
        agent2 = FightAgent(config, version2, checkpoint_path=MODEL_FILE_PATH.format(version=version2), seed=seed, device=device)
    elif version2 == 0:
        agent1 = FightAgent(config, version1, checkpoint_path=MODEL_FILE_PATH.format(version=version1), seed=seed, device=device)
        agent2 = RandomAgent()
    else:
        agent1 = FightAgent(config, version1, checkpoint_path=MODEL_FILE_PATH.format(version=version1), seed=seed, device=device)
        agent2 = FightAgent(config, version2, checkpoint_path=MODEL_FILE_PATH.format(version=version2), seed=seed, device=device)
    agents = [agent1, agent2]

    # 进行num_games局游戏
    results = []
    for _ in range(num_epoch):
        obs = env.reset()
        legal_actions = env.get_legal_actions()
        action_mask = env.action_mask()
        info = {
            'legal_actions': legal_actions,
            'action_mask': action_mask,
        }
        done = False
        cur_idx = random.choice([0, 1])
        agent = agents[cur_idx]
        logger.debug(f'start agent version: {agent.version}')
        while not done:
            state = {'obs': obs, 'legal_actions': info['legal_actions'], 'action_mask': info['action_mask']}
            action = agent.act(state)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            cur_idx = (cur_idx + 1) % 2
            agent = agents[cur_idx]

        next_idx = (cur_idx + 1) % 2
        wins[versions[next_idx]] += reward
        wins[versions[cur_idx]] -= reward
        logger.debug(f'agents[next_idx].version: {agents[next_idx].version}. reward: {reward}. wins: {wins}')
        results.append((agents[next_idx].version, env.board))

    print(f'v1: {version1}, v2: {version2}, results:{results}')
    logger.debug(f'v1: {version1}, v2: {version2}, results:{results}')
    # 输出胜负情况和得分
    logger.debug(f"{wins}")
    return wins[version1], wins[version2]

def arena(net_config, board_size, version1, version2, seed, num_epoch):
    device = 'cpu'
    score1, score2 = 0, 0
    a, b = evaluate(net_config, board_size, version1, version2, seed, num_epoch, device)
    logger.debug(f'version1 vs version2: {version1} vs {version2}, {a} vs {b}')
    score1 += a
    score2 += b
    a, b = evaluate(net_config, board_size, version2, version1, seed, num_epoch, device)
    score1 += b
    score2 += a
    logger.debug(f'version2 vs version1: {version2} vs {version1}, {a} vs {b}')
    return score1, score2