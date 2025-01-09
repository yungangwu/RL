import random
from game.wuziqi import GameState
from arena.fight_agent import FightAgent, RandomAgent


MODEL_FILE_PATH = "./path_to_model/ppo_model_{version}.pth"
def evaluate(config, board_size, version1, version2, seed, num_epoch, device):
    wins = [0, 0]
    # 创建游戏环境
    env = GameState(board_size)
    env.set_seed(seed=seed)

    if version1 == 0:
        agent1 = RandomAgent()
        agent2 = FightAgent(config, checkpoint_path=MODEL_FILE_PATH.format(version=version2), seed=seed, device=device)
    elif version2 == 0:
        agent1 = FightAgent(config, checkpoint_path=MODEL_FILE_PATH.format(version=version1), seed=seed, device=device)
        agent2 = RandomAgent()
    else:
        agent1 = FightAgent(config, checkpoint_path=MODEL_FILE_PATH.format(version=version1), seed=seed, device=device)
        agent2 = FightAgent(config, checkpoint_path=MODEL_FILE_PATH.format(version=version2), seed=seed, device=device)
    agents = [agent1, agent2]

    # 进行num_games局游戏
    for _ in range(num_epoch):
        obs = env.reset()
        legal_actions = env.get_legal_actions()
        action_mask = env.action_mask()
        info = {
            'legal_actions': legal_actions,
            'action_mask': action_mask,
        }
        done = False

        while not done:
            state = {'obs': obs, 'legal_actions': info['legal_actions'], 'action_mask': info['action_mask']}
            chair_id = env.get_chair_id()
            agent = agents[chair_id]
            action = agent.act(state)
            next_obs, reward, done, info = env.step(action)
            obs = next_obs

        next_chair_id = env._get_next_chair_id(chair_id)
        wins[chair_id] += reward
        wins[next_chair_id] -= reward

    # 输出胜负情况和得分
    print(f"{wins[0]} vs {wins[1]}")
    return wins

def arena(net_config, board_size, version1, version2, seed, num_epoch):
    device = 'cpu'
    score1, score2 = 0, 0
    a, b = evaluate(net_config, board_size, version1, version2, seed, num_epoch, device)
    print(f'{a} vs {b}')
    score1 += a
    score2 += b
    a, b = evaluate(net_config, board_size, version2, version1, seed, num_epoch, device)
    score1 += b
    score2 += a
    print(f'{a} vs {b}')
    return score1, score2