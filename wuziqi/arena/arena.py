import random
from game.wuziqi import GameState
from arena.fight_agent import FightAgent


def evaluate(net_config, board_size, version1, version2, seed, num_epoch, device):
    wins = [0, 0]
    # 创建游戏环境
    env = GameState(board_size)
    agent1 = FightAgent(net_config, board_size, version1, seed, device)
    agent2 = FightAgent(net_config, board_size, version2, seed, device)
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
            action = agent.choose_action(state)
            next_obs, reward, done, info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

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

def arena(board_size, version1, version2, seed, num_epoch):
    device = 'cpu'
    score1, score2 = 0, 0
    a, b = evaluate(board_size, version1, version2, seed, num_epoch, device)
    print(f'{a} vs {b}')
    score1 += a
    score2 += b
    a, b = evaluate(board_size, version2, version1, seed, num_epoch, device)
    score1 += b
    score2 += a
    print(f'{a} vs {b}')
    return score1, score2