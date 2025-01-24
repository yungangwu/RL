import random
from collections import defaultdict
from arena.arena import arena
from util.common import plot_loss


def group_round_robin(net_config, models, num_rounds_per_match=3, group_size=4):
    """
    分组循环赛
    :param models: 所有参赛模型列表
    :param num_rounds_per_match: 每两模型对战的轮次数
    :param group_size: 每组的模型数量
    :return: 每个模型的胜率
    """
    random.shuffle(models)
    groups = [models[i:i + group_size] for i in range(0, len(models), group_size)]

    # 每组进行循环赛
    group_results = defaultdict(int)  # 记录每个模型的胜利次数
    for group in groups:
        for i, model in enumerate(group):
            for j, opponent in enumerate(group):
                if i != j:
                    model_score, opponent_score = arena(net_config, board_size, model, opponent, seed, num_rounds_per_match)
                    print(f'version1: {model}, version2: {opponent}, score1: {model_score}, score2: {opponent_score}')
                    group_results[model] += model_score
        print('group_results:', group_results)

    # 计算胜率
    win_rates = {}
    for model in models:
        total_matches = (len(groups[0]) - 1) * num_rounds_per_match
        win_rates[model] = group_results[model] / total_matches if total_matches > 0 else 0

    return win_rates, len(groups)

def tournament(net_config, models, num_rounds_per_match=3, group_size=4, top_k=2):
    current_round = 1
    while len(models) > group_size:
        print(f"=== 第 {current_round} 轮分组循环赛 ===")
        # 执行分组循环赛
        win_rates, group_len = group_round_robin(net_config, models, num_rounds_per_match, group_size)
        print(f"胜率统计: {win_rates}")

        # 每组选出前 top_k 名晋级
        sorted_models = sorted(models, key=lambda m: win_rates[m], reverse=True)
        models = sorted_models[:group_len * top_k]
        print(f"晋级模型: {[m for m in models]}, models: {models}")

        current_round += 1

    # 决赛
    print("=== 决赛 ===")
    win_rates, group_len = group_round_robin(net_config, models, num_rounds_per_match, group_size)
    sorted_final = sorted(models, key=lambda m: win_rates[m], reverse=True)
    print('sorted_final:', sorted_final)
    return sorted_final[0]  # 返回最终冠军

def run_tournament(net_config, models):
    winner = tournament(net_config, models)
    print(f'final winner: {winner}')


def run(net_config, board_size, version1, version2, seed, num_epoch):
    print(f'version1:{version1}, version2:{version2}, seed:{seed}, num_epoch:{num_epoch}')
    return arena(net_config, board_size, version1, version2, seed, num_epoch)

if __name__ == '__main__':
    v1 = 7600
    v2 = 28020
    seed = 3
    num_epoch = 3
    board_size = 15
    test_config = {
        'device': 'cuda',
        'net': {
                'state_dim': board_size ** 2 * 3,
                'action_dim': board_size ** 2,
            },
        'policy': {
            'model_path': './path_to_model/ppo_model_{version}.pth',
            'lr': 0.0001,
            'eps': 0.3,
            'eps_decay': 0.9999,
            'eps_min': 0,
            'device': 'cpu',
            'repeat': 3,
        }
    }

    # 随机指定两个模型进行对战
    score1, score2 = run(test_config, board_size, v1, v2, seed, num_epoch)
    print(f'version1: {v1} vs version2: {v2} -- score1: {score1} vs score2: {score2}')

    # v2为0代表随机对手，测试所有模型与随机对手的竞争分数
    # results = []
    # for v1 in range(5, 28750, 5):
    #     score1, score2 = run(test_config, board_size, v1, v2, seed, num_epoch)
    #     print(f'version1: {v1} vs version2: {v2} -- score1: {score1} vs score2: {score2}')
    #     results.append(score1)
    #     reward_dict = {'reword': results}
    #     plot_loss(reward_dict, save_path='./data/result/reward.png', title='Training reward', ylabel='Reward')

    # 循环锦标赛获取分数最高的模型
    # model_version_list = [i for i in range(5, 30260, 5)]
    # run_tournament(test_config, model_version_list)