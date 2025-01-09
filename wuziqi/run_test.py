from arena.arena import arena


def run(net_config, board_size, version1, version2, seed, num_epoch):
    print(f'version1:{version1}, version2:{version2}, seed:{seed}, num_epoch:{num_epoch}')
    return arena(net_config, board_size, version1, version2, seed, num_epoch)

if __name__ == '__main__':
    v1 = 5
    v2 = 100
    seed = 3
    num_epoch = 100
    board_size = 15
    test_config = {
        'device': 'cpu',
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

    score1, score2 = run(test_config, board_size, v1, v2, seed, num_epoch)
    print(f'version1: {v1} vs version2: {v2} -- score1: {score1} vs score2: {score2}')