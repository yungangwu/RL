from game.wuziqi import GameState
from policy.ppo_policy import init_policy
from agent.agent import Agent
from util.buffer import ReplayBuffer

BATCH_SIZE = 64
GAMMA = 0.9

def train(env: GameState, agent: Agent, replay_buffer: ReplayBuffer, num_epochs: int = 10000):
    buf_s, buf_r, buf_a = [], [], []
    for _ in range(num_epochs):
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
            buf_s.append(obs)
            buf_a.append(action)
            buf_r.append(reward)

            obs = next_obs

        # 在一轮数据收集完了后，要对reward进行回溯，得到每一步的reward，每一步的reward要根据v值加reward做一个折算
        buffer_size = replay_buffer.size()
        v_s_ = agent.get_v(next_obs)
        discounted_r = []
        for r in buf_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()

        replay_buffer.push(buf_s, buf_a, buf_r)
        buf_s, buf_a, buf_r = [], [], []

        if buffer_size > BATCH_SIZE * 2:
            agent.learn(BATCH_SIZE)
            replay_buffer.clear()

if __name__ == '__main__':
    board_size = 15
    buffer_size = 10000
    num_epochs = 100000
    train_config = {
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
            'device': 'cuda',
            'repeat': 3,
        },
        'checkpoint_path': './path_to_model/ppo_model_4890.pth',
    }
    game = GameState(board_size)
    policy = init_policy(**train_config)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = Agent(policy, replay_buffer)

    train(game, agent, replay_buffer, num_epochs)