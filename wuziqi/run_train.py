from game.wuziqi import GameState
from policy.ppo_policy import init_policy
from agent.agent import Agent
from util.buffer import ReplayBuffer

BATCH_SIZE = 64

def train(env: GameState, agent: Agent, replay_buffer: ReplayBuffer, num_epochs: int = 10000):

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
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

        buffer_size = replay_buffer.size()
        if buffer_size > BATCH_SIZE * 2:
            agent.learn(BATCH_SIZE)
            replay_buffer.clear()

if __name__ == '__main__':
    board_size = 15
    buffer_size = 10000
    num_epochs = 10000
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
        }
    }
    game = GameState(board_size)
    policy = init_policy(**train_config)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = Agent(policy, replay_buffer)

    train(game, agent, replay_buffer, num_epochs)