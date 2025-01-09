from util.buffer import ReplayBuffer
from policy.base_policy import Policy


class Agent:
    def __init__(self, policy: Policy, replay_buffer: ReplayBuffer):
        self.policy = policy
        self.replay_buffer = replay_buffer

    def choose_action(self, state):
        return self.policy.choose_action(state)

    def learn(self, batch_size):
        self.policy.learn(self.replay_buffer, batch_size)

    def get_v(self, state):
        return self.policy.get_v(state)
