import os
import torch
import numpy as np
from util.common import get_logger
from policy.network import ACNet
from policy.ppo_policy import PPOPolicy

logger = get_logger(__name__)

class FightAgent:
    def __init__(self, net_config: str, version, checkpoint_path: str, seed: int, device: str) -> None:
        super().__init__()
        self.version = version
        self._device = device
        self.net = ACNet(**net_config['net']).to(device=device)
        self.old_net = ACNet(**net_config['net']).to(device=device)
        self.net.eval()
        self.old_net.eval()
        policy_config = net_config['policy']
        self.policy = PPOPolicy(policy_config, self.net, self.old_net, None)
        self.policy.set_seed(seed)
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f'load model from file [{checkpoint_path}].')
            self.policy.load(checkpoint_path)
        else:
            logger.error(f'checkpoint file [{checkpoint_path}] not found.')

    def act(self, state):
        obs = state['obs']
        action_mask = state['action_mask']

        actions_prob = self.policy.forward(obs)
        masked_actions = actions_prob * action_mask
        action = np.argmax(masked_actions)

        return action

    def _obs_to_torch(self, obs):
        cur_state = obs['cur_state']
        op_state = obs['op_state']
        blank_state = obs['blank_state']

        cur_state = cur_state.flatten()
        cur_state = torch.from_numpy(cur_state).to(self._device, dtype=torch.float32)

        op_state = op_state.flatten()
        op_state = torch.from_numpy(op_state).to(self._device, dtype=torch.float32)

        blank_state = blank_state.flatten()
        blank_state = torch.from_numpy(blank_state).to(self._device, dtype=torch.float32)

        obs = torch.cat([cur_state, op_state, blank_state], dim=-1)
        return obs

class RandomAgent:
    def act(self, state):
        acts = state['legal_actions']
        true_indices = np.where(acts)[0]
        # 如果没有 True，返回 None 或抛出异常
        if len(true_indices) == 0:
            return None  # 或者 raise ValueError("No True values in acts")
        # 在这些索引中随机选择一个
        random_index = np.random.choice(true_indices)
        return random_index