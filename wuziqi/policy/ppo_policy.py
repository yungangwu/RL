import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_policy import Policy
from policy.network import ACNet
from util.buffer import ReplayBuffer

EPSILON = 0.2

class PPOPolicy(Policy):
    def __init__(
        self,
        config,
        model: torch.nn.Module,
        old_model: torch.nn.Module,
        optim: torch.optim.Optimizer,
    ):
        super(PPOPolicy, self).__init__()
        self.config = config
        self.model = model
        self.old_model = old_model
        self.optim = optim
        self._device = config.get('device', 'cpu')
        self._eps = config.get('eps', 0.2)
        self._eps_decay = config.get('eps_decay', 0.9999)
        self._eps_min = config.get('eps_min', 0)
        self._repeat = config.get('repeat', 3)
        self._learning_round = 0
        self.loss_fn = torch.nn.MSELoss().to(self._device)

    def set_eps(self, eps):
        self._eps = eps

    def set_training(self, training: bool):
        if training:
            self.model.train()
        else:
            self.model.eval()

    def learn(self, replay_buffer: ReplayBuffer, batch_size: int):
        losses = []
        self._learning_round += 1
        for _ in range(self._repeat):
            # 得到batch_size个数据，从replay_buffer中进行采样
            obs, act, rew, _, _ = replay_buffer.sample(batch_size)
            obs = self.obs_to_torch(obs)
            act_p, v = self.model(obs)
            old_act_p, old_v = self.old_model(obs)
            act_prob = act_p[torch.arange(batch_size), act]
            old_act_p = old_act_p[torch.arange(batch_size), act]
            ratio = act_prob / (old_act_p + 1e-8)
            adv = rew - old_v
            surr = ratio * adv
            actor_loss = -torch.mean(surr, torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON) * adv)

            critic_loss = F.mse_loss(v, rew)

            loss = actor_loss + critic_loss
            self.optim.zero_grad()
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()
            losses.append(loss.item())

        if self._learning_round % 5 == 0:
            self.save()

        return {"loss": loss}

    def choose_action(self, state):
        action_mask = state['action_mask']
        legal_actions = state['legal_actions']
        obs = state['obs']

        if np.random.rand() < self._eps:
            action = self.exploration_noise(legal_actions)
        else:
            actions = self.forward(obs)
            masked_actions = actions * action_mask

            max_a = np.argmax(masked_actions)
            action = actions[max_a]

        self._eps = max(self._eps * self._eps_decay, self._eps_min)

        return action

    def forward(self, obs):
        obs = self.obs_to_torch(obs)
        actions_prob, _ = self.model(obs)
        # 这里得actions_prob是batch_size的actions_prob，对每条数据，做mask后，取出最大值的索引
        return actions_prob

    def exploration_noise(self, acts):
        # 随机选择一个索引
        random_index = np.random.choice(len(acts))
        # 根据索引选择一个act
        random_act = acts[random_index]
        return random_act

    def update_old_model(self):
        self.old_model.load_state_dict(self.model.state_dict())

    def save(self):
        torch.save(self.model.state_dict(), self.config['model_path'].format(self._learning_round))

    def obs_to_torch(self, obs):
        cur_state = obs['cur_state']
        op_state = obs['op_state']
        blank_state = obs['blank_state']

        cur_state = cur_state.flatten()
        cur_state = torch.from_numpy(cur_state).to(self._device)

        op_state = op_state.flatten()
        op_state = torch.from_numpy(op_state).to(self._device)

        blank_state = blank_state.flatten()
        blank_state = torch.from_numpy(blank_state).to(self._device)

        obs = torch.cat([cur_state, op_state, blank_state], dim=-1)
        return obs


def init_policy(**kwargs):
    policy_config = kwargs.get('policy', {})
    model = ACNet(**kwargs['net'])
    old_model = ACNet(**kwargs['net'])
    optim = torch.optim.Adam(model.parameters(), lr=policy_config['lr'])
    ppo_policy = PPOPolicy(config=policy_config, model=model, old_model=old_model, optim=optim)
    return ppo_policy