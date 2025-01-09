import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policy.base_policy import Policy
from policy.network import ACNet
from util.buffer import ReplayBuffer
from util.common import get_logger

logger = get_logger(__name__)

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

    def set_seed(self, seed):
        self._seed = seed
        if seed is not None:
            np.random.seed(seed=seed)

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
            batch_obs, act, rew = replay_buffer.sample(batch_size)

            # 将观测数据转为Tensor并堆叠成一个batch
            obs = torch.stack([self.obs_to_torch(obs) for obs in batch_obs], dim=0)

            # 前向传播，得到动作概率和状态值
            act_p, v = self.model(obs)
            old_act_p, old_v = self.old_model(obs)

            # 从每个动作概率中选取对应动作的概率
            act_prob = act_p[torch.arange(batch_size), act]
            old_act_p = old_act_p[torch.arange(batch_size), act]

            # 计算比例，避免除以零
            ratio = act_prob / (old_act_p + 1e-8)

            # 将奖励和价值转换为Tensor
            rew = torch.tensor(rew, dtype=torch.float32).to(self._device)  # 确保rewards是float32

            # 如果奖励的维度是1，扩展它的维度
            if rew.dim() == 1:
                rew = rew.unsqueeze(1)

            # 计算优势函数（advantage）
            adv = rew - old_v

            # 计算目标策略的损失
            surr = ratio * adv
            clipped_ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            actor_loss = -torch.mean(torch.min(surr, clipped_ratio * adv))

            # 计算值函数的损失
            critic_loss = F.mse_loss(v, rew)

            # 总损失
            loss = actor_loss + critic_loss

            # 梯度清零并反向传播
            self.optim.zero_grad()
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 更新参数
            self.optim.step()

            losses.append(loss.item())
            logger.info(f'Learning round: {self._learning_round}, Learning loss: {loss.item()}')
            torch.cuda.empty_cache()

        # 每隔5轮保存一次模型
        if self._learning_round % 5 == 0:
            self.save()

        return {"loss": losses}  # 返回最后一个batch的loss

    def get_v(self, obs):
        obs = self.obs_to_torch(obs)

        _, v = self.model(obs)
        return v

    def choose_action(self, state):
        action_mask = state['action_mask']
        legal_actions = state['legal_actions']
        obs = state['obs']

        if np.random.rand() < self._eps:
            action = self.exploration_noise(legal_actions)
        else:
            actions = self.forward(obs)
            masked_actions = actions * action_mask

            action = np.argmax(masked_actions)

        self._eps = max(self._eps * self._eps_decay, self._eps_min)

        return action

    def forward(self, obs):
        obs = self.obs_to_torch(obs)
        actions_prob, _ = self.model(obs)
        # 这里得actions_prob是batch_size的actions_prob，对每条数据，做mask后，取出最大值的索引
        actions_prob = actions_prob.detach().cpu().numpy()
        return actions_prob

    def exploration_noise(self, acts):
        # 获取所有为 True 的索引
        true_indices = np.where(acts)[0]
        # 如果没有 True，返回 None 或抛出异常
        if len(true_indices) == 0:
            return None  # 或者 raise ValueError("No True values in acts")
        # 在这些索引中随机选择一个
        random_index = np.random.choice(true_indices)
        return random_index

    def update_old_model(self):
        self.old_model.load_state_dict(self.model.state_dict())

    def save(self):
        torch.save(self.model.state_dict(), self.config['model_path'].format(version=self._learning_round))
        logger.info(f'Model save {self._learning_round}')

    def obs_to_torch(self, obs):
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

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.old_model.load_state_dict(torch.load(path))

def init_policy(**kwargs):
    device = kwargs.get('device', 'cpu')
    policy_config = kwargs.get('policy', {})
    model = ACNet(**kwargs['net']).to(device)
    old_model = ACNet(**kwargs['net']).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=policy_config['lr'])
    ppo_policy = PPOPolicy(config=policy_config, model=model, old_model=old_model, optim=optim)
    return ppo_policy