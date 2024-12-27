import torch
import numpy as np
from torch import nn
from typing import Any, Dict, Union, List, Optional
from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from tianshou.data.utils.converter import to_torch
from tianshou.utils import RunningMeanStd
from lami.policy.networks import MCQNet, AttentionNet, LSTMAttentionNet, TransformerNet, TransformerNetEx
from krl.registry.policy_registry import registry


def get_obs(b: Batch):
    # 获取批量数据的大小
    size = len(b)
    # 将批量数据的观测值转换为列表
    obs_list = list(b.obs.values())
    # 对列表中的每个观测值进行重塑，使其形状为(size, -1)
    obs_list = list(map(lambda x: np.reshape(x, (size, -1)), obs_list))
    # 将重塑后的观测值列表水平堆叠成一个二维数组
    obs = np.hstack(obs_list)
    # 返回堆叠后的观测值数组
    return obs

def get_info_items(infos, key):
    items = []
    for info in infos:
        items.append(info[key])
    return np.array(items)

class MCQPolicy(BasePolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        reward_normalization: bool = False,
        eps: float = 0.,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self.loss_fn = nn.MSELoss().to(device)
        self._device = device
        self.eps = eps
        self.lr_scheduler = lr_scheduler

    def set_eps(self, eps: float) -> None:
        self.eps = eps

    def set_training(self, training: bool):
        if training:
            self.train()
            self.model.train()
        else:
            self.eval()
            self.model.eval()

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(batch, buffer, indices, v_s_, gamma=1.0, gae_lambda=1.0)
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + 1e-8)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, List[float]]:
        losses = []
        for _ in range(repeat):
            for b in batch.split(batch_size, shuffle=True, merge_last=True):
                obs = get_obs(b)
                act = b.act
                act = np.reshape(act, (act.shape[0], -1))
                s_a = np.concatenate((obs, act), axis=-1)
                s_a = to_torch(s_a, torch.float32, device=self._device)
                self.optim.zero_grad()
                q = self.model(s_a)
                returns = to_torch_as(b.returns, q)
                returns = torch.reshape(returns, (-1, 1))
                loss = self.loss_fn(q, returns)
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {"loss": losses}

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]], **kwargs: Any) -> Batch:
        obs = get_obs(batch)
        info = batch.info
        legal_a = get_info_items(info, "legal_actions")
        legal_a_num = get_info_items(info, "legal_actions_num")
        obs = np.repeat(obs, legal_a_num, 0)
        act = np.vstack(legal_a)
        act = np.reshape(act, (act.shape[0], -1))
        s_a = np.concatenate((obs, act), axis=-1)

        s_a = to_torch(s_a, torch.float32, device=self._device)
        q = self.model(s_a)
        q = to_numpy(q)

        splits = np.add.accumulate(legal_a_num)
        splits = splits[:-1]

        q = np.split(q, splits, axis=0)
        act = []
        for i, logits in enumerate(q):
            max_a = np.argmax(logits)
            act.append(legal_a[i][max_a])
        return Batch(logits=q, act=act)

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if not np.isclose(self.eps, 0.0): # 判断一个值是否在某个范围内接近0.0
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            info = batch.info
            legal_a = get_info_items(info, "legal_actions")
            legal_a_num = get_info_items(info, "legal_actions_num")
            max_a_num = np.max(legal_a_num)
            q = np.random.rand(bsz, max_a_num)
            masks = np.arange(max_a_num, dtype=np.int)
            masks = np.repeat(masks[np.newaxis, :], bsz, axis=0)
            valid_masks = np.repeat(legal_a_num[:, np.newaxis], max_a_num, axis=-1)
            masks = np.where(masks < valid_masks, 1, 0)
            q += masks
            rand_act_ind = q.argmax(axis=1)
            rand_act = []
            for i, max_act_ind in enumerate(rand_act_ind):
                rand_act.append(legal_a[i][max_act_ind])
            rand_act = np.array(rand_act)
            act[rand_mask] = rand_act[rand_mask]
        return act

def init_net(net: nn.Module):
    for m in list(net.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight) # 正交初始化，可以确保权重矩阵是正交的，当权重矩阵需要保持其形状或结构时，这是有益的
            torch.nn.init.zeros_(m.bias) # 对偏置使用零初始化
        if isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith("weight"):
                    nn.init.xavier_normal_(param) # Xavier初始化是为了确保训练时，激活值和梯度能在各层之间很好传播
                else:
                    nn.init.zeros_(param) # 如果不是weight开头的，则使用零初始化，但是一般在LSTM的训练中，一般会对于forget gate的偏置会给予一个正值，确保训练时能记住之前的信息

def lami_policy(**kwargs):
    device = kwargs.get("device", "cpu")
    input_dim = kwargs.pop('input_dim', 64)
    weight_decay = kwargs.pop('weight_decay', 0)
    lr = kwargs.pop('lr', 0.001)
    print(f'lami_policy params: lr {lr}, weight_decay: {weight_decay}, input_dim: {input_dim}, device:{device}')
    net = MCQNet(input_dim=input_dim).to(device)
    init_net(net)
    optim = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=weight_decay)
    policy = MCQPolicy(net, optim, **kwargs)
    return policy

registry('lami_policy', entry_point=lami_policy)

class AttentionPolicy(BasePolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        reward_normalization: bool = False,
        eps: float = 0.,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        device: str = 'cuda',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        if 'rms' in kwargs:
            self.ret_rms.mean = kwargs['rms']['mean']
            self.ret_rms.var = kwargs['rms']['var']
            self.ret_rms.count = kwargs['rms']['count']
        self.loss_fn = nn.MSELoss().to(device)
        self._device = device
        self.eps = eps
        self.lr_scheduler = lr_scheduler

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def set_training(self, training: bool):
        if training:
            self.train()
            self.model.train()
        else:
            self.eval()
            self.model.eval()

    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(batch, buffer, indices, v_s_=v_s_, gamma=1.0, gae_lambda=1.0)
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / np.sqrt(self.ret_rms.var + 1e-8)
        else:
            batch.returns = unnormalized_returns
        return batch

    def learn(self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any) -> Dict[str, List[float]]:
        losses = []
        for _ in range(repeat):
            for b in batch.split(batch_size, shuffle=True, merge_last=True):
                obs = to_torch(b.obs, torch.float32, self._device)
                act = to_torch(b.act, torch.float32, self._device)
                self.optim.zero_grad()
                q = self.model(obs, act)
                returns = to_torch_as(b.returns, q)
                returns = torch.reshape(returns, (-1, 1))
                loss = self.loss_fn(q, returns)
                loss.backward()
                self.optim.step()
                losses.append(loss.item())
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            print(f'lr: {self.lr_scheduler.get_last_lr()}')

        return {'loss': losses}

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]], **kwargs: Any) -> Batch:
        obs = batch.obs
        info = batch.info
        legal_a = get_info_items(info, 'legal_actions')
        legal_a_num = get_info_items(info, 'legal_actions_num')
        for k, v in obs.items():
            obs[k] = np.repeat(v, legal_a_num, 0)
        act = np.vstack(legal_a)
        obs = to_torch(obs, torch.float32, self._device)
        act = to_torch(act, torch.float32, self._device)
        q = self.model(obs, act)
        q = to_numpy(q)

        splits = np.add.accumulate(legal_a_num)
        splits = splits[:-1]
        q = np.split(q, splits, axis=0)
        act = []
        for i, logits in enumerate(q):
            max_a = np.argmax(logits)
            act.append(legal_a[i][max_a])
        return Batch(logits=q, act=act)

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            info = batch.info
            legal_a = get_info_items(info, 'legal_actions')
            legal_a_num = get_info_items(info, 'legal_actions_num')
            max_a_num = np.max(legal_a_num)
            q = np.random.rand(bsz, max_a_num)

            masks = np.arange(max_a_num, dtype=np.int)
            masks = np.repeat(masks[np.newaxis, :], bsz, axis=0)
            valid_masks = np.repeat(legal_a_num[np.newaxis, :], max_a_num, axis=-1)
            masks = np.where(masks < valid_masks, 1, 0)
            q += masks
            rand_act_ind = q.argmax(axis=1)
            rand_act = []
            for i, max_act_ind in enumerate(rand_act_ind):
                rand_act.append(legal_a[i][max_act_ind])
            rand_act = np.array(rand_act)
            act[rand_mask] = rand_act[rand_mask]
        return act

def init_lami_attention_net(net: nn.Module):
    for m in list(net.modules()):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal__(m.weight)
            nn.init.zeros_(m.bias)
        if isinstance(m, torch.nn.MultiheadAttention) or isinstance(m, torch.nn.LSTM):
            for name, param in m.named_parameters():
                if name.startswith('weight'):
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)

def lami_attention_policy(**kwargs):
    device = kwargs.get('device', 'cpu')
    net_params = kwargs.pop('net_params', {})
    weight_decay = kwargs.get('weight_decay', 0)
    lr = kwargs.pop('lr', 0.001)
    print('===================================')
    print(f'lami_attention_policy params:\nlr: {lr}\nweight_decay: {weight_decay}\nnet_params: {net_params}\ndevice: {device}')
    net = AttentionNet(**net_params).to(device)
    init_lami_attention_net(net)
    optim = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=weight_decay)
    policy = AttentionPolicy(net, optim, **kwargs)
    return policy

def lami_attention_lstm_policy(**kwargs):
    device = kwargs.get('device', 'cpu')
    net_params = kwargs.pop('net_params', {})
    weight_decay = kwargs.pop('weight_decay', 0)
    lr = kwargs.pop('lr', 0.001)
    print('===================================')
    print(f'lami_attention_lstm_policy params:\nlr: {lr}\nweight_decay: {weight_decay}\nnet_params: {net_params}\ndevice: {device}')
    net = LSTMAttentionNet(**net_params).to(device)
    init_lami_attention_net(net)
    optim = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=weight_decay)
    lr_scheduler_lambda = kwargs.pop('lr_scheduler', None)
    if lr_scheduler_lambda:
        lr_scheduler_lambda = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=[lr_scheduler_lambda])
        kwargs['lr_scheduler'] = lr_scheduler_lambda
    policy = AttentionPolicy(net, optim, **kwargs)
    return policy

def lami_transformer_policy(**kwargs):
    device = kwargs.get('device', 'cpu')
    net_params = kwargs.pop('net_params', {})
    weight_decay = kwargs.pop('weight_decay', 0)
    lr = kwargs.pop('lr', 0.001)
    print('===================================')
    print(f'lami_transformerex_policy params:\nlr: {lr}\nweight_decay: {weight_decay}\nnet_params: {net_params}\ndevice: {device}')
    net = TransformerNet(**net_params).to(device)
    optim = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=weight_decay)
    lr_scheduler_lambda = kwargs.pop('lr_scheduler', None)
    if lr_scheduler_lambda:
        lr_scheduler_lambda = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=[lr_scheduler_lambda])
        kwargs['lr_scheduler'] = lr_scheduler_lambda
    policy = AttentionPolicy(net, optim, **kwargs)
    return policy

def lami_transformerex_policy(**kwargs):
    device = kwargs.get('device', 'cpu')
    net_params = kwargs.pop('net_params', {})
    weight_decay = kwargs.pop('weight_decay', 0)
    lr = kwargs.pop('lr', 0.001)
    print('===================================')
    print(f'lami_transformerex_policy params:\nlr: {lr}\nweight_decay: {weight_decay}\nnet_params: {net_params}\ndevice: {device}')
    net = TransformerNetEx(**net_params).to(device)
    optim = torch.optim.Adam(list(net.parameters()), lr=lr, weight_decay=weight_decay)
    lr_scheduler_lambda = kwargs.pop('lr_scheduler', None)
    if lr_scheduler_lambda:
        lr_scheduler_lambda = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=[lr_scheduler_lambda])
        kwargs['lr_scheduler'] = lr_scheduler_lambda
    policy = AttentionPolicy(net, optim, **kwargs)
    return policy

registry('lami_attention_policy', entry_point=lami_attention_policy)
registry('lami_attention_lstm_policy', entry_point=lami_attention_lstm_policy)
registry('lami_transformer_policy', entry_point=lami_transformer_policy)
registry('lami_transformerex_policy', entry_point=lami_transformerex_policy)