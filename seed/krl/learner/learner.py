import io
import os
import asyncio
import h5py
import torch
import logging
from typing import Any, Optional
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data.utils.converter import from_hdf5
from tianshou.utils import MovAvg, statistics
from krl.proto import learner_pb2
from krl.learner.model_sender import ModelSender
from krl.registry.policy_registry import policy_registry
from krl.util.utils import PolicyWrapper
from krl.util.timer import Timer
from collections import defaultdict
from tianshou.data import to_numpy

logger = logging.getLogger(__name__)

class WarmuSchedule:
    def __init__(self, step) -> None:
        # 将传入的参数 step 赋值给实例变量 _step
        self._step = step
        # 初始化实例变量 _last_step 为 0
        self._last_step = 0

    def step(self, step):
        self._last_step = step
        return 1 if step >= self._step else 0

    def can_sync_model(self):
        return self._last_step >= self._step

def buffer_from_hdf5(cls, data: str, device: Optional[str]=None):
    bio = io.BytesIO(data)
    with h5py.File(bio, "r") as f:
        buf = cls.__new__(cls)
        buf.__setstate__(from_hdf5(f, device=device))
    return buf

class Learner(object):
    def __init__(self, config: dict) -> None:
        super().__init__()

        device = config.get('device', 'cpu')
        max_queue_size = config.get('max_queue_size', 8)
        self.queue = asyncio.Queue(max_queue_size)

        self._device = device
        self._batch_size = config.get('batch_size', 64)
        self._repeat_per_sample = config.get('repeat_per_sample', 8)
        self._stats = {}
        self._policy_warmups = {}

        self._model_sender = ModelSender(**config.get('sender', {}))
        self._learn_count = 0
        self._model_send_count = config.get('model_sync_interval', 1)

        self.create_policy(config.get('policy', None))

    def create_policy(self, policy_config: dict):
        policies = {}
        for policy_name, policy_option in policy_config.items():
            policy_id = policy_option.pop('id', policy_name)
            version = policy_option.pop('version', -1)
            buffer_size = policy_option.pop('buffer_size', -1)
            optimizer_checkpoint_path = policy_option.pop('optimizer_checkpoint_dir', '')
            warmup_episode = policy_option.pop('warmup_episode', 0)
            if warmup_episode > 0:
                warmup_scheduler = WarmuSchedule(warmup_episode)
                self._policy_warmups[policy_name] = warmup_scheduler
                policy_option['lr_scheduler'] = lambda step: warmup_scheduler.step(step)
            policy = policy_registry.make(policy_id, device=self._device, **policy_option)
            setattr(policy, 'policy_name', policy_name)
            setattr(policy, 'policy_id', policy_id)
            policy.set_training(True)
            self._stats[policy_name] = {
                'gradient_step': 0,
                'loss': defaultdict(MovAvg),
            }
            policies[policy_name] = PolicyWrapper(policy=policy, version=version)
            if buffer_size > 0:
                self._stats[policy_name]['buffer'] = ReplayBuffer(buffer_size)
            if optimizer_checkpoint_path:
                optimizer_checkpoint_file = os.path.join(optimizer_checkpoint_path, f'{policy_name}_optim_checkpoint.mdl')
                self._stats[policy_name]['checkpoint_file'] = optimizer_checkpoint_file
                logger.info(f'policy {policy_name} optim checkpoint path: [{optimizer_checkpoint_file}].')
                if os.path.exists(optimizer_checkpoint_file):
                    with open(optimizer_checkpoint_file, 'rb') as f:
                        print(f'load optim checkpoint from file [{optimizer_checkpoint_file}].')
                        optimizer_parameters = io.BytesIO(f.read())
                        state_dict = torch.load(optimizer_parameters, map_location=self._device)
                        policy.optim.load_state_dict(state_dict)
        self.policies = policies

    async def update_policy(self):
        for _, policy_wrapper in self.policies.items():
            version = policy_wrapper.version
            if version >= 0:
                policy = policy_wrapper.policy
                policy_wrapper.version = await self._model_sender.sync_model_params(policy, 0, version, self._device)

    async def learn(self):
        while True:
            logger.debug(f'current learner sample queue size: {self.queue.qsize()}.')
            replaybuffer_dict = await self.queue.get()
            policy_name = replaybuffer_dict['policy']
            replaybuffer = replaybuffer_dict['replaybuffer']
            policy_stat = self._stats[policy_name]
            update_policy = True
            if 'buffer' in policy_stat:
                policy_replay_buffer = policy_stat['buffer']
                policy_replay_buffer.update(replaybuffer)

                logger.debug(f'policy {policy_name} buffer size: {len(policy_replay_buffer)}.')
                if len(policy_replay_buffer) < self._batch_size:
                    update_policy = False
                else:
                    replaybuffer = policy_replay_buffer
            if update_policy:
                policy = self.policies[policy_name].policy
                with Timer(name='learn'):
                    losses = policy.update(0, replaybuffer, batch_size=self._batch_size, repeat=self._repeat_per_sample)
                losses = to_numpy(losses)
                step = max([1] + [len(v) for v in losses.values() if isinstance(v, list)])
                policy_stat['gradient_step'] += step
                ret_rms = policy.ret_rms
                logger.info(f'reward rms mean:{ret_rms.mean}, var:{ret_rms.var}, count:{ret_rms.count}')
                loss_stat = policy_stat['loss']
                for k in losses.keys():
                    loss_stat[k].add(losses[k])
                    logger.info(f'{k}:{loss_stat[k].get()}')
                if 'buffer' in policy_stat:
                    policy_stat['buffer'].reset()
                self._learn_count += 1
                if self._learn_count >= self._model_send_count:
                    if policy_name not in self._policy_warmups or self._policy_warmups[policy_name].can_sync_model():
                        await self._model_sender.add_model(policy)
                    self._learn_count = 0
                    if 'checkpoint_file' in self._stats[policy_name]:
                        torch.save(policy.optim.state_dict(), self._stats[policy_name]['checkpoint_file'])
            self.queue.task_done()

    async def sync_policy(self):
        await self._model_sender.send()
