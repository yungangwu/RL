import logging
import os

from krl.actor.envs.agent import Agent
from krl.actor.envs.env_manager import GameEnvManager
from krl.actor.collector.collector import GameCollector
from krl.actor.collector.sample_sender import SampleSender
from krl.actor.envs.env import GameEnv
from krl.proto import actor_pb2
from krl.actor.model_sync import ModelSync
from krl.util.utils import PolicyWrapper
from krl.registry.policy_registry import policy_registry
from krl.registry.env_registry import env_registry
from krl.registry.agent_registry import agent_registry
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.utils import BasicLogger
from tianshou.policy.base import BasePolicy
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

logger = logging.getLogger(__name__)

class StepStatistics(object):
    def __init__(self, max_samples_size: int, send_episode_num: int, sample_sender: SampleSender) -> None:
        super().__init__()
        self.reset()
        self._max_samples_size = max_samples_size
        self._episode = send_episode_num
        self._samples = defaultdict(dict)
        self._sample_sender = sample_sender

    def reset(self):
        self._reset_agents = []
        self._reset_envs = []

    def add_reset_agent(self, agent: Agent):
        self._reset_agents.append(
            actor_pb2.Agent(server_id=agent.env.get_env_id(), agent_id=agent.get_agent_id())
        )

    def add_reset_env(self, env: GameEnv):
        self._reset_envs.append(env.get_env_id())

    def get_reset_envs(self, ):
        return self._reset_envs

    def get_reset_agents(self):
        return self._reset_agents

    def add_samples(self, agent: Agent):
        buffer = agent.get_buffer()
        agent_policy = agent.policy

        # 如果在训练中，将agent的buffer更新进去，这里的buffer就是一批新数据
        if agent_policy.training and len(buffer) > 0:
            logger.debug(f"send agent [{agent.get_agent_id()} of env [{agent.env.get_env_id()}]'s sample.")
            policy_name = getattr(agent_policy, 'policy_name')
            policy_buffer_dict = self._samples[policy_name]
            if 'buffer' in policy_buffer_dict:
                policy_buffer = policy_buffer_dict['buffer']
            else:
                policy_buffer = ReplayBuffer(size=self._max_samples_size)
                policy_buffer_dict = {
                    'buffer': policy_buffer,
                    'count': 0
                }
            policy_buffer.update(buffer)
            policy_buffer_dict['count'] += 1
            self._samples[policy_name] = policy_buffer_dict
            total_episode = policy_buffer_dict['count']
            logger.debug(f'total sample size: {len(policy_buffer)} episode: {total_episode}k.')
        buffer.reset()

    async def send_samples(self):
        remove_buffers = []
        for policy_name, buffer_info in self._samples.items():
            if not self._episode or (self._episode and buffer_info['count'] >= self._episode):
                replay_buffer = buffer_info['buffer']
                total_samples = len(replay_buffer)
                campact_replay_buffer = ReplayBuffer(size=total_samples)
                campact_replay_buffer.update(replay_buffer)
                # 等待发送数据
                await self._sample_sender.add_replaybuffer(policy_name, campact_replay_buffer)
                remove_buffers.append(policy_name)

        for policy_name in remove_buffers:
            self._samples.pop(policy_name)


class Actor(object):
    def __init__(self, config: dict):
        super().__init__()

        # env有多个并行的，用于收集数据
        env_manager = GameEnvManager()
        collector = GameCollector(env_manager, config.get('collect', {}))

        # 如果summary path，则加载tensorboard的画图
        if 'logdir' in config:
            log_path = os.path.join(
                config['logdir'], config['game_name'], 'ppo'
            )
            writer = SummaryWriter(log_path)
            self.logger = BasicLogger(writer)
            collector.set_logger(self.logger)
        else:
            self.logger = None

        self.env_manager = env_manager
        self.collector = collector
        self.config = config

        self._sample_sender = SampleSender(**config.get('sender', {}))

        max_samples_size = config.get('max_sample_size', 1024)
        episode_to_send = collector.get_episode()
        self._step_stat = StepStatistics(max_samples_size, episode_to_send, self._sample_sender)

        self._policy_distribution_fn = config.get('policy_distribution', None)
        self._opponent_policy_update_strategy = config.get('opponent_policy_update_strategy', None)

        assert 'env' in config and "please set 'env' section in config file."
        self._env_config = config['env']

        assert 'agent' in config and "please set 'agent' section in config file."
        self._agent_config = config['agent']
        self._device = config.get('device', 'cpu')

        model_sync_config = config.get('model_sync', {})
        self._sync_model_step = model_sync_config.pop('step', None)
        self._sync_model_episode = model_sync_config.pop('episode', None)
        self._sync_model_counter = 0
        self._model_sync = ModelSync(**model_sync_config)

        self.create_policy(config.get('policy', None))

    def create_policy(self, policy_config: dict):
        # assert 如何条件为假，则抛出异常
        assert policy_config and "please set policy params in section 'policy' of config file."
        self.policies = {}
        for policy_name, policy_options in policy_config.items():
            policy_id = policy_options.pop('id', policy_name)
            assert policy_name not in self.policies
            version = policy_options.pop('version', -1)
            training = policy_options.pop('training', True)
            policy = policy_registry.make(policy_id, device=self._device, **policy_options)
            setattr(policy, 'policy_name', policy_name)
            setattr(policy, 'policy_id', policy_id)
            policy.set_training(training)
            logger.info(f'{policy_name} training: {training}')
            self.policies[policy_name] = PolicyWrapper(policy=policy, version=version)

    async def sync_policy(self, ):
        for _, policy_wrapper in self.policies.items():
            version = policy_wrapper.version
            policy = policy_wrapper.policy
            policy_wrapper.version = await self._model_sync.sync_model_params(policy, 0, version, self._device) \
                if version >= 0 else 0

    async def step(self):
        if (self._sync_model_step and self._sync_model_counter >= self._sync_model_step) or \
            (self._sync_model_episode and self._sync_model_counter >= self._sync_model_episode):
            for _, policy_wrapper in self.policies.items():
                if policy_wrapper.policy.training:
                    policy_wrapper.version = await self._model_sync.sync_model_params(policy_wrapper.policy,
                                                                                        policy_wrapper.version,
                                                                                        0,
                                                                                        self._device)
                else:
                    if self._opponent_policy_update_strategy:
                        policy = policy_wrapper.policy
                        latest_version = await self._model_sync.get_model_latest_version(policy)
                        version = self._opponent_policy_update_strategy(policy_wrapper.policy, latest_version)
                        name = getattr(policy, 'policy_name')
                        logger.info(f'opponent {name} version {version}, latest version: {latest_version}')
                        policy_wrapper.version = await self._model_sync.sync_model_params(policy, policy_wrapper.version, version, self._device)

            self._sync_model_counter = 0

        self._step_stat.reset()
        step, episode = self.collector.step(self._step_stat)
        self._sync_model_counter = self._sync_model_counter + step if self._sync_model_step else self._sync_model_counter + episode
        done_envs = self._step_stat.get_reset_envs()
        done_agents = self._step_stat.get_reset_agents()

        if done_envs:
            logger.debug(f'done envs: {done_envs}.')

        if done_agents:
            logger.debug(f'done agents: {done_agents}.')

        await self._step_stat.send_samples()

        agent_actions = self.collector.action()
        actions_msg = actor_pb2.GameActions(actions=agent_actions, done_envs=done_envs, done_agents=done_agents)
        return actions_msg

    async def send_samples(self):
        await self._sample_sender.send()

    def _new_env(self, env_id: str) -> GameEnv:
        if isinstance(self._env_config, str):
            env = env_registry.make(self._env_config, env_id=env_id)
        elif isinstance(self._env_config, dict):
            env = env_registry.make(list(self._env_config.keys())[0], env_id=env_id, **(list(self._env_config.values())[0]))
        else:
            raise ValueError('env config must be str or dict.')
        return env

    def _new_agent(self, agent_id: int, policy: BasePolicy) -> Agent:
        if isinstance(self._agent_config, dict):
            agent_name = list(self._agent_config.keys())[0]
            agent_config = list(self._agent_config.values())[0]
            agent = agent_registry.make(agent_name, agent_id,
                                        policy=policy,
                                        agent_config=agent_config)
        else:
            raise ValueError('env config must be str or dict.')
        return agent

    def add_env(self, env_id: str):
        env = self._new_env(env_id)
        self.env_manager.add_env(env)
        return env

    def get_env_manager(self) -> GameEnvManager:
        return self.env_manager

    def add_agent(self, env: GameEnv, agent_id: int):
        if self._policy_distribution_fn:
            # 根据env_id和agent_id，获取policy_name
            policy_name = self._policy_distribution_fn(
                env.get_env_id(), agent_id
            )
            assert policy_name in self.policies and f'policy {policy_name} is not created.'
            logger.debug(f'env {env.get_env_id()} agent {agent_id} use policy {policy_name}.')
            policy_wrapper = self.policies[policy_name]
        else:
            policy_name = list(self.policies.keys())[0]
            logger.debug(f'env {env.get_env_id()} agent {agent_id} use policy {policy_name}.')
            policy_wrapper = self.policies[policy_name]
        agent = self._new_agent(agent_id, policy_wrapper.policy)
        env.add_agent(agent)
        return agent