import time
import torch
import logging
import numpy as np

from krl.actor.envs.env_manager import GameEnvManager
from krl.actor.envs.agent import Agent, AgentStatus
from krl.actor.envs.env import GameEnv
from typing import Any, Tuple, Optional, Callable, List, Dict
from tianshou.utils.statistics import MovAvg
from tianshou.data.batch import Batch, _create_value
from tianshou.data.utils.converter import to_numpy
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObsCollector(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self.policy_agents = {}

    def __call__(self, agent: Agent) -> Any:
        if agent.get_status() == AgentStatus.WAIT_TO_ACT:
            agent_policy = agent.policy
            agent_list: list = self.policy_agents.get(agent_policy, [])
            agent_list.append(agent)
            self.policy_agents[agent_policy] = agent_list

    def act(self, random: Optional[bool] = False, exploration_noise: Optional[bool] = False):
        action_msgs = []
        for agent_policy, agent_list in self.policy_agents.items():
            if agent_list:
                if random:
                    act = [agent.get_random_actions() for agent in agent_list]
                    for i, agent in enumerate(agent_list):
                        agent.batch.update(policy=Batch(), act=act[i])
                else:
                    num_agent = len(agent_list)

                    # 创建一个初始的策略输入
                    policy_input = _create_value(
                        agent_list[0].batch, num_agent, True
                    )
                    for i, agent in enumerate(agent_list):
                        policy_input[i] = agent.batch

                    # 获取上一次的隐藏状态，为什么还要上一个隐藏状态？
                    last_state = policy_input.policy.pop("hidden_state", None)
                    with torch.no_grad():
                        result = agent_policy(policy_input, last_state)
                    res_policy = result.get("policy", Batch())
                    assert isinstance(res_policy, Batch())
                    state = result.get("state", None)
                    if state is not None:
                        res_policy.hidden_state = state
                    act = to_numpy(result.act)
                    if exploration_noise and agent_policy.training:
                        act = agent_policy.exploration_noise(act, policy_input)
                    for i, agent in enumerate(agent_list):
                        try:
                            data_policy = res_policy[i]
                        except IndexError:
                            data_policy = Batch()
                        agent.batch.update(policy=data_policy, act=act[i])

                action_remap = agent_policy.map_action(act)
                for i, agent in enumerate(agent_list):
                    action_msgs.append(agent.do_action(action_remap[i]))
        return action_msgs

class RewardCollector(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self._ready_agents = []

    # 回调函数，visitor调用时会执行该函数
    def __call__(self, agent: Agent) -> Any:
        if agent.get_status() == AgentStatus.REWARDED:
            self._ready_agents.append(agent)

    def step(self, step_stat):
        step_count = 0
        episode_count = 0
        done_envs = set()
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        for agent in self._ready_agents:
            ep_rew, ep_len, ep_idx = agent.step()
            training = agent.policy.training
            if agent.done():
                step_stat.add_reset_agent(agent)
                agent.reset()
                env: GameEnv = agent.env
                if env.done() and env not in done_envs:
                    done_envs.add(env)
                    step_stat.add_reset_env(env)
                if training:
                    episode_rews.append(ep_rew)
                    episode_lens.append(ep_len)
                    episode_start_indices.append(ep_idx)
                    episode_count += 1
            if training:
                step_count += 1
        for env in done_envs:
            env.reset()
        return step_count, episode_count, episode_rews, episode_lens, episode_start_indices


class GameCollector(object):
    def __init__(self, env_manager: GameEnvManager, config: dict) -> None:
        super().__init__()
        self._env_manager = env_manager
        self._obs_collector = ObsCollector()
        self._rew_collector = RewardCollector()
        self._config = config
        self._step = config.get('step', None)
        self._episode = config.get('episode', None)
        self._random = config.get('random', None)
        self._exploration_noise = config.get('exploration_noise', False)
        self._logger = None
        self._step_count = 0
        self._statistics: Dict[str, MovAvg] = defaultdict(MovAvg)
        self._collect_time = 0
        self._action_time = 0
        self._statistics['rew'] = MovAvg(20)

        if self._step is not None:
            assert self._episode is None, (
                "Only one of n_step or n_episode is allowed in GameCollector."
                f"collect, got n_step={self._step}, n_episode={self._episode}."
            )
            assert self._step > 0
        elif self._episode is not None:
            assert self._episode
        else:
            logger.warning("neither n_step or n_episode is set in collector.")

        self.reset_stat()

    def set_logger(self, logger):
        self._logger = logger

    def reset_stat(self):
        self._stat = {
            'step_count': 0,
            'episode_count': 0,
            'episode_rews': [],
            'episode_lens': [],
            'episode_start_indices': []
        }

    def reset(self):
        self.reset_stat()

    def action(self):
        start_time = time.time()

        obs_collector = self._obs_collector
        obs_collector.reset()

        # calculate agent action
        self._env_manager.visit(obs_collector)
        actions = obs_collector.act(self._random, self._exploration_noise)

        self._statistics['action_time'].add(max(time.time() - start_time, 1e-9))
        return actions

    def get_episode(self):
        return self._episode

    def get_statistics(self):
        return self._statistics

    def step(self, step_stat):
        start_time = time.time()
        rew_collector = self._rew_collector
        rew_collector.reset()

        # 获取相关数据，并更新信息
        self._env_manager.visit(rew_collector)
        step, episode, rews, lens, indices = rew_collector.step(step_stat)

        step_count = self._stat['step_count']
        episode_count = self._stat['episode_count']
        episode_rews = self._stat['episode_rews']
        episode_lens = self._stat['episode_lens']
        episode_start_indices = self._stat['episode_start_indices']

        step_count += step
        episode_count += episode
        episode_rews.extend(rews)
        episode_lens.extend(lens)
        episode_start_indices.extend(indices)

        for rew in rews:
            self._statistics['rew'].add(rew)

        self._stat.update(step_count=step_count, episode_count=episode_count,
                          episode_rews=episode_rews, episode_lens=episode_lens, episode_start_indices=episode_start_indices)

        self._statistics['collect_time'].add(max(time.time() - start_time, 1e-9))

        if (self._step and step_count >= self._step) or (self._episode and episode > 0):
            if episode_count > 0:
                rews, lens, idxs = list(map(
                    np.concatenate, [episode_rews, episode_lens, episode_start_indices]
                ))

                def add_done_agents(agent):
                    if agent.done():
                        step_stat.add_samples(agent)
                self._env_manager.visit(lambda agent: add_done_agents(agent))
            else:
                rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
                self._env_manager.visit(lambda agent: step_stat.add_samples(agent))

            result = {
                "n/ep": episode_count,
                
            }