import time
import logging
from krl.actor.envs.env_manager import GameEnvManager
from typing import Any, Tuple, Optional, Callable, List, Dict
from tianshou.utils.statistics import MovAvg
from collections import defaultdict

logger = logging.getLogger(__name__)

class ObsCollector

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
