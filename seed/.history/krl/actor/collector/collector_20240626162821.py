from krl.actor.envs.env_manager import GameEnvManager

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
        self.