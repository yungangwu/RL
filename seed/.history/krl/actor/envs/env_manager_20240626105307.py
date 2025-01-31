import logging
from krl.actor.envs.env import GameEnv

logger = logging.getLogger(__name__)

class GameEnvManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._envs = []
        self._envs_map = {}

    def add_env(self, env: GameEnv):
        env_id = env.get_env_id()
        for env_iter in self._envs:
            if env_iter.get_env_id() == env_id:
                logger.warning(f'env id {env_id} has already exists when add env to env manager.')
                return

        self._envs.append(env)
        self._envs_map[env_id] = env
        logger.debug(f'add env [{env_id}].')

    def get_env(self, env_id: str) -> GameEnv:
        return self._envs_map.get(env_id, None)

    def __len__(self) -> int:
        