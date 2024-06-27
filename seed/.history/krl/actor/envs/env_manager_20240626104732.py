
from krl.actor.envs.env import GameEnv

class GameEnvManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._envs = []
        self._envs_map = {}

    def add_env(self, env: GameEnv):
        env_id = env.get_env_id()
        