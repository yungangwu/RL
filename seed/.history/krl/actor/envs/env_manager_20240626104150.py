
from krl.actor.envs.env import GameEnv

class GameEnvManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._envs = []
        self._envs_map = {}

    def 