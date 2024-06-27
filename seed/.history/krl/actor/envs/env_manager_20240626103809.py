
from krl.actor.envs.en

class GameEnvManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._envs = []
        self._envs_map = {}
