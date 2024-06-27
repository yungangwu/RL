from krl.actor.envs.env_manager import GameEnvManager

class GameCollector(object):
    def __init__(self, env_manager: GameEnvManager, config: dict) -> None:
        super().__init__()
        self.