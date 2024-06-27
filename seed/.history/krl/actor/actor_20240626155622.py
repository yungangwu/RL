from krl.actor.envs.env_manager import
from krl.actor.envs.agent import 

class Actor(object):
    def __init__(self, config: dict):
        super().__init__()

        # env有多个并行的，用于收集数据
        env_manager = GameEnvManager()
        collector = GameCollector()