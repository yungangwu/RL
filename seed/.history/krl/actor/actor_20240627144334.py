import logging
from krl.actor.envs.env_manager import GameEnvManager
from krl.actor.collector.collector import GameCollector

logger = logging.getLogger(__name__)

class Actor(object):
    def __init__(self, config: dict):
        super().__init__()

        # env有多个并行的，用于收集数据
        env_manager = GameEnvManager()
        collector = GameCollector(env_manager, config.get('collect', {}))

        if 'log'