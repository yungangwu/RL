import logging
import os
from krl.actor.envs.env_manager import GameEnvManager
from krl.actor.collector.collector import GameCollector
from tianshou.utils import BasicLogger
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Actor(object):
    def __init__(self, config: dict):
        super().__init__()

        # env有多个并行的，用于收集数据
        env_manager = GameEnvManager()
        collector = GameCollector(env_manager, config.get('collect', {}))

        # 如果summary path，则加载tensorboard的画图
        if 'logdir' in config:
            log_path = os.path.join(
                config['logdir'], config['game_name'], 'ppo'
            )
            writer = SummaryWriter(log_path)
            self.logger = BasicLogger(writer)
            collector.set_logger(self.logger)
        else:
            self.logger = None

        self.env_manager = env_manager
        self.collector = collector
        self.config = config

        self._sample_sender = SampleSender(**config.get('sender', {}))