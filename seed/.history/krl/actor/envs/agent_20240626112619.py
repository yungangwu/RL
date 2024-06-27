import enum
import google.protobuf.any_pb2 as any_pb2
from typing import Any, Tuple, Optional, Callable, List, Union
from abc import ABC, abstractmethod
from tianshou.policy import BasePolicy
from tianshou.data.batch import Batch
from tianshou.data.buffer.base import ReplayBuffer

@enum.unique
class AgentStatus(enum.Enum):
    NONE = enum.auto()
    WAIT_TO_ACT = enum.auto()
    ACTED = enum.auto()
    REWARDED = enum.auto()

class Agent(ABC):
    def __init__(self, agent_id: int, policy: BasePolicy, agent_config: dict) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.policy = policy
        self.batch = Batch(obs={}, act={}, rew=0, done=False,
                           obs_next={}, info={}, policy={})
        self._buffer = ReplayBuffer(**agent_config['replay_buffer'])
        self.reset()

    def set_env(self, env) -> Any:
        self.env = env

    def get_agent_id(self) -> int:
        return self.agent_id

    def get_buffer(self) -> ReplayBuffer:
        return self._buffer

    def done(self,) -> bool:
        return self.batch.done

    def reset(self):
        self._status = AgentStatus.NONE
        self._raw_obs = None

    def add_obs(self, raw_obs_msg: any_pb2.Any) -> None:
        raw_obs = self.decode_obs(raw_obs_msg)