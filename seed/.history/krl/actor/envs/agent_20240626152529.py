import enum
import numpy as np
import google.protobuf.any_pb2 as any_pb2
from typing import Any, Tuple, Optional, Callable, List, Union
from abc import ABC, abstractmethod
from krl.proto import actor_pb2
from krl.proto import actor_pb2_grpc
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
        obs = self.preprocess(raw_obs)
        status = self.get_status()
        if status == AgentStatus.NONE:
            info = self.get_info()
            self.batch.update(obs=obs, info=info)
            self._set_status(AgentStatus.WAIT_TO_ACT)
        elif status == AgentStatus.ACTED:
            rew, done = self.reward(self._raw_obs, raw_obs)
            self.batch.update(obs_next=obs, rew=rew, done=done)
            self._set_status(AgentStatus.REWARDED)
        self._raw_obs = raw_obs

    def _set_status(self, status: AgentStatus) -> None:
        self._status = status

    def get_status(self) -> AgentStatus:
        return self._status

    @abstractmethod
    def get_random_actions(self):
        pass

    @abstractmethod
    def encode_action(self, act) -> any_pb2.Any:
        pass

    def do_action(self, act: Union[np.array]):
        self._set_status(AgentStatus.ACTED)
        action_msg = self.encode_action(act)
        agent_msg = actor_pb2.Agent(server_id=self.env.get_env_id(), agent_id=self.agent_id)
        return actor_pb2.GameAction(agent=agent_msg, action=action_msg)

    @abstractmethod
    def decode_obs(self, raw_obs: any_pb2.Any):
        pass

    def preprocess(self, raw_obs):
        return raw_obs, {}

    @abstractmethod
    def reward(self, prev_raw_obs, raw_obs)