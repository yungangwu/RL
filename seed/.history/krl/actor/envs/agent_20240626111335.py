import enum
from typing import Any, Tuple, Optional, Callable, List, Union
from abc import ABC, abstractmethod
from tianshou.policy import BasePolicy

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
        