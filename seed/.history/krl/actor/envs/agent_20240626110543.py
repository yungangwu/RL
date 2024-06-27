import enum
from typing import Any, Tuple, Optional, Callable, List, Union
from abc import ABC, abstractmethod

@enum.unique
class AgentStatus(enum.Enum):
    NONE = enum.auto()
    WAIT_TO_ACT = enum.auto()
    