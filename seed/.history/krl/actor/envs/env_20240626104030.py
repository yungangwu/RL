from abc import ABC, abstractmethod

class GameEnv(ABC):
    def __init__(self, env_id) -> None:
        super().__init__()