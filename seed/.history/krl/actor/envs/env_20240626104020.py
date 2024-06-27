from abc import ABC, abstractmethod

class GameEnv(ABC):
    def __init__(self) -> None:
        super().__init__()