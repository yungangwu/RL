from abc import ABC, abstractmethod

class Player(ABC):
    @abstractmethod
    def set_player_ind(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_player(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_action(self, *args, **kwargs):
        pass

    @abstractmethod
    def __str__(self, *args, **kwargs) -> str:
        pass