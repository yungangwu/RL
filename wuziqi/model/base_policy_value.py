from abc import ABC, abstractmethod

class BasePolicyValue(ABC):
    @abstractmethod
    def get_move(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_policy_param(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_model(self, *args, **kwargs) -> str:
        pass