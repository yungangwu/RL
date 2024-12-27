from abc import ABC, abstractmethod

class Policy(ABC):
    """
    Policy 基类，用于定义策略的接口，所有子类都需要继承该类并实现接口。
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def choose_action(self, state):
        """
        定义选择动作的方法，需要子类实现。
        :param state: 环境状态
        :return: 动作
        """
        pass

    @abstractmethod
    def learn(self, experience):
        """
        定义更新策略的方法，需要子类实现。
        :param experience: 经验（如状态、动作、奖励等）
        """
        pass

    def save(self, file_path):
        """
        可选：保存策略到文件。
        :param file_path: 保存文件路径
        """
        print(f"Saving policy to {file_path}...")

    def load(self, file_path):
        """
        可选：从文件加载策略。
        :param file_path: 加载文件路径
        """
        print(f"Loading policy from {file_path}...")
