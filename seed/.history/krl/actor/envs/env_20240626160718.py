from abc import ABC, abstractmethod
from krl.actor.envs.agent import Agent

class GameEnv(ABC):
    def __init__(self, env_id: str) -> None:
        super().__init__()
        self._env_id = env_id
        self._agents = []
        self._map_agents = {}

    def add_agent(self, agent: Agent):
        agent_id = agent.get_agent_id()
        for agent_iter in self._agents:
            if agent_iter.get_agent_id() == agent_id:
                logger.


    def get_env_id(self,):
        return self._env_id

    def get_num_agents(self,):
        return len(self._agents)