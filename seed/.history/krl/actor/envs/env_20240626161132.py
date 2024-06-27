import logging
from abc import ABC, abstractmethod
from krl.actor.envs.agent import Agent

logger = logging.getLogger(__name__)

class GameEnv(ABC):
    def __init__(self, env_id: str) -> None:
        super().__init__()
        self._env_id = env_id
        self._agents = []
        self._map_agents = {}

    # 将env与agent进行绑定
    def add_agent(self, agent: Agent):
        agent_id = agent.get_agent_id()
        for agent_iter in self._agents:
            if agent_iter.get_agent_id() == agent_id:
                logger.warning(f'agent id {agent_id} has already exist when add agent to env {self._env_id}.')
                return
        agent.set_env(self)
        self._agents.append(agent)
        self._map_agents[agent_id] = agent
        logger.debug(f'env [{self._env_id}] add agent []')


    def get_env_id(self,):
        return self._env_id

    def get_num_agents(self,):
        return len(self._agents)