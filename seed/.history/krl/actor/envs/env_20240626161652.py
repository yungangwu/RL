import logging
from typing import Any, Tuple, Optional, Callable, List, Union
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
        logger.debug(f'env [{self._env_id}] add agent [{agent_id}].')

    def get_env_id(self,) -> str:
        return self._env_id

    def get_num_agents(self,) -> int:
        return len(self._agents)

    def done(self) -> bool:
        for agent in self._agents:
            if not agent.done():
                return False

        return True

    def get_agent(self, agent_id: int) -> Agent:
        return self._map_agents.get(agent_id, None)

    def reset(self):
        for agent in self._agents:
            agent.reset()

    def clear(self):
        self._agents = []
        self._map_agents = {}

    def visit(self, visitor: Callable[[Agent], bool]) -> bool