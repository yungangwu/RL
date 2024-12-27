import gym
import numpy as np
from krl.actor.envs.env import GameEnv
from krl.registry.env_registry import register

class LamiEnv(GameEnv):
    def __init__(self, env_id: str) -> None:
        super().__init__(env_id)

register('lami_env', entry_point='lami.env.lami_env:LamiEnv')
