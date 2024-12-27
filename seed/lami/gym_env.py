import gym
import numpy as np
from lami.game import Game

class LamiEnv(gym.Env):
    def __init__(self, game_name='Lami', render=False):
        super().__init__()
        self._game = Game(render)
        setattr(self, 'action_space', gym.Space(shape=(3, 4, 14), dtype=np.float32))

    def reset(self):
        return self._game.reset()

    def game(self):
        return self._game

    def step(self, action):
        desk_id, move = self._game.action_from_numpy(action)
        obs, rews, dones, infos = self._game.step(desk_id, move)
        return obs, rews, dones, infos

    def seed(self, seed=None):
        return self._game.seed(seed)
