import grpc
import gym
import time
import numpy as np
import io
from lami.arena.runner import Runner
from krl.proto import actor_pb2

class Arena(Runner):
    def run(self):
        envs = self._envs
        env_len = len(envs)
        obs = envs.reset()
        rew = np.full(env_len, 0, dtype=np.float32)
        done = np.full(env_len, False, dtype=np.bool)
        env_ids = self._env_ids.copy()
        scores = [0 for _ in range(self.args.num_player)]
        end_envs = set()
        while True:
            actions, env_ids, done_env_ids = self._action(
                obs, rew, done, env_ids
            )
            if env_ids:
                obs, rew, done, info = envs.step(np.stack(actions), env_ids)
                for i, d in enumerate(done):
                    if d:
                        player_index = obs[i]['player_index']
                        scores[player_index] += rew[i]
                        end_env_id = info[i]['env_id']
                        end_envs.add(end_env_id)
                env_ids.clear()
                for _info in info:
                    env_ids.append(_info['env_id'])
            else:
                assert len(end_envs) == len(self._env_ids)
                break

        env_names = list(self._env_id_to_server_id.values())
        reset_server_msg = actor_pb2.ServerList(server_ids=env_names)
        self._stub.Reset(reset_server_msg)
        return scores
