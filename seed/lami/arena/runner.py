import grpc
import gym
import time
import random
import numpy as np
import io
import google.protobuf.any_pb2 as any_pb2
from krl.proto import actor_pb2
from krl.proto import actor_pb2_grpc
from tianshou.env import DummyVectorEnv
from lami.gym_env import LamiEnv
from lami.proto import lami_pb2
from lami.utils.utils import ndarray_to_bytes, bytes_to_ndarray
from typing import Any, Tuple, Optional, Callable, List, Dict, Union

class LamiVectorEnv(DummyVectorEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]], **kwargs: Any) -> None:
        super().__init__(env_fns, **kwargs)

    def step(self, action: np.ndarray, id: Optional[Union[int, List[int], np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        assert len(action) == len(id)
        for i, j in enumerate(id):
            self.workers[j].send_action(action[i])
        result = []
        for j in id:
            obs, rew, done, info = self.workers[j].get_result()
            for _obs, _rew, _done, _info in zip(obs, rew, done, info):
                _info["env_id"] = j
                result.append((_obs, _rew, _done, _info))
        obs_list, rew_list, done_list, info_list = zip(*result)
        try:
            obs_stack = np.stack(obs_list)
        except ValueError:
            obs_stack = np.array(obs_list, dtype=object)
        rew_stack, done_stack, info_stack = map(
            np.stack, [rew_list, done_list, info_list]
        )
        if self.obs_rms and self.update_obs_rms:
            self.obs_rms.update(obs_stack)
        return self.normalize_obs(obs_stack), rew_stack, done_stack, info_stack

class Runner:
    def __init__(self, args):
        self.args = args
        options = []
        if args.send_buffer_size > 0:
            options.append(('grpc.max_send_message_length', args.send_buffer_size))
        if args.recv_buffer_size > 0:
            options.append(('grpc.max_receive_message_length', args.recv_buffer_size))
        channel = grpc.insecure_channel(args.server, options=options)
        self._stub = actor_pb2_grpc.ActorStub(channel)
        envs = LamiVectorEnv([lambda: LamiEnv(args.num_player) for _ in range(args.num_env)])
        print(f'args:seed:{args.seed}')
        np.random.seed(args.seed)
        random.seed(args.seed)
        envs.seed(seed=args.seed)
        env_ids = [i for i in range(len(envs))]
        env_names = [f'{args.env_prefix}_{i}' for i in env_ids]
        server_id_to_env_id = {}
        env_id_to_server_id = {}
        for id, name in zip(env_ids, env_names):
            env_id_to_server_id[id] = name
            server_id_to_env_id[name] = id
        self._env_id_to_server_id = env_id_to_server_id
        self._server_id_to_env_id = server_id_to_env_id
        self._envs = envs
        self._env_ids = env_ids
        reset_server_msg = actor_pb2.ServerList(server_ids=env_names)
        self._stub.ResetServers(reset_server_msg)

    def run(self):
        pass

    def _get_server_id(self, env_id):
        return self._env_id_to_server_id[env_id]

    def _get_env_id(self, server_id):
        return self._server_id_to_env_id[server_id]

    def _get_obs_msg(self, obs):
        stat = ndarray_to_bytes(**obs)
        return lami_pb2.LamiState(stat=stat)

    def _action(self, obses, rews, dones, env_ids):
        obs_list = []
        for env_id, obs, rew, done in zip(env_ids, obses, rews, dones):
            any_msg = any_pb2.Any()
            agent_id = obs.pop('player_index')
            state_msg = self._get_obs_msg(obs)
            lami_obs_msg = lami_pb2.LamiObservation(obs=state_msg, reward=rew, done=done)
            any_msg.Pack(msg=lami_obs_msg)
            server_id = self._get_server_id(env_id)
            obs_list.append(actor_pb2.GameObservation(agent=actor_pb2.Agent(server_id=server_id, agent_id=agent_id), observation=any_msg))
        if obs_list:
            start_time = time.time()
            obs_proto = actor_pb2.GameObservations(observations=obs_list)
            actions_proto = self._stub.SendObservation(obs_proto)
            actions = []
            env_ids = []
            for action in actions_proto.actions:
                server_id = action.agent.server_id
                act_proto = lami_pb2.LamiAction()
                action.action.Unpack(act_proto)
                act = bytes_to_ndarray(act_proto.action)['act']
                actions.append(act)
                env_ids.append(self._get_env_id(server_id))
            done_envs = set()
            for env_id in actions_proto.done_envs:
                done_envs.add(self._get_env_id(env_id))
            for agent in actions_proto.done_agents:
                done_envs.add(self._get_env_id(agent.server_id))
            return actions, env_ids, list(done_envs)
