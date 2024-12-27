import logging
import lami.protobuf.lami_pb2 as lami_pb2
import google.protobuf.any_pb2 as any_pb2
import numpy as np
from typing import Any, Union
from lami.utils import ndarray_to_bytes, bytes_to_ndarray
from krl.registry.policy_registry import registry

logger = logging.getLogger(__name__)

class LamiInfo:
    def __init__(self, legal_actions):
        self.legal_actions_num = legal_actions.shape[0]
        self.legal_actions = legal_actions

    def __getitem__(self, index):
        if isinstance(index, str):
            return getattr(self, index)

    def __setitem__(self, key, value) -> Any:
        if key == 'legal_actions':
            self.__init__(value)

    def __getstate__(self):
        state = ndarray_to_bytes(legal_actions=self.legal_actions)
        return state

    def __setstate__(self, state):
        kargs = bytes_to_ndarray(state)
        self.__init__(**kargs)

class LamiAgent(Agent):
    def decode_obs(self, raw_obs: any_pb2.Any):
        obs_msg = lami_pb2.LamiObservation()
        raw_obs.Unpack(obs_msg)
        state_msg = bytes_to_ndarray(obs_msg.obs.stat)
        # this code is very slow for it will load file
        state = {k: v for k, v in state_msg.items()}
        legal_actions = state.pop('legal_actions')
        obs = {
            'obs': state,
            'rew': obs_msg.reward,
            'done': obs_msg.done,
            'legal_actions': legal_actions,
        }
        return obs

    def get_info(self):
        return self.info

    def preprocess(self, raw_obs):
        self.info = LamiInfo(raw_obs['legal_actions'])
        return raw_obs['obs']

    def encode_action(self, act):
        action_msg = lami_pb2.LamiAction(action=ndarray_to_bytes(act=act))
        any_msg = any_pb2.Any()
        any_msg.Pack(action_msg)
        return any_msg

    def get_random_actions(self):
        return self.env.action_space().sample()

    def do_action(self, act: Union[np.array, list]):
        return super().do_action(act)

    def reward(self, prev_raw_obs, raw_obs):
        rew, done = raw_obs['rew'], raw_obs['done']
        return rew, done

registry('lami_agent', entry_point='lami.agent.lami_agent:LamiAgent')