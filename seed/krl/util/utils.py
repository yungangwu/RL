import platform
import time
import torch
import io
import grpc
from copy import deepcopy
from krl.proto import model_pb2
from tianshou.policy.base import BasePolicy

def override_config(src_config: dict, override_config: dict):
    result = deepcopy(src_config)
    if override_config:
        for k, v in override_config.items():
            result[k] = deepcopy(v)
    return result

def grpc_server_on(channel) -> bool:
    if platform.system() == 'Linux':
        return True
    else:
        channel.get_state(True)
        time.sleep(1)
        return channel.get_state(True) == grpc.ChannelConnectivity.READY

async def sync_policy(stub, policy, from_version, to_version, device: str):
    if stub:
        name = getattr(policy, 'policy_id')
        version_msg = model_pb2.ModelVersion(name=name, base_version=from_version, request_version=to_version)
        result = await stub.FetchModel(version_msg) # result是grpc的返回结果
        assert name == result.name
        if from_version != result.new_version:
            if result.model_pramas.data:
                bio = io.BytesIO(result.model_pramas.data) # grpc结果中定义的model_pramas，将其转化为文件
                state_dict = torch.load(bio, map_location=device)
                print(f'policy [{name}] sync finish.')
                policy.load_state_dict(state_dict)
            else:
                raise RuntimeError(f'fetch policy [{name}] of version [{to_version}] failed.')
        return result.new_version

class PolicyWrapper(object):
    def __init__(self, policy: BasePolicy, version: int = 0) -> None:
        super().__init__()
        self.policy = policy
        self.version = version