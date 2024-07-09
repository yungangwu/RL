import grpc
import io
import asyncio
import h5py
import torch
import logging
from tianshou.data import ReplayBuffer
from tianshou.data.utils.converter import to_hdf5
from tianshou.policy.base import BasePolicy
from krl.proto import model_pb2, model_pb2_grpc
from krl.util.utils import grpc_server_on, sync_policy
from krl.util.timer import Timer

logger = logging.getLogger(__name__)

def buffer_to_hdf5(buffer: ReplayBuffer):
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        to_hdf5(buffer.__dict__, f)
    return bio

class ModelSender(object):
    def __init__(self,
                 max_queue_size: int = 8,
                 address: str = None,
                 write_buffer_size: int = 0,
                 read_buffer_size: int = 0) -> None:
        super().__init__()
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.stub = self._init_client(address, write_buffer_size, read_buffer_size) if address else None

    def _init_client(self, address: str, write_buffer_size: int, read_buffer_size: int):
        options = []
        options.append(('grpc.max_send_message_length', write_buffer_size))
        options.append(('grpc.max_receive_message_length', read_buffer_size))
        channel = grpc.aio.insecure_channel(address, options=options)
        if grpc_server_on(channel):
            stub = model_pb2_grpc.ModelStub(channel)
            return stub
        else:
            logger.error(f'connect to [{address}] failed.')

    async def add_model(self, policy: BasePolicy):
        bio = io.BytesIO()
        torch.save(policy.state_dict(), bio)
        name = getattr(policy, 'policy_id')
        await self.queue.put({'name': name, 'bio_data': bio})

    async def sync_model_params(self, policy, from_version, to_version, device: str):
        if self.stub:
            name = getattr(policy, 'policy_id')
            logger.debug(f'try to fetch [{name}] model params from [{from_version}] to [{to_version}].')
            return await sync_policy(self.stub, policy, from_version, to_version, device)
        return from_version

    async def send(self):
        while True:
            logger.debug(f'current model send queue size: {self.queue.qsize()}')
            bio_dict = await self.queue.get()
            if self.stub:
                bio = bio_dict['bio_data']
                name = bio_dict['name']
                with Timer(name='sample encode'):
                    data = bio.getvalue()
                    model_params_msg = model_pb2.ModelParams(name=name, data=data)
                with Timer(name='model send'):
                    await self.stub.SendModel(model_params_msg)
            self.queue.task_done()