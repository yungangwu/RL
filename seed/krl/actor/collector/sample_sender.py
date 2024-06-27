import io
import logging
import asyncio
import grpc
import h5py
from krl.util.timer import Timer
from krl.util.utils import grpc_server_on
from krl.proto import learner_pb2_grpc, learner_pb2
from tianshou.data.buffer.base import ReplayBuffer
from tianshou.data.utils.converter import to_hdf5

logger = logging.getLogger(__name__)

def buffer_to_hdf5(buffer: ReplayBuffer):
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        to_hdf5(buffer.__dict__, f)
    return bio

class SampleSender(object):
    def __init__(self, max_queue_size: int = 8, learner_address_config: dict = None) -> None:
        super().__init__()
        self.queue = asyncio.Queue(max_queue_size)
        self.stubs = {}
        if learner_address_config:
            for policy_name, config in learner_address_config.items():
                address = config.get('address', None)
                if address:
                    logger.info(f'[{policy_name}] connect to learner server: {address}')
                    self.stubs[policy_name] = self._init_client(**config) # 每个policy对应一个clinet，用于执行数据发送

    def _init_client(self, address: str, write_buffer_size: int = 0, read_buffer_size: int = 0):
        options = []
        if write_buffer_size > 0:
            options.append(('grpc.max_send_message_length', write_buffer_size))
        if read_buffer_size > 0:
            options.append(('grpc.max_receive_message_length', read_buffer_size))
        channel = grpc.aio.insecure_channel(address, options=options)
        if grpc_server_on(channel):
            stub = learner_pb2_grpc.LearnerStub(channel)
            return stub
        else:
            logger.error(f'connect to [{address}] failed.')

    async def add_replaybuffer(self, policy_name, replaybuffer: ReplayBuffer):
        await self.queue.put({'policy': policy_name, 'replay_buffer': replaybuffer})

    async def send(self):
        while True:
            logger.debug(f'current send queue size: {self.queue.qsize()}')
            policy_replaybuffer = await self.queue.get()
            replaybuffer = policy_replaybuffer['replay_buffer']
            policy_name = policy_replaybuffer['policy']
            if policy_name in self.stubs and self.stubs[policy_name]:
                stub = self.stubs[policy_name]
                with Timer(name='sample encode'):
                    data = buffer_to_hdf5(replaybuffer).getvalue()
                    sample_msg = learner_pb2.Sample(policy_name=policy_name, data=data)
                with Timer(name='sample send'):
                    await stub.SendSamples(sample_msg)
            self.queue.task_done()