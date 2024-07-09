import asyncio
import logging
import grpc

from krl.proto import model_pb2, model_pb2_grpc
from krl.util.timer import Timer
from krl.util.utils import grpc_server_on, sync_policy

logger = logging.getLogger(__name__)

class ModelSync(object):
    def __init__(self, address: str = None, write_buffer_size: int = 0, read_buffer_size: int = 0) -> None:
        super().__init__()
        self.stub = self._init_client(address, write_buffer_size, read_buffer_size) if address else None

    def _init_client(self, address: str, write_buffer_size: int, read_buffer_size: int):
        options = []
        if write_buffer_size > 0:
            options.append(('grpc.max_send_message_length', write_buffer_size))
        if read_buffer_size > 0:
            options.append(('grpc.max_receive_message_length', read_buffer_size))

        channel = grpc.aio.insecure_channel(address, options=options)
        if grpc_server_on(channel):
            stub = model_pb2_grpc.ModelStub(channel)
            return stub
        else:
            logger.error(f'connect to [{address}] failed.')

    async def sync_model_params(self, policy, from_version, to_version, device: str):
        if self.stub:
            name = getattr(policy, 'policy_name')
            logger.info(f'try to fetch [{name}] model parameters from version [{from_version}] to [{to_version}].')
            return await sync_policy(self.stub, policy, from_version, to_version, device)
        return from_version

    async def get_model_latest_version(self, policy):
        if self.stub:
            name = getattr(policy, 'policy_id')
            version_msg = model_pb2.ModelLatestVersionRequest(name=name)
            result = await self.stub.GetLatestVersion(version_msg)
            return result.version
        return 0