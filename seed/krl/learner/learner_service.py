import grpc
import logging
from krl.proto import learner_pb2
from krl.proto import learner_pb2_grpc
from krl.learner.learner import Learner
from typing import Optional
from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

class LearnerService(learner_pb2_grpc.LearnerServicer):
    def __init__(self, learner: Learner) -> None:
        super().__init__()
        self._learner = learner

    async def SendSamples(self, request: learner_pb2.Sample, context):
        await self._learner.add_samples(request)
        return empty_pb2.Empty()

async def learner_serve(learner: Learner, args) -> None:
    options = []
    if args.write_buffer_size:
        options.append(('grpc.max_send_message_length', args.write_buffer_size))
    if args.read_buffer_size:
        options.append(('grpc.max_receive_message_length', args.read_buffer_size))
    server = grpc.aio.server(maximum_concurrent_rpcs=1, options=options)
    learner_pb2_grpc.add_LearnerServicer_to_server(LearnerService(learner), server)
    server.add_insecure_port('[::]:' + str(args.port))
    await server.start()
    logger.debug(f'learner service start running at port [{args.port}]...')
    return server