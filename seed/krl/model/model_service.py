import logging
import io
import grpc
from krl.proto import model_pb2, model_pb2_grpc
from krl.model.model_save import ModelCollection, ModelSaver
from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

class ModelSaveService(model_pb2_grpc.ModelServicer):
    def __init__(self, models: ModelSaver) -> None:
        super().__init__()
        self._models = models

    async def SendModel(self, request: model_pb2.ModelParams, context):
        name = request.name
        version = self._models[name].add_model(request.data)
        await self._models.save(name=name, version=version, data=request.data)
        logger.info(f'[{name}] add new model parameters of version {version}')
        return empty_pb2.Empty()

    async def FetchModel(self, request: model_pb2.ModelVersion, context):
        name = request.name
        collection: ModelCollection = self._models[name]
        latest_version = collection.get_latest_version()
        base_version = request.base_version
        request_version = request.version if request.version > 0 else latest_version
        if base_version == request_version:
            return model_pb2.FetchModelResult(name=name, new_version=base_version)
        else:
            data = await self._models.get_model(name, request_version)
            logger.debug(f'[{name}] fetch model parameters of version [{request_version}].')
            params = model_pb2.ModelParams(data=data.getvalue() if data else model_pb2.ModelParams())
            return model_pb2.FetchModelResult(name=name, new_version=request_version, model_params=params)

    async def GetLatestVersion(self, request: model_pb2.ModelLatestVersionRequest, context):
        name = request.name
        collection: ModelCollection = self._models[name]
        latest_version = collection.get_latest_version()
        logger.debug(f'[{name}] get latest version [{latest_version}].')
        return model_pb2.ModelLatestVersionResponse(version=latest_version)

async def model_serve(models: ModelSaver, args) -> None:
    options = []
    if args.write_buffer_size:
        options.append(('grpc.max_send_message_length', args.write_buffer_size))
    if args.read_buffer_size:
        options.append(('grpc.max_receive_message_length', args.read_buffer_size))
    server = grpc.aio.server(maximum_concurrent_rpcs=1, options=options)
    model_pb2_grpc.add_ModelServicer_to_server(ModelSaveService(models), server)
    await server.start()
    logger.debug(f'model service start running at port [{args.port}].')
    return server