import logging
import grpc
from krl.proto import actor_pb2
from krl.proto import actor_pb2_grpc
from krl.actor.actor import Actor
from typing import Optional
from google.protobuf import empty_pb2
from google.protobuf.json_format import MessageToDict

logger = logging.getLogger(__name__)

class ActorService(actor_pb2_grpc.ActorServicer):
    def __init__(self, actor: Actor) -> None:
        super().__init__()
        self._actor = actor
        self._env_manager = self._actor.get_env_manager()

    async def SendObservation(self, request: actor_pb2.GameObservations, context):
        for obs in request.observations:
            server_id = obs.agent.server_id
            agent_id = obs.agent.agent_id
            obs_state = obs.observation
            env = self._env_manager.get_env(server_id)
            if not env:
                env = self._actor.add_env(server_id)
            agent = env.get_agent(agent_id)
            if not agent:
                agent = self._actor.add_agent(env, agent_id)
            agent.add_obs(obs_state)
        actions = await self._actor.step()
        return actions

    async def ResetServers(self, request: actor_pb2.ServerList, context):
        for server_id in request.server_ids:
            self._env_manager.remove_env(server_id)
        return empty_pb2.Empty()


async def actor_serve(actor: Actor, port: Optional[int] = 50051,
                      read_buffer_size: Optional[int] = 0,
                      write_buffer_size: Optional[int] = 0,
                      ) -> None:
    options = []
    if write_buffer_size:
        options.append(('grpc.max_send_message_length', write_buffer_size))
    if read_buffer_size:
        options.append(('grpc.max_receive_message_length', read_buffer_size))
    server = grpc.aio.server(maximum_concurrent_rpcs=1, options=options)
    actor_pb2_grpc.add_ActorServicer_to_server(ActorService(actor), server)
    server.add_insecure_port('[::]:' + str(port))
    await server.start()
    logger.info(f'actor service start running at port [{port}]...')
    return server