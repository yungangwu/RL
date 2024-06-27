# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from krl.proto import actor_pb2 as krl_dot_proto_dot_actor__pb2


class ActorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendObservation = channel.unary_unary(
                '/krl.proto.actor.Actor/SendObservation',
                request_serializer=krl_dot_proto_dot_actor__pb2.GameObservations.SerializeToString,
                response_deserializer=krl_dot_proto_dot_actor__pb2.GameActions.FromString,
                )
        self.ResetServers = channel.unary_unary(
                '/krl.proto.actor.Actor/ResetServers',
                request_serializer=krl_dot_proto_dot_actor__pb2.ServerList.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class ActorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendObservation(self, request, context):
        """Send Observations Of Game
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ResetServers(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ActorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'SendObservation': grpc.unary_unary_rpc_method_handler(
                    servicer.SendObservation,
                    request_deserializer=krl_dot_proto_dot_actor__pb2.GameObservations.FromString,
                    response_serializer=krl_dot_proto_dot_actor__pb2.GameActions.SerializeToString,
            ),
            'ResetServers': grpc.unary_unary_rpc_method_handler(
                    servicer.ResetServers,
                    request_deserializer=krl_dot_proto_dot_actor__pb2.ServerList.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'krl.proto.actor.Actor', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Actor(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendObservation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/krl.proto.actor.Actor/SendObservation',
            krl_dot_proto_dot_actor__pb2.GameObservations.SerializeToString,
            krl_dot_proto_dot_actor__pb2.GameActions.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ResetServers(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/krl.proto.actor.Actor/ResetServers',
            krl_dot_proto_dot_actor__pb2.ServerList.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
