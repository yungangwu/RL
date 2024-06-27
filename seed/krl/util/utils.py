import platform
import time
import grpc

def grpc_server_on(channel) -> bool:
    if platform.system() == 'Linux':
        return True
    else:
        channel.get_state(True)
        time.sleep(1)
        return channel.get_state(True) == grpc.ChannelConnectivity.READY