from krl.registry.registry import Registry

env_registry = Registry()


def register(id, **kwargs):
    return env_registry.registry(id, **kwargs)

def make(id, **kwargs):
    return env_registry.make(id, **kwargs)