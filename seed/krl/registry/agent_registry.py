from krl.registry.registry import Registry

agent_registry = Registry()

def register(id, **kwargs):
    return agent_registry.register(id, **kwargs)

def make(id, **kwargs):
    return agent_registry.make(id, **kwargs)