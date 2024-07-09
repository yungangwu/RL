from krl.registry.registry import Registry

policy_registry = Registry()

def registry(id, **kwargs):
    return policy_registry.registry(id, **kwargs)

def make(id, **kwargs):
    return policy_registry.make(id, **kwargs)