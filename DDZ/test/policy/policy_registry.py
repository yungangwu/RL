
import importlib

def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

class PolicySpec(object):
   
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs



    def make(self, **kwargs):
        if self.entry_point is None:
            raise ValueError('attempting to make policy [{}] without entry point.'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            policy = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            policy = cls(**_kwargs)
        return policy

    def __repr__(self):
        return "policy spec:({})".format(self.id)



class PolicyRegistery(object):

    def __init__(self):
        self.policy_specs = {}

    def register(self, id, **kwargs):
        if id in self.policy_specs:
            raise RuntimeError('duplicate register policy id: {}'.format(id))
        self.policy_specs[id] = PolicySpec(id, **kwargs)

    def make(self, id, **kwargs):
        if id not in self.policy_specs:
            raise ValueError("attempt to make policy id:{}, which is not registered.".format(id))
  
        spec = self.policy_specs[id]
        policy = spec.make(**kwargs)

        if hasattr(policy, "_evaluate"):
            raise RuntimeError('policy has no method of [evaluate]: {}'.format(id))
        return policy
        

registry = PolicyRegistery()

def register(id, **kwargs):
    return registry.register(id, **kwargs)

def make(id, **kwargs):
    return registry.make(id, **kwargs)