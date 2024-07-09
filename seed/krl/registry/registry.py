import importlib

def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

class RegistrySpec(object):
    def __init__(self, id, entry_point=None, kwargs=None) -> None:
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        if self.entry_point is None:
            raise ValueError(
                f'attempting to make registerer [{self.id}] without entry point.'
            )
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            registerer = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            registerer = cls(**kwargs)
        return registerer

    def __repr__(self) -> str:
        return "registerer spec: ({})".format(self.id)

class Registry(object):
    def __init__(self) -> None:
        self._specs = {}

    def register(self, id, **kwargs):
        if id in self._specs:
            raise RuntimeError(f'registerer {id} has already registered.')
        self._specs[id] = RegistrySpec(id, **kwargs)

    def make(self, id, **kwargs):
        if id not in self._specs:
            raise ValueError(
                f'attempt to make registerer id: {id}, which is not registered.'
            )
        spec: RegistrySpec = self._specs[id]
        registerer = spec.make(**kwargs)
        return registerer