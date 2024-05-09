from .policy_registry import make

class PolicyManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._policys = {}
        
    def get_policy(self, name, id, **kwargs):
        if name in self._policys:
            return self._policys[name]
        policy = make(id, **kwargs)
        self._policys[name] = policy
        print("make policy:", name)
        return policy

policy_manager = PolicyManager()