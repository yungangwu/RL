

class ReplayBuffer(object):
    def __init__(self, max_capacity):
        super().__init__()
        self.max_capacity = max_capacity

    def store(self, transition):
        pass

    def clear(self):
        pass

    def size(self):
        pass

    def sample(self, batch_size):
        pass