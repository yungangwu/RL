import random

class Episode(object):
    def __init__(self):
        super().__init__()
        self._buffer = []

    def store(self, transition):
        self._buffer.append(transition)

    def size(self):
        return len(self._buffer)

    def buffer(self):
        return self._buffer

    def sample(self, batch_size):
        random_indexes = list(range(len(self._buffer)))
        random.shuffle(random_indexes)
        samples = [[self._buffer[random_indexes[i]].s,
                    self._buffer[random_indexes[i]].a,
                    self._buffer[random_indexes[i]].r,
                    self._buffer[random_indexes[i]].s_next,
                    self._buffer[random_indexes[i]].done
                    ]  for i in range(batch_size)]

        return samples

    def print(self):
        for t in self._buffer:
            t.print()
