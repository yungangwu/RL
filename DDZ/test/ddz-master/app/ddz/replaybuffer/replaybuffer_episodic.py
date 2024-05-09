import random
import numpy as np
from ddz.replaybuffer.episode import Episode
from .replaybuffer import ReplayBuffer

class EpisodicReplayBuffer(ReplayBuffer):
    def __init__(self, max_capacity):
        super().__init__(max_capacity)
        self._buffer = [Episode()]
        self.episode_count = 0

    def store(self, transition):
        episode = self._buffer[-1]
        episode.store(transition)

    def close_episode(self):
        self.episode_count = self.episode_count + 1
        self._buffer.append(Episode())
        self._ensure_max_capacity()
        print("episode num:{} and size:{}".format(self.episode_count, self.size()))
    
    def _ensure_max_capacity(self):
        remove_count = self.episode_count - self.max_capacity
        if remove_count > 0:
            self._buffer = self._buffer[remove_count:]
            self.episode_count = self.max_capacity

    def clear(self):
        self._buffer = [Episode()]
        self.episode_count = 0

    def size(self):
        count = 0
        for i in range(self.episode_count):
            count += self._buffer[i].size()
        return count

    def sample(self, batch_size):
        if self.size() < batch_size:
            return None
        batch = []
        if self.episode_count > 0:
            ep_indexes = np.arange(self.episode_count)
            np.random.shuffle(ep_indexes)
            selected_index = 0
            while batch_size > 0:
                ep_index = ep_indexes[selected_index]
                ep = self._buffer[ep_index]
                ep_size = ep.size()
                if ep_size >= batch_size:
                    batch.extend(ep.sample(batch_size))
                    batch_size = 0
                else:
                    batch.extend(ep.sample(ep_size))
                    batch_size = batch_size - ep_size
                selected_index = selected_index + 1
        return map(np.asarray, zip(*batch))

    def print(self):
        # print("ep count {}:".format(self.episode_count))
        for i in range(self.episode_count):
            ep = self._buffer[i]
            print("ep {}:".format(i))
            ep.print()

