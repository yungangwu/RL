import random
from env import DouDizhu

MAX_EPISODE = 1000
agents = [None, None, None]
rl = None

env = DouDizhu(agents, rl)
for i_episode in range(MAX_EPISODE):
    landlord_id = random.randint(1, 3)
    cur_player_id = landlord_id
    s = env.reset(landlord_id)
    t = 0
    track_r = []
    track_episode = []
    while True:
        a = env.get_action(cur_player_id)
        s_, r, done, cur_player_id = env.step(cur_player_id, a)

        track_episode.append((s, a, r, s_))
        s = s_
