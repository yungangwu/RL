import random
from env import DouDizhu
from typing import List

MAX_EPISODE = 1
agents = [None, None, None]
rl = None

env = DouDizhu(agents, rl)
landlord_id = random.randint(0, 2)
s = env.reset(landlord_id)
player_0_s = env.get_next_state(0)
player_1_s = env.get_next_state(1)
player_2_s = env.get_next_state(2)

players = [player_0_s[0], player_1_s[0], player_2_s[0]]

for player in players:
    temp = []
    for hand_card in player:
        temp.append(hand_card.name)

    print(temp)

for i_episode in range(MAX_EPISODE):

    cur_player_id = landlord_id

    t = 0
    track_r = []
    track_episode = []
    renders = []
    while True:
        a = env.get_action(cur_player_id)
        print('action', a, 'cur_player_id', cur_player_id)
        s_, r, done, cur_player_id, cur_move_type, cur_move = env.step(
            cur_player_id, a)
        temp = []
        if isinstance(cur_move, List):
            for card in cur_move:
                temp.append(card.name)
            renders.append((temp, cur_player_id))
        else:
            renders.append((cur_move, cur_player_id))

        track_episode.append((s, a, r, s_))
        s = s_
        if done:
            break

    print('renders', renders)
