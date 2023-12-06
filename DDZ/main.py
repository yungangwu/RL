import random
import numpy as np
import torch

from env import DouDizhu
from typing import List
from util import Encode, LSTMEncode

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
    bomb_num = []
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

    # state 编码
    one_state = track_episode[1]
    s, _, _, _ = one_state
    encode_instance = Encode()
    (cur_hands, last_move_type, last_move, up_player_cards_num,
     down_player_cards_num, landlord_id, bomb_num, desk_record,
     cur_player_idx) = s

    print('s', s)

    cur_hands_encode = encode_instance.encode_hand_cards(cur_hands)
    last_move_encode = encode_instance.encode_last_move(last_move)
    up_player_cards_num_encode = encode_instance.encode_other_player_cards_num(
        up_player_cards_num)
    down_player_cards_num_encode = encode_instance.encode_other_player_cards_num(
        down_player_cards_num)
    landlord_id_encode = encode_instance.encode_landlord(landlord_id)
    bomb_num_encode = encode_instance.encode_bomb_num(bomb_num)

    state_encode = np.concatenate(
        (cur_hands_encode, last_move_encode, up_player_cards_num_encode,
         down_player_cards_num_encode, landlord_id_encode, bomb_num_encode))

    print(state_encode)

    # desk_record 编码
    record_encode_data = []
    for round_record in desk_record:
        player_id, played_cards = round_record
        player_id_encode = encode_instance.encode_cur_player_id(player_id)
        player_cards_encode = encode_instance.encode_hand_cards(played_cards)
        record_encode = np.concatenate((player_id_encode, player_cards_encode))
        record_encode_data.append(record_encode)

    record_encode_data = np.array(record_encode_data)
    record_encode_tensor = torch.Tensor(record_encode_data).unsqueeze(0)
    print(record_encode_tensor.shape)
    record_encode_shape = record_encode_tensor.shape
    lstm_encode = LSTMEncode(record_encode_shape[-1])

    record_hidden_encode = lstm_encode.get_hidden_state(record_encode_tensor)

    print('record_hidden_encode', record_hidden_encode.shape)