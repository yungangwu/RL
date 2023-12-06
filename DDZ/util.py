import torch
import numpy as np
import torch.nn as nn

from typing import List

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

SUITS = ['♠', '♥', '♦', '♣']
NUMS = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2']
JOKERS = ['joker', 'JOKER']


class Card:
    name: str = None
    suit: str = None
    rank: int = None

    def bigger_than(self, card_instance):
        if self.rank > card_instance.rank:
            return True
        else:
            return False


class MoveType:
    dan: int = 0
    dui: int = 1
    san: int = 2
    san_dai_yi: int = 3
    san_dai_er: int = 4
    si_dai_er: int = 5
    si_dai_er_dui: int = 6
    shunzi: int = 7
    bomb: int = 8
    start: int = 9
    last: int = 10
    yaobuqi: int = 11
    buyao: int = 12


class Encode:
    def encode_last_move(self, move):  # 4*15
        if isinstance(move, int) or move == []:
            return self.encode_hand_cards([])
        else:
            return self.encode_hand_cards(move)

    def encode_bomb_num(self, num):
        bomb_encode = self.encode_decimal_to_binary(num,
                                                    3)  # 炸弹数量最多不会超过5个，也就是一定小于8
        return bomb_encode

    def encode_other_player_cards_num(self, num):
        cards_num_encode = self.encode_decimal_to_binary(num, 5)
        return cards_num_encode

    def encode_cur_player_id(self, cur_player_id):
        return self.encode_landlord(cur_player_id)

    def encode_landlord(self, landlord_id: int):  # 3位
        landlord_encode = np.zeros(3)
        landlord_encode[landlord_id] = 1
        return landlord_encode

    def encode_hand_cards(self, hand_cards: List[Card]):
        max_cards_per_type, num_card_types = len(
            SUITS), len(NUMS) + len(JOKERS)
        # 创建一个全零的矩阵
        hand_matrix = np.zeros((max_cards_per_type, num_card_types))
        for card in hand_cards:
            print('card', card.name)
            if card.name == 'joker':
                hand_matrix[-1][-2] = 1
            elif card.name == 'JOKER':
                hand_matrix[-1][-1] = 1
            else:
                card_suits_idx = SUITS.index(card.suit)
                card_name_idx = NUMS.index(card.name)
                hand_matrix[card_suits_idx][card_name_idx] = 1

        print(hand_matrix)
        return hand_matrix.flatten()

    def encode_decimal_to_binary(self, num, length):
        # 将十进制数字转换为二进制表示
        binary = bin(num)[2:]
        # 将二进制表示转换为 8 位二进制编码的数组
        bits = np.array([int(b) for b in binary], dtype=np.uint8)
        bits = np.pad(bits, (length - len(bits), 0), mode='constant')
        # 将 8 位二进制编码的数组转换为固定长度的 01 二进制编码的数组
        encoding = np.unpackbits(bits)[-length:]
        return encoding


class LSTMEncode(nn.Module):
    def __init__(self, state_dim) -> None:
        super(LSTMEncode, self).__init__()
        self.num_layers = 3
        self.hidden_size = 100
        self.lstm = nn.LSTM(state_dim,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device=x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def get_hidden_state(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device=x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn[-1]
