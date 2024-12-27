import os
import math
import io
import numpy as np
from collections import defaultdict
from lami.types import CARD_RANK_TO_INDEX, CARD_SUIT_TO_INDEX, GUI_RANK, CARD_RANK, CARD_SUIT, MAX_TAIL

def card_to_matrix(cards: list):
    matrix = np.zeros((4, 14, 2))
    num_gui = 0
    for suit, rank in cards:
        if rank != GUI_RANK:
            matrix[suit][rank][1] = matrix[suit][rank][0]
            matrix[suit][rank][0] = 1
        else:
            matrix[num_gui % 4][rank][num_gui // 4] = 1
            num_gui += 1
    return matrix

def matrix_to_card(matrix: np.ndarray):
    indices = np.where(matrix)
    cards = []
    for suit, rank, double in zip(*indices):
        if rank != GUI_RANK:
            cards.append((suit, rank))
        else:
            cards.append((0, GUI_RANK))
    return cards

def cards_to_str(cards: list):
    def show_single_card(card):
        return f"{CARD_SUIT[card[0]]}{CARD_RANK[card[1]]}"
    return "".join(list(map(show_single_card, cards)))

def str_to_cards(cards_str: str):
    str_list = cards_str.split(",")
    cards = []
    for card_str in str_list:
        if card_str:
            suit = CARD_SUIT_TO_INDEX[card_str[0:2]]
            rank = CARD_RANK_TO_INDEX[card_str[2:]]
            cards.append((suit, rank))
    return cards

def get_action_count(length):
    assert length > 1 and length <= 13
    return 15 - length

ACTION_TABLE = [get_action_count(i) for i in range(3, 14, 1)]

def get_action_index(suit, start_rank, length):
    action_index = 0
    for i in range(suit):
        for count in ACTION_TABLE:
            action_index += count
    for i in range(length - 3):
        action_index += ACTION_TABLE[i]
    return action_index + start_rank

def get_cards_tail(cards: list):
    offset = 0
    for card in reversed(cards):
        if card[1] != GUI_RANK:
            tail = card[1] - offset
            if tail != 0:
                return tail
            else:
                if len(cards) > 1:
                    return MAX_TAIL
                else:
                    return 0
        else:
            offset += 1

def get_cards_head(cards: list):
    offset = 0
    for card in cards:
        if card[1] != GUI_RANK:
            head = card[1] + offset
            if head < 0:
                return MAX_TAIL + offset
            return head
        else:
            offset -= 1

def get_cards_suit(cards: list):
    for card in cards:
        if card[1] != GUI_RANK:
            return card[0]

def is_cards_kanpai(cards: list):
    for i, card in enumerate(cards):
        if card[1] != GUI_RANK:
            for j in range(i + 1, len(cards)):
                if cards[j][1] != GUI_RANK:
                    return cards[j][1] == card[1]

def get_gui_cards_num(cards: list):
    gui = 0
    for _, rank in cards:
        if rank == GUI_RANK:
            gui += 1
    return gui

def assert_sequence(cards: list):
    start = None
    suit = None
    for card in cards:
        if card[1] != GUI_RANK:
            if start is None:
                start = card[1]
                suit = card[0]
            else:
                start += 1
                assert start <= 13 and start % 13 == card[1] and suit == card[0], f'{cards_to_str(cards)} is not tonghuashun.'
        else:
            if start:
                start += 1

def assert_kanpai(cards: list, values: list):
    for i in range(len(values) - 1):
        assert values[i][1] == values[i + 1][1], f'{cards_to_str(cards)} is not kanpai.'
    card_types = defaultdict(int)
    for i, card in enumerate(cards):
        if card[1] == GUI_RANK:
            value = values[i]
            card_types[value[0]] += 1
            assert card_types[value[0]] <= 2, f'{cards_to_str(cards)} gui pai type is more than 2.'

def bytes_to_ndarray(data: bytes):
    nda_bytes = io.BytesIO(data)
    return np.load(nda_bytes)

def ndarray_to_bytes(**kargs):
    nda_bytes = io.BytesIO()
    np.savez_compressed(nda_bytes, **kargs)
    return nda_bytes.getvalue()

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def print_obs(obs):
    from lami.game.move import Move
    from lami.game.player import sort_card
    import functools

    print('================================================================')
    handcards = matrix_to_card(obs['handcards'])
    handcards = sorted(handcards, key=functools.cmp_to_key(sort_card))
    print(f'hadcards:{cards_to_str(handcards)}')

    remain_cards = matrix_to_card(obs['remain_cards'])
    remain_cards = sorted(remain_cards, key=functools.cmp_to_key(sort_card))
    print(f'remain_cards:{cards_to_str(remain_cards)}')

    desks = obs['desk_cards']
    for i, desk in enumerate(desks):
        if not np.all(desk == 0):
            move = Move.from_numpy(desk)
            print(f'desk {i}: {move}')
        else:
            break

    handcards_num = obs['handcards_num']
    indices = np.where(handcards_num > 0)
    for _, card_num in zip(*indices):
        print(f'card num: {card_num}')

    gui_num = obs['gui_num']
    indices = np.where(gui_num > 0)
    print(f'gui num: {indices[0]}')

    played_gui_num = obs['played_gui_num']
    indices = np.where(played_gui_num > 0)
    for _, played_gui_num in zip(*indices):
        print(f'played gui num: {played_gui_num}')

    played_A_num = obs['played_A_num']
    indices = np.where(played_A_num > 0)
    for _, played_A_num in zip(*indices):
        print(f'played A num: {played_A_num}')

    if 'legal_actions' in obs:
        legal_actions = obs['legal_actions']
        for i, action in enumerate(legal_actions):
            move_action = action[0]
            if np.all(move_action == 0):
                print('legal action: pass')
            else:
                move = Move.from_numpy(move_action)
                if np.all(action[1] == 0):
                    print(f'legal action: {i}: {move} in new desk')
                else:
                    desk_move = Move.from_numpy(action[1])
                    print(f'legal action: {i}: {move} in desk {desk_move}')

    is_free_play = obs['is_free_play']
    indices = np.where(is_free_play > 0)
    print(f'is free play: {indices[0]}')

    is_lose = obs['is_lose']
    indices = np.where(is_lose > 0)
    print(f'is lose: {indices[0]}')

    for act in obs['history_actions']:
        if np.all(act == 0):
            print('history action: None')
        else:
            move = matrix_to_card(act)
            print(f'history action: {cards_to_str(move)}')

    print('================================================================')


def decode_card(card):
    if card == (0x50):
        return (0, GUI_RANK)
    return (((card & 0xF0) >> 4) - 1, (card & 0x0F) - 1)

def encode_card(card):
    if card[1] == GUI_RANK:
        return (0x50)
    return (((card[0] & 0x0F) + 1) << 4) | ((card[1] & 0x0F) + 1)

def decode_cards(cards):
    return list(map(lambda x: decode_card(x), cards))

def encode_cards(cards):
    return list(map(lambda x: encode_card(x), cards))
