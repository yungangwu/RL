
import numpy as np
import os
from collections import Counter

NUM_AGENT = 3
NUM_CARD_TYPE = 4
NUM_REGULAR_CARD_VALUE = 13
SMALL_JOKER = 14
BIG_JOKER = 15

DEBUG = False

ROLE_LANDLORD = 1
ROLE_FARMER = 0

CARD_VALUE_ICON = ["err", "3", "4", "5", "6", "7", "8",
                   "9", "10", "J", "Q", "K", "A", "2", "SJ", "BJ"]
CARD_TYPE_ICON = ["err", "♠", "♥", "♣", "♦", ""]

Action_Single = 1
Action_Sequence_Single = 2
Action_Pair = 3
Action_Sequence_Pair = 4
Action_Tri = 5
Action_Sequence_Tri = 6
Action_Tri_With_Single_Wing = 7
Action_Sequence_Tri_With_Single_Wing = 8
Action_Tri_With_Pair_Wing = 9
Action_Sequence_Tri_With_Pair_Wing = 10
Action_Bomb_With_Single_Wing = 11
Action_Bomb_With_Pair_Wing = 12
Action_Bomb = 13
Action_Rocket = 14
Action_Pass = 15
Action_Num = 16

ACTION_COUNT = [
    0,
    15,    # Action_Single = 1
    36,    # Action_Sequence_Single = 2
    13,    # Action_Pair = 3
    52,    # Action_Sequence_Pair = 4
    13,    # Action_Tri = 5
    45,    # Action_Sequence_Tri = 6
    13,    # Action_Tri_With_Single_Wing = 7
    38,    # Action_Sequence_Tri_With_Single_Wing = 8
    13,    # Action_Tri_With_Pair_Wing = 9
    30,    # Action_Sequence_Tri_With_Pair_Wing = 10
    13,    # Action_Bomb_With_Single_Wing = 11
    13,    # Action_Bomb_With_Pair_Wing = 12
    13,    # Action_Bomb = 13
    1,     # Action_Rocket = 14
    1,     # Action_Pass = 15
]


def decode_card(card):
    if card == 0x51:
        return (1, 14)
    elif card == 0x52:
        return (1, 15)
    else:
        return ((card & 0xF0) >> 4, (card & 0x0F) - 2)

def encode_card(card):
    if card[1] == 14:
        return 0x51
    if card[1] == 15:
        return 0x52
    return ((card[0] & 0x0F) << 4) | ((card[1] & 0x0F) + 2)
        
def upload_directory_to_s3(s3, path, bucket, bucket_key):
    for root, _,files in os.walk(path):
        for file in files:
            relpath = os.path.relpath(root, path)
            if relpath != '.':
                dest_file_name = bucket_key + relpath + '/' + file
                print("upload file:", dest_file_name)
                s3.upload_file(os.path.join(root,file), bucket, dest_file_name)
            else:
                dest_file_name = bucket_key + file
                print("upload file:", dest_file_name)
                s3.upload_file(os.path.join(root,file), bucket, dest_file_name)


def download_directory_from_s3(s3_resource, bucket_name, remote_directory):
    bucket = s3_resource.Bucket(bucket_name) 
    for obj in bucket.objects.filter(Prefix = remote_directory):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        if not obj.key.endswith('/'):
            print(obj.key)
            bucket.download_file(obj.key, obj.key)

def cp_directory_s3(s3_resource, bucket_name, from_dir, to_dir):
    bucket = s3_resource.Bucket(bucket_name) 
    for obj in bucket.objects.filter(Prefix = from_dir):
        source = { 'Bucket': bucket_name, 'Key': obj.key}
        new_key = obj.key.replace(from_dir, to_dir, 1)
        target = bucket.Object(new_key)
        target.copy(source)
        print("cp from {} to {}".format(obj.key, new_key))
        
def cards_value_to_str(cards):
    sort_cards = cards.copy()
    sort_cards.sort()

    def show_single_card(card):
        return CARD_VALUE_ICON[card] + ","
    return "".join(list(map(show_single_card, sort_cards)))


def str_to_cards(card_str):
    str_list = card_str.split(',')
    cards = []
    for card_str in str_list:
        for t in range(1, NUM_CARD_TYPE+1):
            if CARD_TYPE_ICON[t] == card_str[0]:
                break
        for c in range(1, BIG_JOKER+1):
            if CARD_VALUE_ICON[c] == card_str[1:]:
                break
        cards.append((t,c))
    return cards

def cards_to_str(cards):
    def cards_to_str_per_list(cards):
        sort_cards = cards.copy()
        sort_cards.sort(key=lambda a: a[1])

        def show_single_card(card):
            return CARD_TYPE_ICON[card[0]] + CARD_VALUE_ICON[card[1]] + ","
        return "".join(list(map(show_single_card, sort_cards)))
    if cards:
        if isinstance(cards[0], list):
            str = ""
            for card_list in cards:
                str.join(['[', cards_to_str_per_list(card_list), "],"])
            return str
        else:
            return cards_to_str_per_list(cards)
    return ""


def get_card_values(cards):
    return [x[1] for x in cards]


def decode_action(action):
    action_base = 0
    action_type = 0
    action_index = 0
    for i in range(0, Action_Num):
        if action <= action_base + ACTION_COUNT[i]:
            action_type = i
            action_index = action - action_base
            break
        action_base = action_base + ACTION_COUNT[i]
    return action_type, action_index


def encode_action(action_type, action_index):
    action = 0
    for i in range(0, action_type):
        action = action + ACTION_COUNT[i]
    action = action + action_index
    # if DEBUG:
    #     print("encode action type {} and index {} to {}".format(
    #         action_type, action_index, action))
    assert action >= 1 and action <= 309
    return action


def action_index_to_sequence_cards(action_index, start, minlength, maxlength):
    total_length = 12 - (start - 1)
    total_length = total_length if total_length <= maxlength else maxlength
    total_length = total_length - (minlength - 1)
    if action_index <= total_length:
        return start, minlength + action_index - 1
    else:
        return action_index_to_sequence_cards(action_index - total_length, start + 1, minlength, maxlength)


def sequence_cards_to_action_index(start, length, minlength, maxlength=None):
    if start < 1:
        return 0
    else:
        prev = start - 1
        pre_max_length = 13 - prev
        if maxlength:
            length = length if length <= maxlength else maxlength
        length = length - minlength + 1
        return sequence_cards_to_action_index(prev, pre_max_length, minlength, maxlength) + length


def action_to_card(action_type, action_index):
    if action_type == Action_Single:
        return [action_index]
    elif action_type == Action_Sequence_Single:
        action_index, length = action_index_to_sequence_cards(
            action_index, 1, 5, 12)
        card = [x for x in range(action_index, action_index + length)]
        return card
    elif action_type == Action_Pair:
        card = [action_index, action_index]
        return card
    elif action_type == Action_Sequence_Pair:
        action_index, length = action_index_to_sequence_cards(
            action_index, 1, 3, 10)
        card = [x for x in range(action_index, action_index + length)]
        card = np.repeat(card, 2)
        return card.tolist()
    elif action_type == Action_Tri:
        card = [action_index, action_index, action_index]
        return card
    elif action_type == Action_Sequence_Tri:
        action_index, length = action_index_to_sequence_cards(
            action_index, 1, 2, 6)
        card = [x for x in range(action_index, action_index + length)]
        card = np.repeat(card, 3)
        return card.tolist()
    elif action_type == Action_Tri_With_Single_Wing:
        card = [action_index, action_index, action_index]
        return card
    elif action_type == Action_Sequence_Tri_With_Single_Wing:
        action_index, length = action_index_to_sequence_cards(
            action_index, 1, 2, 5)
        card = [x for x in range(action_index, action_index + length)]
        card = np.repeat(card, 3)
        return card.tolist()
    elif action_type == Action_Tri_With_Pair_Wing:
        card = [action_index, action_index, action_index]
        return card
    elif action_type == Action_Sequence_Tri_With_Pair_Wing:
        action_index, length = action_index_to_sequence_cards(
            action_index, 1, 2, 4)
        card = [x for x in range(action_index, action_index + length)]
        card = np.repeat(card, 3)
        return card.tolist()
    elif action_type == Action_Bomb_With_Single_Wing:
        card = [action_index, action_index, action_index, action_index]
        return card
    elif action_type == Action_Bomb_With_Pair_Wing:
        card = [action_index, action_index, action_index, action_index]
        return card
    elif action_type == Action_Bomb:
        card = [action_index, action_index, action_index, action_index]
        return card
    elif action_type == Action_Rocket:
        card = [SMALL_JOKER, BIG_JOKER]
        return card
    elif action_type == Action_Pass:
        return []
    raise ValueError("unexpected major action:(%d, %d)" %
                     (action_type, action_index))


def get_request_minor_card_count(action_type, action_index):
    card_count_list = None
    if action_type == Action_Tri_With_Single_Wing:
        card_count_list = [1]
    elif action_type == Action_Sequence_Tri_With_Single_Wing:
        _, length = action_index_to_sequence_cards(action_index, 1, 2, 5)
        card_count_list = [1 for _ in range(length)]
    elif action_type == Action_Tri_With_Pair_Wing:
        card_count_list = [2]
    elif action_type == Action_Sequence_Tri_With_Pair_Wing:
        _, length = action_index_to_sequence_cards(action_index, 1, 2, 4)
        card_count_list = [2 for _ in range(length)]
    elif action_type == Action_Bomb_With_Single_Wing:
        card_count_list = [1, 1]
    elif action_type == Action_Bomb_With_Pair_Wing:
        card_count_list = [2]
    return card_count_list

