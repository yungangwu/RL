from collections import Counter
from ddz.game.utils import *
import pymongo
import time
import numpy as np
import os


GameEvent_FaPai = 1  # 发牌
GameEvent_GameStart = 2  # 游戏开始
GameEvent_MingPai = 3  # 明牌
GameEvent_QiangDiZhu = 4  # 抢地主
GameEvent_JiaoFen = 5  # 叫分
GameEvent_DiZhuNotify = 6  # 地主通知
GameEvent_JiaBei = 7  # 加倍
GameEvent_Double = 8  # 加倍
GameEvent_DaPai = 9  # 打牌
GameEvent_BuChu = 10  # 不出
GameEvent_BeiShuNotify = 11  # 倍数通知
GameEvent_End = 17  # 结束

show_verbose_info = False


class Statistics(object):

    def __init__(self) -> None:
        self.num_game = 0
        self.num_invalid_game = 0
        self.num_items = 0

    def show(self):
        if self.num_game > 0:
            print("===============================================")
            print("totol game num:", self.num_game)
            print("good game num: %d(%.2f %%)" % (self.num_game - self.num_invalid_game,
                                                  (self.num_game - self.num_invalid_game) * 100. / self.num_game))
            print("invalid game num: %d(%.2f %%)" % (
                self.num_invalid_game, self.num_invalid_game * 100. / self.num_game))
            print("total inputs:", self.num_items)


statistics = Statistics()


def get_winner_uid(record):
    players = record['result']['players']
    for player in players:
        if player['is_winner']:
            return player['uid']


def get_card_list(cards):
    card_list = []
    for card in cards:
        card_list.append(decode_card(card))
    return card_list

def analyze_events(events):
    handcards = {}
    actions = []
    dizhu_id = None
    for event in events:
        event_id = event['event_id']
        event_uid = event['uid']
        if event_id == GameEvent_FaPai:
            cards = get_card_list(event['data']['handcards'])
            handcards[event_uid] = cards
            if show_verbose_info:
                print("initial cards if {}: {}". format(event_uid, cards_to_str(cards)))

        elif event_id == GameEvent_DiZhuNotify:
            cards = get_card_list(event['data']['dipai'])
            current = handcards[event_uid]
            current.extend(cards)
            handcards[event_uid] = current
            dizhu_id = event_uid
            if show_verbose_info:
                print("dizhu is:", event_uid)
                print("dizhu cards: {}". format(cards_to_str(current)))

        elif event_id == GameEvent_DaPai:
            cards = get_card_list(event['data']['cardlist'])
            actions.append(cards)
            if show_verbose_info:
                print("action {}: {}". format(event_uid, cards_to_str(cards)))

        elif event_id == GameEvent_BuChu:
            actions.append([])
            if show_verbose_info:
                print("action {}: pass". format(event_uid))

    return handcards, actions, dizhu_id


def find_upper_and_lower_player(record, uid):
    chair_infos = [0, 0, 0, 0]
    chair_id = 0
    for player_info in record['players']:
        current_chair_id = player_info['chair_id']
        current_uid = player_info['uid']
        chair_infos[current_chair_id] = current_uid
        if uid == current_uid:
            chair_id = current_chair_id
    lower_chair_id = chair_id + 1 if chair_id < 3 else 1
    upper_chair_id = chair_id - 1 if chair_id > 1 else 3
    return chair_infos[upper_chair_id], chair_infos[lower_chair_id]


def analyze_record(record):
    statistics.num_game = statistics.num_game + 1
    if record['result']['result_type'] != 1:
        # print("skip invalid record.")
        statistics.num_invalid_game = statistics.num_invalid_game + 1
        return
    print("start analyze ...")
    winner_id = get_winner_uid(record)
    assert winner_id
    handcards, actions, dizhu_id = analyze_events(record["events"])
    upper_id, lower_id = find_upper_and_lower_player(record, dizhu_id)
    if show_verbose_info:
        print("winner uid:",winner_id)
        print("dizhu uid:", dizhu_id)
        print("upper uid:", upper_id)
        print("lower uid:", lower_id)
    initial_cards = [[] for _ in range(3)]
    initial_cards[0] = handcards[dizhu_id]
    initial_cards[1] = handcards[lower_id]
    initial_cards[2] = handcards[upper_id]
    if show_verbose_info:
        for n, cards in enumerate(initial_cards):
            print("initial cards of {}:{}".format(n, cards_to_str(cards)))
        print("winner:", (len(actions)+2) % 3)
    print("end analyze ...")
    return [initial_cards, actions]


def process_mongodb_data(host="localhost", port=27017, database_name='ddz', collection_name='ai', save_path='./data'):
    start_time = time.time()
    with pymongo.MongoClient(host=host, port=port) as client:
        db = client[database_name]
        collection = db[collection_name]
        filepath = os.path.join(save_path, ''.join("records.npy"))
        dir = os.path.dirname(filepath)
        if not os.path.exists(dir):
            os.mkdir(dir)
        game_records = []
        for record in collection.find():
            record = analyze_record(record)
            if record:
                game_records.append(record)
        print('saved data to:', filepath)
        with open(filepath, 'wb') as f:
            np.save(f, game_records)
        statistics.show()
    end_time = time.time()
    print("cost time: %f s" % (end_time - start_time))
