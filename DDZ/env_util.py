import random
import numpy as np

SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
cards = [suit + rank for suit in SUITS for rank in RANKS]
cards = cards + ['joker', 'JOKER']


class Game:
    def __init__(self, agent, RL=None) -> None:
        self.cards = Cards()
        self.game_end = False
        self.last_move_type = "start"
        self.last_move = "start"
        self.play_round = 1
        self.cur_player_id = 0
        self.yaobuqis = []

        self.models = agent.model

        self.agent = agent
        self.RL = RL

    def game_start(self, train, landlord_id=None):
        self.players = []
        for player_idx in range(1, 4):
            self.players.append(
                Player(player_idx, self.models[player_idx], self.agent, self,
                       self.RL))

        self.play_records = PlayRecords()

        if landlord_id is None:
            self.landlord_id = random.randint(1, 3)
        else:
            self.landlord_id = landlord_id

        self.game_init(self, )

    def game_init(self, ):
        desk_cards = np.random.shuffle(self.cards.cards)

        remain_landlord_cards = desk_cards[-3:]
        distribute_cards = desk_cards[:-3]

        players_cards = [[], [], []]
        players_cards[self.landlord_id] += remain_landlord_cards

        players_cards[0] += distribute_cards[:17]
        players_cards[1] += distribute_cards[17:34]
        players_cards[2] += distribute_cards[34:]
        for player_cards in players_cards:
            player_cards.sort(key=lambda x: x.rank)

        self.play_records.player1_hands = players_cards[0]
        self.play_records.player2_hands = players_cards[1]
        self.play_records.player3_hands = players_cards[2]

    def get_next_moves(self, ):
        next_moves_type, next_moves = self.players[
            self.cur_player_id].get_moves(self.last_move_type, self.last_move,
                                          self.play_records)
        return next_moves_type, next_moves

    def get_next_move(self, action):
        while self.cur_player_id <= 2:
            if self.cur_player_id != 0:
                self.get_next_moves()

            self.last_move_type, self.last_move, self.game_end, self.yaobuqi = self.players[
                self.cur_player_id].play(self.last_move_type, self.last_move,
                                         self.play_records, action)
            if self.yaobuqi:
                self.yaobuqis.append(self.yaobuqi)
            else:
                self.yaobuqis = []

            if len(self.yaobuqis) == 2:
                self.yaobuqis = []
                self.last_move_type = 'start'
                self.last_move = 'start'

            if self.game_end:
                self.play_records.winner = self.cur_player_id + 1
                break
            self.cur_player_id += 1
        self.play_round += 1
        self.cur_player_id = 0
        return self.play_records.winner, self.game_end


class Cards:
    def __init__(self, cards: list) -> None:
        # 牌面三元组
        self.org_cards = cards
        self.cards = self.get_card_info()

    def get_card_info(self, ):
        cards = []
        for card in self.org_cards:
            cur_card = Card()
            if card == 'joker':
                cur_card.name = 'joker'
                cur_card.rank = 14
            elif card == 'JOKER':
                cur_card.name = 'JOKER'
                cur_card.rank = 15
            elif card[1] == 'A':
                cur_card.name = card[1]
                cur_card.suit = card[0]
                cur_card.rank = 12
            elif card[1] == '2':
                cur_card.name = card[1]
                cur_card.suit = card[0]
                cur_card.rank = 13
            else:
                cur_card.name = card[1]
                cur_card.suit = card[0]
                cur_card.rank = int(card[1]) - 2
            cards.append(cur_card)

        return cards


class Card:
    name: str = None
    suit: str = None
    rank: int = None

    def bigger_than(self, card_instance):
        if self.rank > card_instance.rank:
            return True
        else:
            return False


class PlayRecords:
    def __init__(self) -> None:
        self.player1_hands = []
        self.player2_hands = []
        self.player3_hands = []

        self.legal_moves1 = []
        self.legal_moves2 = []
        self.legal_moves3 = []

        self.played_record1 = []
        self.played_record2 = []
        self.played_record3 = []

        self.desk_record = []

        self.winner = -1

        self.cur_player = 1


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


class Move:
    def __init__(self) -> None:
        # 出牌的信息
        self.dan = []
        self.dui = []
        self.san = []
        self.san_dai_yi = []
        self.san_dai_er = []
        self.si_dai_er = []
        self.si_dai_er_dui = []
        self.bomb = []
        self.shunzi = []

        self.card_num_info = {}
        self.card_order_info = []
        self.king = []

        self.next_moves = []
        self.next_moves_type = []

    def get_total_moves(self, hands_cards):

        # 统计手牌里，王牌信息，其他牌信息，顺序
        for card in hands_cards:
            if card.rank in [14, 15]:
                self.king.append(card)

            cards_num = self.card_num_info.get(card.rank, [])
            if len(cards_num) == 0:
                self.card_num_info[card.rank] = [card]
            else:
                self.card_num_info[card.rank].append(card)

            if card.rank in [13, 14, 15]:
                continue
            elif len(self.card_order_info) == 0:
                self.card_order_info.append(card)
            elif card.rank != self.card_order_info[-1].rank:
                self.card_order_info.append(card)

        # 王炸
        if len(self.king) == 2:
            self.bomb.append(self.king)

        # 出单，出对，出三，四带，炸弹（拆开）
        for card_rank, cards in self.card_num_info.items():
            if len(cards) == 1:
                self.dan.append(cards)
            elif len(cards) == 2:
                self.dui.append(cards)
                self.dan.append(cards[:1])
            elif len(cards) == 3:
                self.san.append(cards)
                self.dui.append(cards[:2])
                self.dan.append(cards[:1])
            elif len(cards) == 4:
                self.bomb.append(cards)
                self.san.append(cards[:3])
                self.dui.append(cards[:2])
                self.dan.append(cards[:1])

        # 三带一，三带二
        for san in self.san:
            for dan in self.dan:
                if dan[0].name != san[0].name:
                    self.san_dai_yi.append(san + dan)

            for dui in self.dui:
                if dui[0].name != san[0].name:
                    self.san_dai_er.append(san + dui)

        # 得到所有的2张，2对组合
        two_dans = self.get_cards_combinations(self.dan)
        two_duis = self.get_cards_combinations(self.dui)

        # 四带二，四带二对
        for si in self.bomb:
            if len(si) == 4:
                for two_dan in two_dans:
                    dan1, dan2 = two_dan
                    if dan1[0].name != si[0].name and dan2[0].name != si[
                            0].name:
                        self.si_dai_er.append(si + dan1 + dan2)

                for two_dui in two_duis:
                    dui1, dui2 = two_dui
                    if dui1[0].name != si[0].name and dui2[0].name != si[
                            0].name:
                        self.si_dai_er_dui.append(si + dui1 + dui2)

        max_len = []
        for card in self.card_order_info:  # card_order_info已经是有序的了么？
            if card == self.card_order_info[0]:
                max_len.append(card)
            elif max_len[-1].rank == (card.rank - 1):
                max_len.append(card)
            else:
                if len(max_len) >= 5:
                    self.shunzi.append(max_len)
                max_len = [card]

        if len(max_len) >= 5:
            self.shunzi.append(max_len)

        # 拆分所有顺子子串
        shunzi_sub = []
        for shunzi_cards in self.shunzi:
            len_total = len(shunzi_cards)
            n = len_total - 5
            while n > 0:
                len_sub = len_total - n
                j = 0
                while len_sub + j <= len_total:
                    shunzi_sub.append(shunzi_cards[j:len_sub + j])
                    j += 1
                n -= 1
        self.shunzi.extend(shunzi_sub)

    # 合法出牌列表
    def get_next_moves(self, last_move_type: MoveType, last_move):
        if last_move_type == MoveType.start:
            moves_types = [
                MoveType.dan,
                MoveType.dui,
                MoveType.san,
                MoveType.san_dai_yi,
                MoveType.san_dai_er,
                MoveType.si_dai_er,
                MoveType.si_dai_er_dui,
                MoveType.shunzi,
            ]
            move_types_index = 0
            for moves_type in [
                    self.dan, self.dui, self.san, self.san_dai_yi,
                    self.san_dai_er, self.si_dai_er, self.si_dai_er_dui,
                    self.shunzi
            ]:
                for move in moves_type:
                    self.next_moves.append(move)
                    self.next_moves_type.append(moves_types[move_types_index])
                move_types_index += 1
        elif last_move_type == MoveType.dan:
            self.add_move_to_next_moves(self.dan, last_move, MoveType.dan)
        elif last_move_type == MoveType.dui:
            self.add_move_to_next_moves(self.dui, last_move, MoveType.dui)
        elif last_move_type == MoveType.san:
            self.add_move_to_next_moves(self.san, last_move, MoveType.san)
        elif last_move_type == MoveType.san_dai_yi:
            self.add_move_to_next_moves(self.san_dai_yi, last_move,
                                        MoveType.san_dai_yi)
        elif last_move_type == MoveType.san_dai_er:
            self.add_move_to_next_moves(self.san_dai_er, last_move,
                                        MoveType.san_dai_er)
        elif last_move_type == MoveType.si_dai_er:
            self.add_move_to_next_moves(self.si_dai_er, last_move,
                                        MoveType.si_dai_er)
        elif last_move_type == MoveType.si_dai_er_dui:
            self.add_move_to_next_moves(self.si_dai_er_dui, last_move,
                                        MoveType.si_dai_er_dui)
        elif last_move_type == MoveType.bomb:
            self.add_move_to_next_moves(self.bomb, last_move, MoveType.bomb)
        elif last_move_type == MoveType.shunzi:
            self.add_move_to_next_moves(self.shunzi, last_move,
                                        MoveType.shunzi)
        else:
            print("last_move_type is wrong!")

        if last_move_type != MoveType.bomb:
            for move in self.bomb:
                self.next_moves.append(move)
                self.next_moves_type.append(MoveType.bomb)

        return self.next_moves_type, self.next_moves

    def add_move_to_next_moves(self, candidate_moves, last_move,
                               cur_move_type):
        for move in candidate_moves:
            if move[0].bigger_than(last_move[0]):
                self.next_moves.append(move)
                self.next_moves_type.append(cur_move_type)

    def get_cards_combinations(cards):
        combinations = []
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                combination = (cards[i], cards[j])
                combinations.append(combination)
        return combinations


class Player:
    def __init__(self,
                 player_id,
                 play_model,
                 agent=None,
                 game=None,
                 RL=None) -> None:
        self.player_id = player_id
        self.hand_cards = []
        self.play_model = play_model
        self.game = game
        self.agent = agent
        self.RL = RL

    def record_move(self, player_records: PlayRecords):
        player_records.cur_player = self.player_id
        if self.next_move_type in ["yaobuqi", "buyao"]:
            self.next_move = self.next_move_type
            player_records.desk_record.append(
                [self.player_id, self.next_move_type])
        else:
            player_records.desk_record.append([self.player_id, self.next_move])
            for card in self.next_move:
                self.hand_cards.remove(card)

        if self.player_id == 1:
            player_records.player1_hands = self.hand_cards
            player_records.legal_moves1.append(self.next_moves)
            player_records.played_record1.append(self.next_move)
        elif self.player_id == 2:
            player_records.player2_hands = self.hand_cards
            player_records.legal_moves2.append(self.next_moves)
            player_records.played_record2.append(self.next_move)
        elif self.player_id == 3:
            player_records.player3_hands = self.hand_cards
            player_records.legal_moves3.append(self.next_moves)
            player_records.played_record3.append(self.next_move)

        desk_end = False
        if len(self.hand_cards) == 0:
            desk_end = True

        return desk_end

    def get_moves(self, last_move_type, last_move, player_records):
        self.total_moves = Move()
        self.total_moves.get_total_moves(self.hand_cards)
        self.next_moves_type, self.next_moves = self.total_moves.get_next_moves(
            last_move_type, last_move)
        return self.next_moves_type, self.next_moves

    def play(self, last_move_type, last_move, player_records, action):
        self.next_move_type, self.next_move = choose_cards(
            next_moves_type=self.next_moves_type,
            next_moves=self.next_moves,
            last_move_type=last_move_type,
            last_move=last_move,
            hand_cards=self.hand_cards,
            model=self.play_model,
            RL=self.RL,
            agent=self.agent,
            game=self.game,
            player_id=self.player_id,
            action=action)
        desk_end_state = self.record_move(player_records)
        yaobuqi = False
        if self.next_move_type in ["yaobuqi", "buyao"]:
            yaobuqi = True
            self.next_move_type = last_move_type
            self.next_move = last_move

        return self.next_move_type, self.next_move, desk_end_state, yaobuqi
