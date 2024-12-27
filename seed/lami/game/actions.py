import collections
import itertools
from krl.util.timer import Timer
from lami.game.move import Move
from lami.utils import get_cards_head, get_cards_tail, get_cards_suit
from collections import defaultdict

def _gen_tonghuashun_table(min_seq_len=3):
    table = []
    for i in range(0, 13):
        e = 14 if i != 0 else 13
        for j in range(i + min_seq_len - 1, e):
            for s in range(4):
                seq = [(s, k % 13) for k in range(i, j + 1)]
                table.append(seq)
    return table

def _gen_kanpai_table(min_seq_len=3):
    table = []
    for i in range(0, 13):
        all_cards = [(s, i) for s in range(4)]
        all_cards = all_cards * 2
        for j in range(min_seq_len, 9):
            combs = itertools.combinations(all_cards, j)
            for comb in combs:
                table.append(comb)
    table = list(dict.fromkeys(table))
    return list(map(lambda x: list(x), table))

TONGHUASHUN_TABLE = _gen_tonghuashun_table()
TONGHUASHUN_ADD_TABLE = _gen_tonghuashun_table(min_seq_len=1)
KANPAI_TABLE = _gen_kanpai_table()
KANPAI_ADD_TABLE = _gen_kanpai_table(min_seq_len=1)

def sort_card(card_A, card_B):
    if card_A[1] != card_B[1]:
        return card_A[1] - card_B[1]
    return card_A[0] - card_B[0]

class combinationNode:
    def __init__(self, combination: list, cards: list, guis: list, handcards: dict) -> None:
        self.combination = combination
        self.childs = []
        self.leaf = not cards
        if cards:
            card = cards.pop()
            if card in handcards and handcards[card] > 0:
                new_comb = self.combination.copy()
                new_comb.append(card)
                new_handcards = handcards.copy()
                new_handcards[card] -= 1
                child = combinationNode(new_comb, cards.copy(), guis, new_handcards)
                self.childs.append(child)
            if guis:
                new_comb = self.combination.copy()
                new_guis = guis.copy()
                gui = new_guis.pop()
                new_comb.append(gui)
                child = combinationNode(new_comb, cards.copy(), new_guis, handcards)
                self.childs.append(child)

    def get_combinations(self, combinations: list):
        if self.childs:
            for child in self.childs:
                child.get_combinations(combinations)
        elif self.leaf:
            combinations.append(self.combination)

class CombinationTree:
    def __init__(self, cards: list, guis: list, handcards: dict) -> None:
        reversed_cards = cards.copy()
        reversed_cards.reverse()
        self._root = combinationNode(list(), reversed_cards, guis, handcards)

    def get_combinations(self, combinations: list):
        return self._root.get_combinations(combinations)

def get_combinations(cards: list, guis: list, handcards: dict):
    combinations = []
    tree = CombinationTree(cards, guis, handcards)
    tree.get_combinations(combinations)
    return combinations

def get_normal_card_count(cards: list):
    count = 0
    for card in cards:
        if card[1] != GUI_RANK:
            count += 1
    return count

def _check_kanpai_type_valid(move: Move, desk: Move):
    card_type_dict = defaultdict(int)
    for i, card in enumerate(desk.cards):
        if card[1] == GUI_RANK:
            card_type = desk.values[i][0]
            card_type_dict[card_type] += 1
    for i, card in enumerate(move.cards):
        if card[1] == GUI_RANK:
            card_type = move.values[i][0]
            card_type_dict[card_type] += 1
            if card_type_dict[card_type] > 2:
                return False
    return True

def get_tonghuashun(handcards):
    ths = set()
    handcards_map = collections.Counter(handcards)
    gui = [card for card in handcards if card[1] == GUI_RANK]
    for cards in TONGHUASHUN_TABLE:
        combs = get_combinations(cards.copy(), gui, handcards_map)
        for comb in combs:
            if get_normal_card_count(comb) > 0:
                move = Move(values=cards, cards=comb, kanpai=False)
                ths.add(move)
    return list(ths)

def get_tonghuashun_sequence(handcards, desk: list):
    ths = set()
    if len(desk) < 13:
        handcards_map = collections.Counter(handcards)
        gui = [card for card in handcards if card[1] == GUI_RANK]
        desk_head = get_cards_head(desk)
        desk_tail = get_cards_tail(desk)
        desk_suit = get_cards_suit(desk)
        for cards in TONGHUASHUN_ADD_TABLE:
            if len(cards) + len(desk) <= 13:
                head = get_cards_head(cards)
                tail = get_cards_tail(cards)
                suit = cards[0][0]
                if desk_suit == suit and ((tail + 1 == desk_head) or (head - 1 == desk_tail) or \
                    (head == 0 and tail == 0 and desk_tail + 1 == MAX_TAIL)):
                    combs = get_combinations(cards.copy(), gui, handcards_map)
                    for comb in combs:
                        move = Move(values=cards, cards=comb, kanpai=False)
                        ths.add(move)
    return list(ths)

def get_kanpai(handcards):
    ths = set()
    handcards_map = collections.Counter(handcards)
    gui = [card for card in handcards if card[1] == GUI_RANK]
    for cards in KANPAI_TABLE:
        combs = get_combinations(cards.copy(), gui, handcards_map)
        for comb in combs:
            if get_normal_card_count(comb) > 0:
                move = Move(values=cards, cards=comb, kanpai=True)
                ths.add(move)
    return list(ths)

def get_kanpai_sequence(handcards, desk: Move):
    ths = set()
    for card in desk.get_cards():
        if card[1] != GUI_RANK:
            desk_rank = card[1]
            break
    handcards_map = collections.Counter(handcards)
    gui = [card for card in handcards if card[1] == GUI_RANK]
    for cards in KANPAI_ADD_TABLE:
        if card[0][1] == desk_rank:
            combs = get_combinations(cards.copy(), gui, handcards_map)
            for comb in combs:
                move = Move(values=cards, cards=comb, kanpai=True)
                if _check_kanpai_type_valid(move, desk):
                    ths.add(move)
    return list(ths)

def get_legal_actions(handcards, free_play, desks) -> list:
    with Timer('get_legal_actions'):
        if free_play:
            new_desk_id = len(desks)
            actions = [(new_desk_id, move) for move in get_tonghuashun(handcards) if get_normal_card_count(move.get_cards()) >= 2]
            return actions
        else:
            new_desk_moves = get_tonghuashun(handcards)
            new_desk_moves.extend(get_kanpai(handcards))
            new_desk_id = len(desks)
            actions = [(new_desk_id, move) for move in new_desk_moves]
            for i, desk in enumerate(desks):
                if desk.kanpai:
                    desk_move = get_kanpai_sequence(handcards, desk)
                else:
                    desk_move = get_tonghuashun_sequence(handcards, desk.get_cards())
                for move in desk_move:
                    actions.append((i, move))
            return actions