import functools
import numpy as np
from lami.utils import assert_sequence, cards_to_str, assert_kanpai
from lami.types import CARD_RANK_TO_INDEX, GUI_RANK, MAX_TAIL

DEBUG = False

def sort_card_value_pair(pair_A, pair_B):
    for i in range(2):
        if pair_A[i][1] != pair_B[i][1]:
            return pair_A[i][1] - pair_B[i][1]
        if pair_A[i][0] != pair_B[i][0]:
            return pair_A[i][0] - pair_B[i][0]
    return 0

class Move:
    def __init__(self, values: list, cards: list, kanpai: bool):
        assert len(cards) == len(values)
        self.kanpai = kanpai
        self.values = values.copy()
        self.cards = cards.copy()
        self._sort()
        if DEBUG:
            if kanpai:
                assert len(self.values) <= 16
                assert_kanpai(self.cards, self.values)
            else:
                assert len(self.values) <= 13
                assert_sequence(self.cards, self.values)
            for card, value in zip(self.cards, self.values):
                if card[1] != GUI_RANK:
                    assert card == value, f'values {cards_to_str(self.values)} must match cards {cards_to_str(self.cards)}'

    def _sort(self):
        card_value_pairs = list(zip(self.values.copy(), self.cards.copy()))
        card_value_pairs.sort(key=functools.cmp_to_key(sort_card_value_pair))
        v, c = zip(*card_value_pairs)
        self.values = list(v)
        self.cards = list(c)
        if self.values[0][1] == CARD_RANK_TO_INDEX['A'] and self.values[-1][1] == CARD_RANK_TO_INDEX['K']:
            self.values.append(self.values.pop(0))
            self.cards.append(self.cards.pop(0))
        self._update_state()

    def _update_state(self):
        self.head = self.values[0][1]
        self.tail = self.values[-1][1]
        if self.tail == 0 and self.head > self.tail:
            self.tail = 13
        assert self.tail >= self.head

    def concatenate(self, move):
        assert self.kanpai == move.kanpai, f"kanpai {self} is not concatenate with tonghuashun {move}."
        if self.kanpai:
            if DEBUG:
                self.values[0][1] == move.values[0][1]
            self.values.extend(move.values)
            self.cards.extend(move.cards)
            self._sort()
            if DEBUG:
                assert len(self.values) <= 16
                assert_kanpai(self.cards, self.values)
        else:
            if DEBUG:
                self.values[0][0] == move.values[0][0]
            if move.tail + 1 == self.head:
                self.values = move.values + self.values
                self.cards = move.cards + self.cards
            elif move.head == (self.tail + 1) % MAX_TAIL:
                self.values.extend(move.values)
                self.cards.extend(move.cards)
            else:
                assert 0, f'{self} concatenate {move}.'
            self._sort()
            if DEBUG:
                assert len(self.values) <= 13
                assert_sequence(self.values)
        if DEBUG:
            for card, value in zip(self.cards, self.values):
                if card[1] != GUI_RANK:
                    assert card == value, f'values {cards_to_str(self.values)} must match cards {cards_to_str(self.cards)}'

    def get_cards(self):
        return self.cards.copy()

    def to_numpy(self, ):
        matrix = np.zeros((4, 14, 4))
        num_gui = 0
        for i, (suit, rank) in enumerate(self.cards):
            if rank != GUI_RANK:
                matrix[suit][rank][1] = matrix[suit][rank][0]
                matrix[suit][rank][0] = 1
            else:
                matrix[num_gui % 4][rank][num_gui // 4] = 1
                v_rank = self.values[i][1]
                v_suit = self.values[i][0]
                matrix[v_suit][v_rank][3] = matrix[v_suit][v_rank][2]
                matrix[v_suit][v_rank][2] = 1
                num_gui += 1
        return matrix

    @classmethod
    def from_numpy(cls, matrix: np.ndarray):
        cards = []
        values = []
        indices = np.where(matrix > 0)
        gui_cards = []
        gui_values = []
        for suit, rank, cat in zip(*indices):
            if cat >= 2:
                gui_values.append((suit, rank))
            elif rank == GUI_RANK:
                gui_cards.append((0, rank))
            else:
                values.append((suit, rank))
                cards.append((suit, rank))
        cards.extend(gui_cards)
        values.extend(gui_values)
        kanpai = False
        if len(values) > 1:
            kanpai = (values[0][1] == values[-1][1])
        return cls(values=values, cards=cards, kanpai=kanpai)

    def __str__(self):
        return f'cards: {cards_to_str(self.cards)}, values: {cards_to_str(self.values)} head: {self.head} tail: {self.tail}.'

    def __eq__(self, o: object) -> bool:
        if type(self) == type(o) and self.kanpai == o.kanpai and len(self.cards) == len(o.cards):
            for card, other_card in zip(self.cards, o.cards):
                if card != other_card:
                    if card[1] != GUI_RANK or other_card[1] != GUI_RANK:
                        return False
            if not self.kanpai:
                for card, other_card in zip(self.values, o.values):
                    if card != other_card:
                        return False
            return True
        return False

    def __hash__(self) -> int:
        return hash(tuple(self.cards))

    def copy(self):
        return Move(self.values.copy(), self.cards.copy(), self.kanpai)
