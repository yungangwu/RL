import functools
from krl.util.timer import Timer
from lami.types import GUI_RANK, A_RANK
from lami.game.actions import get_legal_actions

def sort_card(card_A, card_B):
    if card_A[1] != card_B[1]:
        return card_A[1] - card_B[1]
    return card_A[0] - card_B[0]

class Player:
    def __init__(self, position: int, game) -> None:
        self.position = position
        self.game = game

    def init_handcards(self, handcards: list):
        self.handcards = handcards.copy()
        self.handcards = sorted(self.handcards, key=functools.cmp_to_key(sort_card))
        self._update_state()
        self.init_gui_num = self.gui
        self.init_A_num = self.A
        self._lose = False

    def _update_state(self):
        self.gui = 0
        self.A = 0
        for _, rank in self.handcards:
            if rank == GUI_RANK:
                self.gui += 1
            elif rank == A_RANK:
                self.A += 1

    def get_handcards_num(self):
        return len(self.handcards)

    def step(self, handout_cards: list):
        if handout_cards:
            if DEBUG:
                for card in handout_cards:
                    assert card in self.handcards
            total_len = len(self.handcards)
            handout_cards_copy = handout_cards.copy()
            while handout_cards_copy:
                card = handout_cards_copy.pop()
                self.handcards.remove(card)
            assert len(handout_cards) + len(self.handcards) == total_len
            self._update_state()

    def get_legal_actions(self, free_play, desks) -> list:
        with Timer('get_legal_actions'):
            actions = get_legal_actions(self.handcards, free_play, desks)
            return actions

    def is_lose(self):
        return self._lose

    def lose(self):
        self._lose = True

    def get_addition_score(self):
        return self.init_A_num + self.init_gui_num
