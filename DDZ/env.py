import random
from random import choice

SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
cards = [suit + rank for suit in SUITS for rank in RANKS]
cards = cards + ['RedKing', 'BlackKing']

class DouDizhu:
    def __init__(self):
        self.deck = cards.copy()
        self.players = ['Player1', 'Player2', 'Player3']
        self.hands = {player:[] for player in self.players}
        self.played_cards = []
        self.player_played_cards = {player:[] for player in self.players}

    def deal_cards(self):
        random.shuffle(self.deck)
        remain_landlord_cards = self.deck[-3:]
        distribute_cards = self.deck[:-3]
        for _ in range(len(distribute_cards)):
            for player in self.players:
                card = distribute_cards.pop(0)
                self.hands[player].append(card)

        landlord_num = random.randint(0, 2)
        landlord_player = self.players[landlord_num]
        self.hands[landlord_player] += remain_landlord_cards

    def get_state(self, player):
        return self.hands[player]
