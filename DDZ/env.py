import random
from random import choice


class DouDizhu:
    def __init__(self):
        self.deck = cards.copy()
        self.players = ['Player1', 'Player2', 'Player3']
        self.hands = {player: [] for player in self.players}
        self.played_cards = []
        self.player_played_cards = {player: [] for player in self.players}

        self.last_move_type = 'start'
        self.last_move = 'start'

    def deal_cards(self, landlord_num=None):
        random.shuffle(self.deck)
        remain_landlord_cards = self.deck[-3:]
        distribute_cards = self.deck[:-3]
        for _ in range(len(distribute_cards)):
            for player in self.players:
                card = distribute_cards.pop(0)
                self.hands[player].append(card)
        if landlord_num is None:
            self.landlord_num = random.randint(0, 2)
        else:
            self.landlord_num = landlord_num
        self.landlord_player = self.players[self.landlord_num]
        self.hands[self.landlord_player] += remain_landlord_cards

    def get_state(self, player):
        return self.hands[player]

    def step(self, player, action):
        played_card = self.hands[player].pop(action)

        self.player_played_cards[player].append(played_card)
        self.played_cards.append(played_card)
        done = self.is_done()
        reward = [0, 0, 0]
        if done:
            if player == self.landlord_player:
                reward = [-1, -1, -1]
                reward[self.landlord_num] = 2
            else:
                reward = [1, 1, 1]
                reward[self.landlord_num] = -2

        # state include hands cards, played cards
        player_remain_cards = self.hands[player]
        next_player = self.players[(self.players.index(player) + 1) %
                                   len(self.players)]

        state = (self.hands, self.played_cards, self.player_played_cards)

        return state, reward, done, next_player

    def get_legal_action(self, player):
        cur_desk_cards = self.played_cards[-1]

    def get_next_moves(self, player):
        next_move_type, next_move = get_moves(self.last_move_type,
                                              self.last_move,
                                              self.player_played_cards[player])

    def is_done(self):
        done = all(len(hand) == 0 for hand in self.hands.values())
        return done