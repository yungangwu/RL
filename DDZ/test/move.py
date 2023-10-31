from utils import *


ACTIONS_WITH_MINOR_CARD = [
    Action_Tri_With_Single_Wing,
    Action_Sequence_Tri_With_Single_Wing,
    Action_Tri_With_Pair_Wing,
    Action_Sequence_Tri_With_Pair_Wing,
    Action_Bomb_With_Single_Wing,
    Action_Bomb_With_Pair_Wing
]

ACTIONS_WITH_SEQUENCE = [
    Action_Sequence_Single,
    Action_Sequence_Pair,
    Action_Sequence_Tri,
    Action_Sequence_Tri_With_Single_Wing,
    Action_Sequence_Tri_With_Pair_Wing,
]


class Move(object):
    def __init__(self, action_id):
        super().__init__()
        self._action_id = action_id
        self._action_type, self._action_index = decode_action(action_id)
        self._cards = action_to_card(self._action_type, self._action_index)
        self._cards.sort()
        if self._cards:
            card_counter = Counter(self._cards)
            self.min_value = self._cards[0]
            self.length = len(card_counter)
            self.sequence_type = card_counter[self.min_value]

    def get_action_id(self):
        return self._action_id

    def get_action_cards(self):
        return self._cards.copy()

    def has_minor_card(self):
        return self._action_type in ACTIONS_WITH_MINOR_CARD

    def get_action_type(self):
        return self._action_type

    def can_hold(self, move):
        if move.get_action_type() == Action_Rocket:
            return False
        if self._action_type == move.get_action_type():
            if self._action_type in ACTIONS_WITH_SEQUENCE:
                return self.length == move.length and self.sequence_type == move.sequence_type and self.min_value > move.min_value
            else:
                return self.min_value > move.min_value
        elif self._action_type == Action_Bomb or self._action_type == Action_Rocket:
            return True
        return False

    def __str__(self) -> str:
        if self._cards:
            return "action:{} cards:{}, min:{}, length:{}, seq_type:{}".format(self._action_id, 
                cards_value_to_str(self._cards), 
                self.min_value, 
                self.length, 
                self.sequence_type)
        else:
            return "action:{} cards:pass".format(self._action_id)


class MoveCollections(object):
    def __init__(self):
        super().__init__()
        self._moves = {}
        for i in range(1, 310):
            self._moves[i] = Move(i)

        self._card_to_id_dict = {0: 309}
        for _, move in self._moves.items():
            if not move.has_minor_card():
                self._init_card_dict(move)

    def _init_card_dict(self, move):
        cards = move.get_action_cards()
        action_id = move.get_action_id()
        dict = self._card_to_id_dict
        cards_len = len(cards)
        for i in range(cards_len):
            card = cards[i]
            v = dict.get(card, {})
            dict[card] = v
            if i == cards_len-1:
                v[0] = action_id
            else:
                dict = v

    def get_move_by_id(self, action_id):
        return self._moves[action_id]

    def get_move_by_cards(self, cards):
        cards_counter = Counter(cards)
        cards_num_couter = Counter(cards_counter.values())
        if 3 in cards_num_couter:
            card_with_tri = [k for k in cards_counter if cards_counter[k] == 3]
            card_with_tri_num = len(card_with_tri)
            if card_with_tri_num == 1:
                if 2 in cards_num_couter:
                    return encode_action(Action_Tri_With_Pair_Wing, card_with_tri[0])
                if 1 in cards_num_couter:
                    return encode_action(Action_Tri_With_Single_Wing, card_with_tri[0])
            elif card_with_tri_num > 1:
                card_with_tri.sort()
                with_wings = False
                sequence_start = card_with_tri[0]
                sequence_length = 1
                sequence = []
                for i in range(len(card_with_tri)):
                    if i < len(card_with_tri) - 1:
                        if card_with_tri[i]+1 != card_with_tri[i+1]:
                            with_wings = True
                            sequence.append((sequence_start, sequence_length))
                            sequence_start = card_with_tri[i+1]
                            sequence_length= 1
                        else:
                            sequence_length+=1
                sequence.append((sequence_start, sequence_length))
                max_sequence = max(sequence, key=lambda k: k[1])
                if 2 in cards_num_couter or with_wings:
                    return encode_action(Action_Sequence_Tri_With_Pair_Wing, sequence_cards_to_action_index(max_sequence[0], max_sequence[1], 2, 4))
                if 1 in cards_num_couter or with_wings:
                    return encode_action(Action_Sequence_Tri_With_Single_Wing, sequence_cards_to_action_index(max_sequence[0], max_sequence[1], 2, 5))
        elif 4 in cards_num_couter and (2 in cards_num_couter or 1 in cards_num_couter):
            card_with_bomb = [
                k for k in cards_counter if cards_counter[k] == 4]
            if 2 in cards_num_couter:
                return encode_action(Action_Bomb_With_Pair_Wing, card_with_bomb[0])
            if 1 in cards_num_couter:
                return encode_action(Action_Bomb_With_Single_Wing, card_with_bomb[0])

        cards.sort()
        dict = self._card_to_id_dict
        for card in cards:
            dict = dict[card]
        return dict[0]

    def can_hold(self, test_action_id, base_action_id):
        test_move = self._moves[test_action_id]
        base_move = self._moves[base_action_id]
        return test_move.can_hold(base_move)

    def print(self):
        for k, v in self._moves.items():
            print(v)


move_manager = MoveCollections()


def legal_sequence_actions_from_handcards(handcard_counter, count, min_length, max_length, with_single=False, with_pair=False):
    sequence_start = None
    ret = []
    for card in range(1, 13):
        if handcard_counter.get(card, 0) >= count:
            if not sequence_start:
                sequence_start = card
            else:
                length = card-sequence_start+1
                sequence_start = max(sequence_start, card - max_length + 1)
                if length >= min_length:
                    for start in range(sequence_start, card - min_length + 2):
                        required_minor_count = 0
                        if count == 3 and with_single:
                            required_minor_count = 1
                        elif count == 3 and with_pair:
                            required_minor_count = 2
                        if required_minor_count > 0:
                            candidate_handcard_counter = {}
                            candidate_handcard_counter.update(handcard_counter)
                            keys = [k for k in candidate_handcard_counter if candidate_handcard_counter[k]
                                    >= required_minor_count and (k < start or k > card)]
                            if len(keys) < card-start+1:
                                continue
                        action_index = sequence_cards_to_action_index(
                            start, card-start+1, min_length, max_length)
                        ret.append(action_index)
        else:
            sequence_start = None
    return ret


def get_legal_actions(handcards, follow_action=None):
    handcard_values = [x[1] for x in handcards]
    handcard_values.sort()
    handcard_counter = Counter(handcard_values)
    handcard_counter_nums = Counter(handcard_counter.values())
    num_card_type = len(handcard_counter)
    actions = {}
    for i in range(Action_Pass):
        actions[i] = []
    same_card_actions = [Action_Single, Action_Pair, Action_Tri, Action_Bomb]
    if SMALL_JOKER in handcard_counter and BIG_JOKER in handcard_counter:
        actions[Action_Rocket].append(1)
    for k, v in handcard_counter.items():
        for i in range(v):
            action_type = same_card_actions[i]
            actions[action_type].append(k)
        if v >= 3:
            if num_card_type > 1:
                actions[Action_Tri_With_Single_Wing].append(k)
                if handcard_counter_nums.get(2, 0) + handcard_counter_nums.get(3, 0) + handcard_counter_nums.get(4, 0) > 1:
                    actions[Action_Tri_With_Pair_Wing].append(k)
        if v == 4:
            if num_card_type > 2:
                actions[Action_Bomb_With_Single_Wing].append(k)
                if handcard_counter_nums.get(2, 0) + handcard_counter_nums.get(3, 0) + handcard_counter_nums.get(4, 0) > 2:
                    actions[Action_Bomb_With_Pair_Wing].append(k)

    sequence_with_minor = [0, 0, 0, 1, 2]
    sequence_count = [1, 2, 3, 3, 3]
    sequence_min_length = [5, 3, 2, 2, 2]
    sequence_max_length = [13, 10, 6, 5, 4]
    sequence_actions = [Action_Sequence_Single, Action_Sequence_Pair, Action_Sequence_Tri,
                        Action_Sequence_Tri_With_Single_Wing, Action_Sequence_Tri_With_Pair_Wing]
    for i in range(5):
        action_index = legal_sequence_actions_from_handcards(
            handcard_counter, sequence_count[i], sequence_min_length[i], sequence_max_length[i], sequence_with_minor[i] == 1, sequence_with_minor[i] == 2)
        action_type = sequence_actions[i]
        actions[action_type].extend(action_index)

    ret = []
    for action_type, v in actions.items():
        for action_index in v:
            action_id = encode_action(action_type, action_index)
            # print(action_type, action_index, action_id)
            if not follow_action or move_manager.can_hold(action_id, follow_action):
                ret.append(action_id)
    if follow_action:
        ret.append(encode_action(Action_Pass, 1))
    return ret
    
