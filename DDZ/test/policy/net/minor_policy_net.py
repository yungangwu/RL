
from ddz.policy.policy_registry import make
from ddz.policy.minor_cards_rules import *
from ddz.evaluator.evaluator_manager import evaluator_manager
from ddz.game.move import move_manager, get_legal_actions
from ddz.model.networks import cards_to_onehot_60, card_values_to_onehot_60
from ddz.mcts.state import State
import numpy as np



class MinorPolicyNet(object):

    def __init__(self, model_path=None, **kwargs):
        super().__init__()
        self._evaluator = evaluator_manager.get_play_evaluator(model_path, 0)

    def _get_state(self, state, candidate_cards):
        ar = []
        ar.append(cards_to_onehot_60(candidate_cards))
        ar.append(cards_to_onehot_60(state.remain_cards))
        ar.append(card_values_to_onehot_60([]))
        ar.append(card_values_to_onehot_60([]))
        ar.append(card_values_to_onehot_60([]))
        ar.append(cards_to_onehot_60(state.playedout_cards))
        remain_handcards_num = state.encode_remain_handcards_num()
        ar.append(remain_handcards_num)

        s = np.stack(ar, axis=-1)
        return s

    def _get_action_mask(self, handcards, card_count):
        mask = np.zeros(310, dtype=np.bool)
        actions = get_legal_actions(handcards)
        for action in actions:
            action_type, _ = decode_action(action)
            if action_type == Action_Single and card_count == 1:
                mask[action] = True
            elif action_type == Action_Pair and card_count == 2:
                mask[action] = True
        return mask

    def evaluate(self, state, card_count, major_cards):
        handcards = state.handcards.copy()
        handcards_values = [x[1] for x in handcards]
        handcards_values_counter = Counter(handcards_values)
        candidate_cards = [
            card for card in handcards if handcards_values_counter[card[1]] >= card_count and card[1] not in major_cards]
        if candidate_cards:
            legal_mask = self._get_action_mask(candidate_cards, card_count)
            state = self._get_state(state, candidate_cards)
            state = np.expand_dims(state, axis=0)
            action_prob, _ = self._evaluator.predict(state)
            action_prob = action_prob[0]

            action_prob[legal_mask] += 1e-6
            illegal_mask = ~legal_mask
            action_prob[illegal_mask] = 0
            
            action_prob = action_prob / np.sum(action_prob)
            a = np.argmax(action_prob)
            move = move_manager.get_move_by_id(a)
            minor_card = move.get_action_cards()
            # if DEBUG:
            #     print("minor net policy evaluate action:" +
            #             cards_value_to_str(minor_card))
            return minor_card
        else:
            raise ValueError("no minor card found.")
        
