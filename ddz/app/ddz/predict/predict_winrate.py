from ddz.evaluator.evaluator_manager import evaluator_manager
from ddz.game.utils import *
from ddz.model.networks import cards_to_onehot_60, card_values_to_onehot_60
from ddz.mcts.state import State
from ddz.predict.agent_level_config import dizhu_winrate_list

class PredictWinrate(object):
    def __init__(self, model_path):
        super().__init__()
        self._model_path = model_path
        self._evaluator = evaluator_manager.get_play_evaluator(model_path, 0)

    def predict_winrate(self, handcards):
        decoded_cards = list(map(lambda x:decode_card(x), handcards))
        # print('predict win rate handcards:{}'.format(cards_to_str(decoded_cards)))
        s = self.get_state(decoded_cards)
        s = np.expand_dims(s, axis=0)
        _, value = self._evaluator.predict(s)
        value = value[0][0]
        # print("value:", value)
        for lv, max_value in enumerate(dizhu_winrate_list):
            if value <= max_value:
                return lv
        return len(dizhu_winrate_list)

    def get_state(self, handcards):
        state = State(0)
        state.handcards = handcards.copy()
        state.current_round = 0
        state.actions = [[] for _ in range(NUM_AGENT)]
        remain_cards = [(x + 1, y + 1) for x in range(NUM_CARD_TYPE) for y in range(NUM_REGULAR_CARD_VALUE)]
        remain_cards.extend([(1, SMALL_JOKER), (1, BIG_JOKER)])
        state.remain_cards = [x for x in remain_cards if x not in handcards]
        state.playedout_cards = []
        # state.print()
        return state.get_policy_input()
