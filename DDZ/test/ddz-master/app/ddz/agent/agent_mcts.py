from numpy.core.fromnumeric import argmax
from ddz.agent.agent_net import AgentNet
from ddz.game.utils import *
from ddz.policy.policy_manager import policy_manager
from ddz.mcts.mcts import MCTS
from ddz.evaluator.evaluator_manager import evaluator_manager
from ddz.game.move import move_manager
from ddz.model.network_defines import ACTION_DIM
import random, time
import numpy as np



class AgentMCTS(AgentNet):
    
    def __init__(self, model_path, puct=5, playout=100, self_play=True, minor='net'):
        super(AgentMCTS, self).__init__(model_path=model_path,  minor=minor)
        self._puct = puct
        self._playout = playout
        self._self_play = self_play
        self._model_path = model_path
        self._temperature = 1e-3

    def set_position(self, position):
        super(AgentMCTS, self).set_position(position)
        policy_model = evaluator_manager.get_play_evaluator( self._model_path, position).model
        infer_model  = evaluator_manager.get_infer_evaluator(self._model_path, position).model
        self.mcts = MCTS(position, policy_model, infer_model, self._puct)

    def set_temperature(self, temperature):
        self._temperature = temperature
        if DEBUG:
            print("set temperature of agent {}:{}".format(self.position, temperature))

    def reset(self, cards):
        super(AgentMCTS, self).reset(cards)
        self.mcts.reset_root()
        # self.memories = []

    def step_mcts(self):
        env = self.env
        position = self.position
        last_actions = []
        for i in range(NUM_AGENT):
            agent = env.get_agent(position-1-i)
            last_cards = agent.get_last_action()
            if last_cards:
                last_cards = get_card_values(last_cards)
            last_action = move_manager.get_move_by_cards(last_cards)
            last_actions.append(last_action)
        if self.mcts.step(last_actions) and not self.mcts.match_env(self.env):
            # print("reset root...")
            self.mcts.reset_root()

    def get_action(self):
        if self.env.mcts_enabled():
            self.step_mcts()
            s = self.env.get_state(self.position)
            # s.print()
            # start = time.time()
            # if self._self_play:
            #     ms = s.get_policy_input()
            acts, act_probs = self.mcts.get_action_probs(s, self._playout, self._temperature)
            # if self._self_play:
            #     ma = np.zeros(ACTION_DIM, dtype=np.float32)
            #     ma[np.array(acts, dtype=np.int)] = act_probs
            #     self.memories.append([ms, ma])
            if self._self_play:
                a = np.random.choice(acts, p=act_probs)
            else:
                index = argmax(act_probs)
                a = acts[index]
            move = move_manager.get_move_by_id(a)
            cards = move.get_action_cards()
            action_type, action_index = decode_action(a)
            # print("mcts:", cards_value_to_str(cards))
            minor_card_count = get_request_minor_card_count(
                action_type, action_index)
            if minor_card_count:
                for card_count in minor_card_count:
                    minor_cards = self.get_minor_cards(s, card_count, cards)
                    cards.extend(minor_cards)
            return self.get_card_from_handcards(cards)
        else:
            return super(AgentMCTS, self).get_action()

   
