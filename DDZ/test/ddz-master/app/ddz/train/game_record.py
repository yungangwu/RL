from ddz.agent.agent_record import AgentRecord
from ddz.game.utils import Action_Pass, NUM_AGENT, get_card_values, encode_action
from ddz.environment.env_record import EnvRecord
from ddz.game.move import move_manager
import numpy as np

class GameRecord(object):
    def __init__(self, initial_cards, actions):
        super().__init__()
        self.initial_cards = initial_cards.copy()
        self.actions = actions.copy()
        winner = (len(actions)+2) % 3
        if winner != 0:
            self.winner = [1, 2]
        else:
            self.winner = [0]

    def action_to_id(self, action):
        if action:
            return move_manager.get_move_by_cards(get_card_values(action))
        else:
            return encode_action(Action_Pass, 1)

    def generate_play_actions_samples(self, play_position):
        env = EnvRecord()
        for i in range(NUM_AGENT):
            env.set_agent(AgentRecord(), i)
        done = False
        env.reset(self.initial_cards, self.actions)
        position = 0
        samples = []
        while not done:
            agent = env.get_agent(position)
            action = agent.get_action()
            if position == play_position:
                # print("============================================")
                state = env.get_state(position)
                # state.print()
                s = state.get_policy_input()
                a = self.action_to_id(action)
                v = 1.0 if position in self.winner else -1.0
                # print("agent {} s:{}".format(position, s))
                # print("agent {} a:{}".format(position, a))
                # print("agent {} v:{}".format(position, v))
                samples.append([s, a, v])
            done = env.step(position, action)
            position = (position + 1) % 3
        return samples

    def generate_inference_samples(self, infer_position):
        env = EnvRecord()
        for i in range(NUM_AGENT):
            env.set_agent(AgentRecord(), i)
        done = False
        samples = []
        env.reset(self.initial_cards, self.actions)
        position = 0
        while not done:
            agent = env.get_agent(position)
            action = agent.get_action()
            if position != infer_position:
                # print("============================================")
                state = env.get_state(infer_position)
                # state.print()
                s = state.get_infer_input()
                a = self.action_to_id(action)
                # print("agent {} s:{}".format(position, s))
                # print("agent {} a:{}".format(position, a))
                samples.append([s, a])
            done = env.step(position, action)
            position = (position + 1) % 3
        return samples

        
        
