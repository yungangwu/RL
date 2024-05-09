from ddz.evaluator.evaluator_manager import evaluator_manager
from ddz.model.networks import legal_actions_mask
import math
import tensorflow.keras as K
import numpy as np
import tensorflow as tf

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class PolicyNet(object):

    def __init__(self, position=0, model_path=None, **kwargs):
        super().__init__()
        if model_path:
            self._evaluator = evaluator_manager.get_play_evaluator(model_path, position)
        else:
            self._evaluator = evaluator_manager.new_play_evaluator()

    def evaluate(self, agent, state, temperature=0):
        legal_mask = legal_actions_mask(agent)
        state = np.expand_dims(state, axis=0)
        action_prob, v = self._evaluator.predict(state)
        action_prob = action_prob[0]
        # print("v:", v[0])
        illegal_mask = ~legal_mask
        action_prob[illegal_mask] = 0
        if temperature > 0:
            acts = np.where(legal_mask)[0]
            action_prob = action_prob[acts]
            action_prob = softmax(1.0/temperature * np.log(action_prob))
            a = np.random.choice(acts, p=action_prob)
            return a
        else:
            action_prob = action_prob / np.sum(action_prob)
            # print("action_prob:", action_prob)
            return np.argmax(action_prob)
        

