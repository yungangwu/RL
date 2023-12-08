from ddz.evaluator.evaluator_play_card import EvaluatorPlayCard
from ddz.evaluator.evaluator_inference import EvaluatorInference
from ddz.model.network_defines import *
import os

PLAY_CHECKPOINT_PATH  = "{}/play/pos_{}/variables/variables"
INFER_CHECKPOINT_PATH = "{}/infer/pos_{}/variables/variables"

class EvaluatorManager(object):
    def __init__(self) -> None:
        super().__init__()
        self._evaluators_play  = {}
        self._evaluators_infer = {}
        
    def get_play_evaluator(self, path, position):
        filepath = PLAY_CHECKPOINT_PATH.format(path, position)
        if filepath in self._evaluators_play:
            return self._evaluators_play[filepath]
        
        evaluator = EvaluatorPlayCard(PLAY_MODEL_INPUTS_SHAPE,  ACTION_DIM)
        evaluator.load(filepath)
        print("load play evaluator of position {} from {}".format(position, filepath))
        self._evaluators_play[filepath] = evaluator
        return evaluator

    def new_play_evaluator(self):
        print("create play evaluator")
        return EvaluatorPlayCard(PLAY_MODEL_INPUTS_SHAPE,  ACTION_DIM)
         

    def get_infer_evaluator(self, path, position):
        filepath = INFER_CHECKPOINT_PATH.format(path, position)
        if filepath in self._evaluators_infer:
            return self._evaluators_infer[filepath]
                
        evaluator = EvaluatorInference(INFER_MODEL_INPUTS_SHAPE, ACTION_DIM)
        evaluator.load(filepath)
        print("load infer evaluator of position {} from {}".format(position, filepath))
        self._evaluators_infer[filepath] = evaluator
        return evaluator
    
    def new_infer_evaluator(self):
        print("create infer evaluator")
        return EvaluatorInference(INFER_MODEL_INPUTS_SHAPE, ACTION_DIM)

evaluator_manager = EvaluatorManager()