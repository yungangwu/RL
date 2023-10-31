from ddz.evaluator.evaluator_manager import evaluator_manager
from ddz.model.network_defines import *
from ddz.train.game_record import GameRecord
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import os, datetime

## sample position: 0->lord, 1->lord down, 2->lord up
def load_samples_from_game_records(datas, position):
    samples = []
    for data in datas:
        initial_cards = data[0]
        actions = data[1]
        record = GameRecord(initial_cards, actions)
        sample = record.generate_inference_samples(position)
        samples.extend(sample)
        # if len(samples) >= 1000:
        #     break


    return map(np.asarray, zip(*samples))

def create_train_dataset(datas, position):
    states, actions = load_samples_from_game_records(datas, position)
    actions = K.utils.to_categorical(actions, ACTION_DIM)
    print("states:{}, actions:{}".format(states.shape, actions.shape))
    train_dataset = tf.data.Dataset.from_tensor_slices((states, actions))
    print("train_dataset size:", len(train_dataset))
    
    return train_dataset


def train_infer(datas, position, batch, lr, epochs, **kwargs):
    print("start train infer {}".format(position))
    for k, v in kwargs.items():
        print("train infer {}: {}".format(k, v))
    model_save_path = None
    checkpoint_path = None
    summary_path = None
    if "checkpoint_path" in kwargs:
        checkpoint_path = kwargs["checkpoint_path"]
        dirname = os.path.dirname(checkpoint_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    if "model_save_path" in kwargs:
        model_save_path = kwargs["model_save_path"]
        dirname = os.path.dirname(model_save_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    if "summary_path" in kwargs:
        summary_path = kwargs["summary_path"]
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)

    train_dataset = create_train_dataset(datas, position)
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(batch)
    if "base_model_path" in kwargs:
        evaluator = evaluator_manager.get_infer_evaluator(kwargs["base_model_path"], position)
    else:
        evaluator = evaluator_manager.new_infer_evaluator()
    evaluator.run(train_dataset, epochs, lr, checkpoint_path, summary_path)
    if model_save_path:
        evaluator.save(model_save_path)


