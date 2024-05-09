from ddz.evaluator.evaluator import EvaluatorBase
from ddz.model.model_inference import InferenceModel
from ddz.config.config_inference import ConfigInference
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import time, datetime, os, math


def lr_schedule(epoch, lr):
    if epoch > 20:
        lr = lr * 0.9
    print('Learning rate: ', lr)
    return lr

class EvaluatorInference(EvaluatorBase):
    def __init__(self, infer_shape, action_dim, training=True):
        super(EvaluatorInference, self).__init__()
        self.config = ConfigInference()
        self.action_dim = action_dim
        self.training = training
        self.infer_shape = infer_shape
        self.model = self.create_model()
        self.restored = False
    
    def create_model(self):
        config = self.config
        model = InferenceModel(config.filter, self.action_dim, config.resduals)
        model.build((1,) + self.infer_shape)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self, state):
        return self.model.predict_on_batch(state)

    def run(self, train_dataset, epochs, lr, save_model_path=None, summary_path=None):
        model = self.model
        
        K.backend.set_value(model.optimizer.lr, K.backend.get_value(lr))
        lr_scheduler = LearningRateScheduler(lr_schedule)
        callbacks = [lr_scheduler]
        if save_model_path:
            period = max(int(epochs / 5), 20)
            checkpoint = ModelCheckpoint(filepath=save_model_path, verbose=1, save_weights_only=True, period=period)
            callbacks.append(checkpoint)
        
        if summary_path:
            tensorboard = TensorBoard(log_dir=summary_path, write_graph=False, profile_batch=0)
            callbacks.append(tensorboard)

        model.fit(train_dataset, epochs=epochs, callbacks=callbacks, verbose=2)

    def save(self, model_path):
        self.model.save(model_path)

    def load(self, model_path):
        if not self.restored:
            print("load infer model weight:{}".format(model_path))
            self.model.load_weights(model_path).assert_existing_objects_matched()
            self.restored = True

