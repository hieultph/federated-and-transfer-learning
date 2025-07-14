import tensorflow as tf
import os
import numpy as np

class Server:
    def __init__(self, model_save_path):
        self.client_models = {}
        self.model_save_path = model_save_path
        self.global_model = self.build_model()

    def build_model(self):
        IMG_HEIGHT = 128
        IMG_WIDTH = 128
        CHANNELS = 3

        base_model = tf.keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS), pooling='max')
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.Dense(4, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000003125), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def send_global_model(self):
        return self.global_model

    def receive_client_model(self, client_id, weights):
        self.client_models.update({client_id: weights})

    def aggregate_client_models(self):
        if len(self.client_models) == 0:
            return
        
        average_weights = [np.mean([client_weights[i] for client_weights in self.client_models.values()], axis=0) for i in range(len(self.global_model.get_weights()))]
        self.global_model.set_weights(average_weights)
        self.client_models = {}
        self.save_global_model()

    def save_global_model(self):
        self.global_model.save_weights(os.path.join(self.model_save_path, 'global_weights.h5'))
        self.global_model.save(os.path.join(self.model_save_path, 'global_model.h5'))

    def evaluate(self, test_gen):
        loss, accuracy = self.global_model.evaluate(test_gen)
        return loss, accuracy