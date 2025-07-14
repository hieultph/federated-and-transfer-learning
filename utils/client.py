import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from utils.vars import *

class Client:
    def __init__(self, client_id, server):
        self.client_id = client_id
        self.server = server
        self.local_model = None
        self.local_df = pd.DataFrame()

    def get_global_model(self):
        self.local_model = self.server.send_global_model()
    
    def predict_image(self, image_path):
        image = self.load_and_preprocess_image(image_path)
        prediction = self.local_model.predict(image)

        self.append_image_to_local_df(image_path, prediction)

        return prediction

    def append_image_to_local_df(self, image_path, label):
        new_row_df = pd.DataFrame([(image_path, label)], columns=['filepaths', 'labels'])
        self.local_df = pd.concat([self.local_df, new_row_df])

    def append_df_to_local_df(self, new_df):
        self.local_df = pd.concat([self.local_df, new_df])

    def delete_image_from_local_df(self, image_path):
        self.local_df = [(img, lbl) for img, lbl in self.local_df if img != image_path]

    def clear_local_df(self):
        self.local_df = pd.DataFrame()

    # def re_train_model(self, epochs=100, batch_size=32):
    #     if self.local_model is None:
    #         raise ValueError("Local model is not initialized.")
        
    #     train_df, valid_df = train_test_split(self.local_df, train_size=0.7, shuffle=True, random_state=42)
        
    #     train_gen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True).flow_from_dataframe(
    #         train_df, 
    #         x_col='filepaths', 
    #         y_col='labels', 
    #         target_size=IMAGE_SIZE, 
    #         class_mode='categorical', 
    #         batch_size=batch_size,
    #         subset='training'
    #     )

    #     valid_gen = ImageDataGenerator().flow_from_dataframe(
    #         valid_df, 
    #         x_col='filepaths', 
    #         y_col='labels', 
    #         target_size=IMAGE_SIZE, 
    #         class_mode='categorical', 
    #         batch_size=batch_size,
    #         subset='validation'
    #     )
        
    #     # Train the model
    #     self.local_model.fit(x=train_gen, epochs=epochs, verbose= 1, validation_data=valid_gen, validation_steps= None, shuffle= False)
    #     self.send_model_to_server()

    def re_train_model(self, epochs=100, batch_size=32):
        if self.local_model is None:
            raise ValueError("Local model is not initialized.")

        # Debug: Check initial DataFrame
        print(f"Initial local_df shape: {self.local_df.shape}")
        
        # Split data into train and validation sets
        train_df, valid_df = train_test_split(self.local_df, train_size=0.7, shuffle=True, random_state=42)
        
        # Debug: Check train/validation split
        print(f"Train_df shape: {train_df.shape}, Valid_df shape: {valid_df.shape}")

        # Ensure valid filepaths
        train_df['filepaths'] = train_df['filepaths'].apply(lambda x: os.path.abspath(x))
        valid_df['filepaths'] = valid_df['filepaths'].apply(lambda x: os.path.abspath(x))

        # Debug: Check valid_df filepaths
        for path in valid_df['filepaths']:
            if not os.path.exists(path):
                print(f"File not found: {path}")

        # Create generators
        train_gen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True).flow_from_dataframe(
            train_df,
            x_col='filepaths',
            y_col='labels',
            target_size=IMAGE_SIZE,
            class_mode='categorical',  # or 'categorical'
            batch_size=batch_size
        )
        valid_gen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True).flow_from_dataframe(
            valid_df,
            x_col='filepaths',
            y_col='labels',
            target_size=IMAGE_SIZE,
            class_mode='categorical',  # or 'categorical'
            batch_size=batch_size
        )

        # Train the model
        self.local_model.fit(
            x=train_gen,
            epochs=epochs,
            verbose=1,
            validation_data=valid_gen,
            validation_steps=None,
            shuffle=False
        )

        # Send model to server
        self.send_model_to_server()


    def send_model_to_server(self):
        self.server.receive_client_model(self.client_id, self.local_model.get_weights())

    def load_and_preprocess_image(self, path):
        image = tf.keras.preprocessing.image.load_img(path, target_size=(128, 128))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0
        return image