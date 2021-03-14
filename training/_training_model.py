# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.keras as keras
import idx2numpy
import numpy as np
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import json
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

ACCURACY_THRESHOLD = 0.997
VAL_ACCURACY_THRESHOLD = 0.998

LOSS_THRESHOLD = 0.02
VAL_LOSS_THRESHOLD = 0.02

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        try:
            if(logs.get('val_accuracy') >= VAL_ACCURACY_THRESHOLD and logs.get('accuracy') >= ACCURACY_THRESHOLD \
                and logs.get('val_loss') <= VAL_LOSS_THRESHOLD and logs.get('loss') <= LOSS_THRESHOLD):
                print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
                print("\nReached %2.2f%% val_accuracy, so stopping training!!" %(VAL_ACCURACY_THRESHOLD*100))
                self.model.stop_training = True
        except BaseException as e:
            print(e)

class TrainingModel:

    def __init__(self, configs={}):
        self.configs = {
            'dataset': {
                'training_labels': '',
                'training_images': '',
                'testing_labels': '',
                'testing_images': ''
            },
            'image': {
                'width': 0,
                'height': 0,
                'channels': 0
            },
            'nn_output': 0,
            'model': {
                'epochs': 0,
                'batch_size': 0
            },
            'path': {
                'chart': '',
                'model': '',
                'history': ''
            },
            'binary': False
        }
        self.configs.update(configs)

        self.images = []
        self.labels = []
        self.test_images = []
        self.test_labels = []
        self.image_width = self.configs['image']['width']
        self.image_height = self.configs['image']['height']
        self.image_channels = self.configs['image']['channels']
        self.train_labels_file_name = self.configs['dataset']['training_labels']
        self.train_images_file_name = self.configs['dataset']['training_images']
        self.test_labels_file_name = self.configs['dataset']['testing_labels']
        self.test_images_file_name = self.configs['dataset']['testing_images']
        self.nn_output_dimension = self.configs['nn_output']
        matplotlib.use('tkagg')

        self.adjust_data()

    def get_data(self, file_name):
        data = np.load(file_name)
        return data['arr_0']

    def adjust_data(self):
        images = self.get_data(self.train_images_file_name)
        self.images = images.reshape(images.shape[0], self.image_width, self.image_height, self.image_channels) # (batch, height, width, channels) # 1000, 28, 28, 1

        labels = self.get_data(self.train_labels_file_name)
        if self.configs['binary']:
            self.labels = labels
        else:
            self.labels = keras.utils.to_categorical(labels, self.nn_output_dimension)

        test_images = self.get_data(self.test_images_file_name)
        self.test_images = test_images.reshape(test_images.shape[0], self.image_width, self.image_height, self.image_channels)

        test_labels = self.get_data(self.test_labels_file_name)
        if self.configs['binary']:
            self.test_labels = test_labels
        else:
            self.test_labels = keras.utils.to_categorical(test_labels, self.nn_output_dimension)

    def instantiate_classifier(self):
        classifier = Sequential()
        return classifier

    
    def set_model(self, model_function, params=None):
        if params:
            classifier = model_function(params)
        else:
            classifier = model_function()

        self.classifier = classifier

    def get_model(self):
        return self.classifier

    def data_augmentation(self):
        print('Data augmentation')
        datagen = ImageDataGenerator(
            # shear_range=0.2,
            shear_range=0.9, #0.8,
            horizontal_flip=False,
            vertical_flip=False,
            rotation_range=10,
            fill_mode="constant",
            cval=0.0
        )
        datagen.fit(self.images)
        return datagen

    def train(self):
        callbacks = myCallback()
        # callbackES = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
        # callbackES = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) # 22:54 changed
        callbackES1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True) # 28/06 15:12
        callbackES2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) # 28/06 15:12
        
        datagen = self.data_augmentation()

        self.train = self.classifier.fit_generator(
            datagen.flow(
                self.images, 
                self.labels, 
                batch_size=self.configs['model']['batch_size']
            ),
            epochs=self.configs['model']['epochs'],
            steps_per_epoch=(len(self.images) // self.configs['model']['batch_size']), # * 1.5, # * 3 (16/06 até 21:08), # * 2, # not number
            validation_data=(self.test_images, self.test_labels),
            callbacks=[callbacks, callbackES1, callbackES2] # changed 26/06 15:13
        )
        return self.train

    def visualize(self, result):
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

        plt.plot(result.history['accuracy'])
        plt.plot(result.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig = plt.gcf()
        fig.savefig(self.configs['path']['chart'] + 'acc_' + dt_string, dpi=100)
        plt.show()

        plt.plot(result.history['loss'])
        plt.plot(result.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        fig = plt.gcf()
        fig.savefig(self.configs['path']['chart'] + 'loss_' + dt_string, dpi=100)
        plt.show()
    
    def save_model(self): 
        train = self.train
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        self.classifier.save(self.configs['path']['model'] + "model_" + dt_string + ".h5")
        with open(self.configs['path']['history'] + 'history' + dt_string + '.json', 'w') as f:
            json.dump(str(train.history), f)
        score = self.classifier.evaluate(self.test_images, self.test_labels, verbose=0)
        print("Test loss: ", score[0])
        print("Test accuracy: ", score[1])
        self.visualize(train)
        return (score[0], score[1])
