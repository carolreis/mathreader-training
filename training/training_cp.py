# -*- coding: utf-8 -*-
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from _training_model import TrainingModel
from tensorflow.keras.regularizers import l2
keras.regularizers.l2(0.01)

path = 'training/'

configs = {
    'dataset': {
        'training_images': "treatment/treated_data/training_images_dataset.npz",
        'training_labels': "treatment/treated_data/training_labels_dataset.npz",
        'testing_images': "treatment/treated_data/testing_images_dataset.npz",
        'testing_labels': "treatment/treated_data/testing_labels_dataset.npz"
    },
    'image': {
        'width': 28,
        'height': 28,
        'channels': 1
    },
    'nn_output': 30,
    'model': {
        'epochs': 200,
        'batch_size': 512
    },
    'path': {
        'chart': path + 'charts/',
        'model': path + 'model/',
        'history': path + 'history/'
    },
    'binary': False
}

nn = TrainingModel(configs)

training_arc = [
    {
        'conv2d1': {
            'filters': 32,
            'size': (3, 3)
        },
        'conv2d2': {
            'filters': 64,
            'size': (3, 3)
        },
        'conv2d3': {
            'filters': 32,
            'size': (3, 3)
        },
        'dense1': 768,
        'dense2': 192,
        'dense3': False,
        'lr': 0.001
    }

]

def classifier_func(training_arc):
    classifier = nn.instantiate_classifier()

    classifier.add(
        Conv2D(
            training_arc['conv2d1']['filters'],
            training_arc['conv2d1']['size'],
            padding='same',
            input_shape=(28, 28, 1), 
        )
    )
    classifier.add(Activation('relu'))

    classifier.add(Flatten())

    classifier.add(Dense(units = 30, activation = 'softmax')) 

    opt = optimizers.Adamax(learning_rate=training_arc['lr'], beta_1=0.9, beta_2=0.999)

    classifier.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.summary()

    return classifier

nn.set_model(classifier_func, training_arc[0])
nn.train()
nn.save_model()
