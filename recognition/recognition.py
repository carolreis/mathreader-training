 # -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import helpers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '3'
print('\n')

def fit(image):

    labels = helpers.get_labels()

    try:
        model = load_model('treinamento/model_all/model_17-06-2020_20-18-12.h5') 
        prediction = model.predict(image)
        index = np.argmax(prediction)

        print(index)

        label_rec = labels["labels_parser"][str(index)]
        return {
            'label': labels["labels_recognition"][label_rec],
            'prediction': prediction,
            'type': 'not-number'
        }

    except Exception as e:
        raise e