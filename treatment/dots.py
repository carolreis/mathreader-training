# -*- coding: utf-8 -*-
from sklearn.utils import shuffle
from mathreader.image_processing import preprocessing as preprocessing
import helpers as helpers
import numpy as np
import cv2
import imutils
import os
import re
import random
import sys
import math


treatment_dir = 'treatment/'
treated_dir = 'treated_data/'

if not os.path.exists(treatment_dir + treated_dir):
    os.mkdir(treatment_dir + treated_dir)

main_dir = treatment_dir + '/datasets/dots/'

configs = {
    'black': False,
    'dilate': True,
    'dataset': False,
    'resize': 'smaller'
}

helpers.debug("Dot all")

def segmenta_dots():
    images = []
    for filename in os.listdir(main_dir):
        helpers.debug('filename: %s' % filename)
        if re.search("\.(jpg|jpeg|png)$",filename, re.IGNORECASE):
            objs = preprocessing.ImagePreprocessing(configs).treatment(main_dir + filename)
            for obj in objs[0]:
                try:
                    images.append(obj['image'])
                except BaseException as e:
                    pass
    return images

def prepara_dataset(images):

    # labels = ['29' for x in images]
    labels = ['28' for x in images]

    images = shuffle(images)
    amount = len(images)

    training_size = math.floor(amount * 70 / 100)

    training_images = images[0:training_size]
    training_images = np.asarray(training_images)
    training_labels = labels[0:training_size]
    training_labels = np.asarray(training_labels)

    testing_images = images[training_size::]
    testing_images = np.asarray(testing_images)
    testing_labels = labels[training_size::]
    testing_labels = np.asarray(testing_labels)

    helpers.debug('Treinamento:')
    helpers.debug(training_images.shape)
    helpers.debug('Teste:')
    helpers.debug(testing_images.shape)

    np.savez(treatment_dir + treated_dir + "dot_training_images", training_images)
    np.savez(treatment_dir + treated_dir + "dot_training_labels", training_labels)

    np.savez(treatment_dir + treated_dir + "dot_testing_images", testing_images)
    np.savez(treatment_dir + treated_dir + "dot_testing_labels", testing_labels)
    
images = segmenta_dots()
helpers.debug('Total:')
helpers.debug(len(images))
prepara_dataset(images)
