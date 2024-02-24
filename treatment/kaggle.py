from mathreader.image_processing import preprocessing as preprocessing
from helpers import debug as debug
import helpers as helpers
import numpy as np
import cv2
import imutils
import os
import re
import math
import random
import sys
from sklearn.utils import shuffle
from threading import Thread


treatment_dir = 'treatment/'
treated_dir = 'treated_data/'

if not os.path.exists(treatment_dir + treated_dir):
    os.mkdir(treatment_dir + treated_dir)

main_dir = 'treatment/datasets/handwrittenmathsymbols_all/'

debug('Kaggle all')

# COLOCA OS LABELS

def salvar_imagem(dirname, roi, i):
    filename = "roi" + str(i) + ".png"
    cv2.imwrite(dirname+ "/" + filename, roi)

'''
    Obtém a imagem e o label a partir dos arquivos de imagens 
'''
def get_symbols():

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []

    dirs = [
        '0/', 
        '1/', 
        '2/', 
        '3/', 
        '4/', 
        '5/', 
        '6/', 
        '7/', 
        '8/', 
        '9/', 
        # '=/', 
        '-/', 
        '(/', 
        ')/', 
        '[/', 
        ']/', 
        '{/', 
        '}/', 
        '+/', 
        'a/', 
        'b/', 
        'c/', 
        'm/', 
        'n/', 
        'sqrt/', 
        'x/', 
        'y/', 
        'z/', 
        'neq/']

    labels = {
        '0/': '0', 
        '1/': '1', 
        '2/': '2', 
        '3/': '3', 
        '4/': '4', 
        '5/': '5', 
        '6/': '6', 
        '7/': '7', 
        '8/': '8', 
        '9/': '9',
        # "=/": "10",
        "-/": "10",
        "(/": "11",
        ")/": "12",
        "[/": "13",
        "]/": "14",
        "{/": "15",
        "}/": "16",
        "+/": "17",
        "a/": "18",
        "b/": "19",
        "c/": "20",
        "m/": "21",
        "n/": "22",
        "sqrt/": "23",
        "x/": "24",
        "y/": "25",
        "z/": "26",
        "neq/": "27"
    }

    configs = {
        'black': False,
        'dilate': True,
        'dataset': True,
        'resize': 'smaller'
    }

    def parallel(interval, tid):
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for j in range(interval[0], interval[1]+1):
            files = dirs[j]
            f = os.listdir(main_dir + files)
            count = 1
            amount = len(f)
            training_size = math.floor(amount * 80 / 100)
            testing_size = math.floor(amount * 20 / 100)

            for filename in f:
                debug('filename: %s' % filename)
                if re.search("\.(jpg|jpeg|png)$",filename, re.IGNORECASE):
                    if count <= training_size:
                        image = cv2.imread(main_dir + files + filename)
                        image = preprocessing.ImagePreprocessing(configs).treatment_without_segmentation(image)
                        if len(image) > 0:
                            train_images.append(image)
                            train_labels.append(labels[files])
                        else:
                            print("EMPTY: ", main_dir + files + filename)
                    elif count <= training_size + testing_size:
                        image = cv2.imread(main_dir + files + filename)
                        image = preprocessing.ImagePreprocessing(configs).treatment_without_segmentation(image)
                        if len(image) > 0:
                            test_images.append(image)
                            test_labels.append(labels[files])
                        else:
                            print("EMPTY: ", main_dir + files + filename)
                    else:
                        break

                    count = count + 1

        training_images.extend(train_images)
        training_labels.extend(train_labels)
        testing_images.extend(test_images)
        testing_labels.extend(test_labels)

        debug('Thread - Treinamento:')
        debug(len(training_images))
        debug('Thread - Teste:')
        debug(len(testing_images))

    size = len(dirs) // 16
    remain = len(dirs) % 16
    initial = 0
    threads = []

    for i in range(size, len(dirs)+1, size):
        if i == len(dirs)-remain:
            interval = (initial, i-1+remain)
        else:
            interval = (initial, i-1)
        initial += size

        t = Thread(target=parallel,args=(interval, i))
        threads.append(t)

    for t in threads:
        debug('Iniciando Threads')
        t.start()

    for t in threads:
        debug('Esperando Threads')
        t.join()

    debug('Threads finalizadas')

    training_labels, training_images = shuffle(training_labels, training_images)
    training_images = np.asarray(training_images)
    training_labels = np.asarray(training_labels)

    testing_labels, testing_images = shuffle(testing_labels, testing_images)
    testing_images = np.asarray(testing_images)
    testing_labels = np.asarray(testing_labels)

    debug('Treinamento:')
    debug(training_images.shape)
    debug('Teste:')
    debug(testing_images.shape)
    
    np.savez(treatment_dir + treated_dir + "kaggle_all_training_images", training_images)
    np.savez(treatment_dir + treated_dir + "kaggle_all_training_labels", training_labels)

    np.savez(treatment_dir + treated_dir + "kaggle_all_testing_images", testing_images)
    np.savez(treatment_dir + treated_dir + "kaggle_all_testing_labels", testing_labels)

get_symbols()
