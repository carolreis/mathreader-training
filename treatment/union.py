# -*- coding: utf-8 -*-
import numpy as np
from sklearn.utils import shuffle
import cv2 as cv

path = 'treatment/treated_data/'

# une AST + DOT + KAGGLE_all

ast = np.load(path+'ast_training_images.npz')
ast = ast['arr_0']
ast_labels = np.load(path+'ast_training_labels.npz')
ast_labels = ast_labels['arr_0']

dot = np.load(path+'dot_training_images.npz')
dot = dot['arr_0']
dot_labels = np.load(path+'dot_training_labels.npz')
dot_labels = dot_labels['arr_0']

symbol = np.load(path+'kaggle_all_training_images.npz')
symbol = symbol['arr_0']
symbol_labels = np.load(path+'kaggle_all_training_labels.npz')
symbol_labels = symbol_labels['arr_0']

all_training_images = np.concatenate([ast, dot, symbol])
all_training_labels = np.concatenate([ast_labels, dot_labels, symbol_labels])
result_symbols, result_symbols_labels = shuffle(all_training_images, all_training_labels)

training_images = []
training_labels = []

for i in range(0, len(result_symbols)):
    training_images.append(result_symbols[i])
    training_labels.append(result_symbols_labels[i])

np.savez(path + "training_images_dataset", training_images)
np.savez(path + "training_labels_dataset", training_labels)

ast_testing = np.load(path+'ast_testing_images.npz')
ast_testing = ast_testing['arr_0']
ast_testing_labels = np.load(path+'ast_testing_labels.npz')
ast_testing_labels = ast_testing_labels['arr_0']

dot_testing = np.load(path+'dot_testing_images.npz')
dot_testing = dot_testing['arr_0']
dot_testing_labels = np.load(path+'dot_testing_labels.npz')
dot_testing_labels = dot_testing_labels['arr_0']

symbol_testing = np.load(path+'kaggle_all_testing_images.npz')
symbol_testing = symbol_testing['arr_0']
symbol_testing_labels = np.load(path+'kaggle_all_testing_labels.npz')
symbol_testing_labels = symbol_testing_labels['arr_0']

all_testing_images = np.concatenate([ast_testing, dot_testing, symbol_testing])
all_testing_labels = np.concatenate([ast_testing_labels, dot_testing_labels, symbol_testing_labels])

result_symbols_testing, result_symbols_testing_labels = shuffle(all_testing_images, all_testing_labels)

testing_images = []
testing_labels = []

for i in range(0, len(result_symbols_testing)):
    testing_images.append(result_symbols_testing[i])
    testing_labels.append(result_symbols_testing_labels[i])

np.savez(path + "testing_images_dataset", testing_images)
np.savez(path + "testing_labels_dataset", testing_labels)
