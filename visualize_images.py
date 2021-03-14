import numpy as np
import cv2
import os

number = np.load('treatment/treated_data_all/dot_testing_images.npz')
number = number['arr_0']

label_t = np.load('treatment/treated_data_all/dot_testing_labels.npz')
label_t = label_t['arr_0']

def exibir_imagem(image):
    for i in range(0, len(image), 100):
        cv2.imshow("Image", image[i])
        print(label_t[i])
        cv2.waitKey(0)

exibir_imagem(number)

cv2.destroyAllWindows()
