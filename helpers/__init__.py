import os
import cv2 as cv
import idx2numpy
import json

def criar_diretorio(name):
    dirname = name
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname

def exibir_imagem(image):
    cv.namedWindow("Imagem", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Imagem", 725, 325)
    # cv.moveWindow("Imagem", 40,525)
    cv.imshow("Imagem", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def salvar_imagem(self, dirname, roi, i):
        filename = "char" + str(i) + ".png"
        cv2.imwrite(dirname+ "/" + filename, roi)

def get_labels():
    try:
        with open('docs/config/config_all.json') as json_file:
            labels_json = json_file.read()
            labels_dict = json.loads(labels_json)
            labels = labels_dict
    except Exception as e:
        print(e)
        labels = {}

    return labels

def save_json_file(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))

def debug(data):
    print(data)



