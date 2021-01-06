import os
import re
import cv2
import numpy as np 

def loadData(imgpath):
    """
    loadData carga las imágenes de cada clase y hace un preprocesamiento:
    reduce el tamaño de las imágenes, trasnforma a escala de grises y binariza las imaágenes. 
    """
    images = []
    directories = []
    dircount = []
    prevRoot = ''
    amount = 0
    print("leyendo imagenes de ", imgpath)
    for root, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            if re.search("\.(bmp)$", filename):
                amount = amount+1
                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)
                image = image[::8, ::8]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
                images.append(thresh1)
                if prevRoot != root:
                    prevRoot = root
                    directories.append(root)
                    dircount.append(amount)
                    amount = 0
    dircount.append(amount)

    dircount = dircount[1:]
    dircount[-1] = dircount[-1]+1
    print('Directorios leidos:', len(directories))
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:', sum(dircount))
    return images, dircount, directories

def tagImages(images, dircount, directories):
    """ 
    tagImages etiqueta a cada imágen dependiendo del directorio en el que se encuentra
    """
    labels = []
    indice = 0
    for amount in dircount:
        for i in range(amount):
            labels.append(indice)
        indice = indice+1
    print("Cantidad etiquetas creadas: ", len(labels))

    refSanitary = []
    indice = 0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice, name[len(name)-1])
        refSanitary.append(name[len(name)-1])
        indice = indice+1

    y = np.array(labels)
    X = np.array(images, dtype=np.uint8)  # convierto de lista a numpy

    # Find the unique numbers from the train labels
    classes = np.unique(y)
    nClasses = len(classes)
    print('Número total de salidas : ', nClasses)
    print('Clases : ', classes)
    return X, y, nClasses, refSanitary
