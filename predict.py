import numpy as np
import os

from keras.models import load_model
from numpy.core.numeric import load
from loadData import loadData, tagImages

model = load_model('sanitaryModel.h5py')

directoryName = os.path.join(os.getcwd(), 'ImagenesExamenCeramicos')
imagePath = directoryName + os.sep

imagesTest, dircountTest, directoriesTest = loadData(imagePath)

X, y, nClasses = tagImages(imagesTest, dircountTest, directoriesTest)

X = np.array(X, dtype=np.uint8)  # convierto de lista a numpy
testX = X / 255.0

testX = testX.reshape(42, 150, 200, 1)

predicted_classes = model.predict(testX)

count = 0 
for predict in predicted_classes:
    clase = np.where(predict == np.amax(predict))[0][0]
    print(clase == y[count])
    count = count + 1
