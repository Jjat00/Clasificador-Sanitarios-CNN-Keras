import cv2
import os
import re
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

class DataAugmentation():
    """
    this class generates the new images
    """
    def __init__(self) -> None:
        super().__init__()
        dirname = os.path.join(os.getcwd(), 'ImagenesExamenCeramicos')
        self.imgpath = dirname + os.sep 
        self.images = {}
 
    def loadOriginalImages(self):
        """
        docstring
        """
        amount = 0
        print("reading images of: ", self.imgpath)
        for root, dirnames, filenames in os.walk(self.imgpath):
            for filename in filenames:
                if re.search("\.(bmp)$", filename):
                    amount = amount+1
                    filepath = os.path.join(root, filename)
                    image = cv2.imread(filepath)
                    self.images[filename] = image
        print(self.images)
        print("total images: ", amount)

    def generateHorizontalImages(self):
        """
        docstring
        """
        for key in self.images:
            # convert to numpy array
            data = img_to_array(self.images[key])
            ## expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagenHor = ImageDataGenerator(width_shift_range=[-200, 200])
            # prepare iterator
            itHor = datagenHor.flow(samples, batch_size=1)
            # generate samples
            for i in range(2):
                # generate batch of images
                batch = itHor.next()
                # convert to unsigned integers
                image = batch[0].astype('uint8')
                # save images with horizontal shift
                nameImage = '%sh%i.bmp' % (key[:2], i)
                if key[0] == "A":
                    cv2.imwrite('nuevasImagenes/ReferenciaA/%s' %
                                (nameImage), image)
                if key[0] == "B":
                    cv2.imwrite('nuevasImagenes/ReferenciaB/%s' %
                                (nameImage), image)
                if key[0] == "C":
                    cv2.imwrite('nuevasImagenes/ReferenciaC/%s' %
                                (nameImage), image)
                if key[0] == "D":
                    cv2.imwrite('nuevasImagenes/ReferenciaD/%s' %
                                (nameImage), image)
                if key[0] == "E":
                    cv2.imwrite('nuevasImagenes/ReferenciaE/%s' %
                                (nameImage), image)

    def generateVertialImages(self):
        """
        docstring
        """
        for key in self.images:
            # convert to numpy array
            data = img_to_array(self.images[key])
            ## expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(height_shift_range=0.2)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples
            for i in range(2):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers
                image = batch[0].astype('uint8')
                # save images with horizontal shift
                nameImage = '%sv%i.bmp' % (key[:2], i)
                if key[0] == "A":
                    cv2.imwrite('nuevasImagenes/ReferenciaA/%s' %
                                (nameImage), image)
                if key[0] == "B":
                    cv2.imwrite('nuevasImagenes/ReferenciaB/%s' %
                                (nameImage), image)
                if key[0] == "C":
                    cv2.imwrite('nuevasImagenes/ReferenciaC/%s' %
                                (nameImage), image)
                if key[0] == "D":
                    cv2.imwrite('nuevasImagenes/ReferenciaD/%s' %
                                (nameImage), image)
                if key[0] == "E":
                    cv2.imwrite('nuevasImagenes/ReferenciaE/%s' %
                                (nameImage), image)

    def generateRotationImages(self):
        """
        docstring
        """
        for key in self.images:
            # convert to numpy array
            data = img_to_array(self.images[key])
            ## expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(rotation_range=45)
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples
            for i in range(2):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers
                image = batch[0].astype('uint8')
                # save images with horizontal shift
                nameImage = '%sr%i.bmp' % (key[:2], i)
                if key[0] == "A":
                    cv2.imwrite('nuevasImagenes/ReferenciaA/%s' %
                                (nameImage), image)
                if key[0] == "B":
                    cv2.imwrite('nuevasImagenes/ReferenciaB/%s' %
                                (nameImage), image)
                if key[0] == "C":
                    cv2.imwrite('nuevasImagenes/ReferenciaC/%s' %
                                (nameImage), image)
                if key[0] == "D":
                    cv2.imwrite('nuevasImagenes/ReferenciaD/%s' %
                                (nameImage), image)
                if key[0] == "E":
                    cv2.imwrite('nuevasImagenes/ReferenciaE/%s' %
                                (nameImage), image)

    def generateBrightnessImages(self):
        """
        docstring
        """
        for key in self.images:
            # convert to numpy array
            data = img_to_array(self.images[key])
            ## expand dimension to one sample
            samples = expand_dims(data, 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
            # prepare iterator
            it = datagen.flow(samples, batch_size=1)
            # generate samples
            for i in range(2):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers
                image = batch[0].astype('uint8')
                # save images with horizontal shift
                nameImage = '%sb%i.bmp' % (key[:2], i)
                if key[0] == "A":
                    cv2.imwrite('nuevasImagenes/ReferenciaA/%s' %
                                (nameImage), image)
                if key[0] == "B":
                    cv2.imwrite('nuevasImagenes/ReferenciaB/%s' %
                                (nameImage), image)
                if key[0] == "C":
                    cv2.imwrite('nuevasImagenes/ReferenciaC/%s' %
                                (nameImage), image)
                if key[0] == "D":
                    cv2.imwrite('nuevasImagenes/ReferenciaD/%s' %
                                (nameImage), image)
                if key[0] == "E":
                    cv2.imwrite('nuevasImagenes/ReferenciaE/%s' %
                                (nameImage), image)


if __name__ == "__main__":
    data = DataAugmentation()
    data.loadOriginalImages()
    data.generateHorizontalImages()
    data.generateVertialImages()
    data.generateRotationImages()
    data.generateBrightnessImages()
