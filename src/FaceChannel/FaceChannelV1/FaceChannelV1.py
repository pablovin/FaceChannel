"""
FaceChannelV1.py
====================================
Version1 of the FaceChannel model.
"""



import numpy

import keras
from keras.models import load_model, Model, Input

from keras.models import Sequential, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D

from tensorflow.keras.layers import (
    BatchNormalization, MaxPooling2D, Activation, Flatten, Dropout, Dense, Lambda
)

from FaceChannel.Metrics import metrics

from FaceChannel.FaceChannelV1 import imageProcessingUtil

import sys
import os
import urllib
import tarfile
from pathlib import Path

import tensorflow as tf

class FaceChannelV1:

    IMAGE_SIZE = (64, 64)
    """Image size used as input used by FaceChannelV1"""


    BATCH_SIZE = 32
    """Batch size used by FaceChannelV1"""

    DOWNLOAD_FROM = "https://github.com/pablovin/FaceChannel/raw/master/src/FaceChannel/FaceChannelV1/trainedNetworks.tar.xz"
    """URL where the model is stored """

    CAT_CLASS_ORDER = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
    """ Order of the pre-trained categorical model's output """


    CAT_CLASS_COLOR = [(255, 255, 255), (0, 255, 0), (0, 222, 255), (255, 0, 0), (0, 0, 255), (255, 0, 144), (0, 144, 255),
                    (75, 75, 96)]
    """ Color associated with each output of the pre-trained categorical model """


    DIM_CLASS_ORDER = ["Arousal", "Valence"]
    """ Order of the pre-trained dimensional model's output """

    DIM_CLASS_COLOR = [(0, 255, 0), (255, 0, 0)]
    """ Color associated with each output of the pre-trained dimensional model """


    def __init__(self, type="Cat", loadModel=True, numberClasses=7):
        """Constructor of the FaceChannelV1

                :param type: Type of the model, choose between "Cat" and "Dim".
                :param loadModel: Load the pre-trained model. Boolean.
                :param numberClasses: If not loading a pre-trained model, you can choose the number of classes of a categorical model.
        """

        folderName = Path(os.path.abspath(sys.modules[FaceChannelV1.__module__].__file__)).parent

        if not os.path.exists(folderName / "TrainedNetworks"):
            getFrom = self.DOWNLOAD_FROM
            downloadName = folderName / "trainedNetworks.tar.xz"

            print("-----------------------------------------------")
            print ("Wait ... Downloading  the trained networks...")
            print("-----------------------------------------------")

            urllib.request.urlretrieve(getFrom, downloadName)

            with tarfile.open(downloadName) as f:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(f, folderName)

            os.remove(downloadName)

            print("-----------------------------------------------")
            print("Download complete!")
            print("-----------------------------------------------")

        if type =="Cat":
            modelDirectory = folderName / "TrainedNetworks" / "CategoricalFaceChannel.h5"
        elif type =="Dim":
            modelDirectory = folderName / "TrainedNetworks" / "DimensionalFaceChannel.h5"
        else:
            raise("Model type not found!")

        if loadModel:
            model = self.loadModel(modelDirectory)
        else:
            if type =="Cat":
                model = self.getCategoricalModel(numberClasses)
            else:
                model = self.getDimensionalModel(numberClasses)


        self.model = model
        print("-----------------------------------------------")
        print("------- LOADED MODEL: Face Channel "+str(type)+" ----------- ")
        print("-----------------------------------------------")
        self.model.summary()

        self.imageProcessing = imageProcessingUtil.imageProcessingUtil()

    def predict(self, images, preprocess=True):
        """This method returns the prediction for one or more images.

                :param images: The images as one or a list of ndarray.
                :param preprocess: If the image is already pre-processed or not.
                                   a pre-processed image has a format of (64,64,1).

                :return: The prediction of the given image(s) as a ndarray
                :rtype: ndarray
        """

        if len(images) == 1:
            images = numpy.array([images])

        if preprocess:
            processedImages = []
            for img in images:
                facePoints, face = self.imageProcessing.detectFace(img)
                if not len(face) == 0 and not facePoints == None:  # If a face is detected
                    face = self.imageProcessing.preProcess(face, imageSize=self.IMAGE_SIZE)  # pre-process the face
                    processedImages.append(face)
                else:
                    processedImages.append(img)

        classification = self.model.predict(images,batch_size=self.BATCH_SIZE, verbose=0)

        return classification


    def loadModel(self, modelDirectory):
        """This method returns a loaded FaceChannelV1.rst.

                :param modelDirectory: The directory where the loaded model is.

                :return: The loaded model as a tensorflow-keras model
                :rtype: tensorflow model
        """
        return tf.keras.models.load_model(modelDirectory,
                                 custom_objects={'fbeta_score': metrics.fbeta_score, 'rmse': metrics.rmse,
                                                 'recall': metrics.recall, 'precision': metrics.precision,
                                                 'ccc': metrics.ccc})

    def getDimensionalModel(self):
        """This method returns a dimensional FaceChannelV1.rst.

                :return: a dimensional FaceChannelV1.rst
                :rtype: tensorflow model
        """
        backbone = self.buildFaceChannel()

        dense = Dense(200, activation="relu", name="denseLayer")(backbone)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

        return Model(inputs=backbone.input, outputs=[arousal_output, valence_output])

    def getCategoricalModel(self, numberClasses):
        """This method returns a categorical FaceChannelV1.rst.

                :return: a dimensional FaceChannelV1.rst
                :rtype: tensorflow model
        """

        backbone = self.buildFaceChannel()

        dense = Dense(200, activation="relu", name="denseLayer")(backbone)

        categoricalOutput = Dense(units=numberClasses, activation='tanh', name='categoricalOutput')(dense)

        return Model(inputs=backbone.input, outputs=categoricalOutput)

    def buildFaceChannel(self):
        """This method returns a Keras model of the FaceChannelV1.rst feature extractor.

                :return: a Keras model of the FaceChannelV1.rst feature extractor
                :rtype: tensorflow model
        """

        def shuntingInhibition(inputs):
            inhibitionDecay = 0.5

            v_c, v_c_inhibit = inputs

            output = (v_c / (inhibitionDecay
                             + v_c_inhibit))

            return output

        keras.backend.set_image_data_format("channels_first")

        nch = 256

        inputShape = numpy.array((1, 64, 64)).astype(numpy.int32)
        inputLayer = Input(shape=inputShape, name="Vision_Network_Input")

        # Conv1 and 2
        conv1 = Conv2D(nch / 4, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv1")(
            inputLayer)
        bn1 = BatchNormalization(axis=1)(conv1)
        actv1 = Activation("relu")(bn1)

        conv2 = Conv2D(nch / 4, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv2")(actv1)
        bn2 = BatchNormalization(axis=1)(conv2)
        actv2 = Activation("relu")(bn2)

        mp1 = MaxPooling2D(pool_size=(2, 2))(actv2)
        drop1 = Dropout(0.25)(mp1)

        # Conv 3 and 4
        conv3 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv3")(drop1)
        bn3 = BatchNormalization(axis=1)(conv3)
        actv3 = Activation("relu")(bn3)

        conv4 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv4")(actv3)
        bn4 = BatchNormalization(axis=1)(conv4)
        actv4 = Activation("relu")(bn4)

        mp2 = MaxPooling2D(pool_size=(2, 2))(actv4)
        drop2 = Dropout(0.25)(mp2)

        # Conv 5 and 6 and 7
        conv5 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv5")(drop2)
        bn5 = BatchNormalization(axis=1)(conv5)
        actv5 = Activation("relu")(bn5)

        conv6 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv6")(actv5)
        bn6 = BatchNormalization(axis=1)(conv6)
        actv6 = Activation("relu")(bn6)

        conv7 = Conv2D(nch / 2, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv7")(actv6)
        bn7 = BatchNormalization(axis=1)(conv7)
        actv7 = Activation("relu")(bn7)

        mp3 = MaxPooling2D(pool_size=(2, 2))(actv7)
        drop3 = Dropout(0.25)(mp3)

        # Conv 8 and 9 and 10

        conv8 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="Vision_conv8")(drop3)
        bn8 = BatchNormalization(axis=1)(conv8)
        actv8 = Activation("relu")(bn8)

        conv9 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", name="conv9")(actv8)
        bn9 = BatchNormalization(axis=1)(conv9)
        actv9 = Activation("relu")(bn9)

        conv10 = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                        name="conv10")(actv9)

        conv10_inhibition = Conv2D(nch, (3, 3), padding="same", kernel_initializer="glorot_uniform", activation="relu",
                                   name="conv10_inhibition")(actv9)

        v_conv_inhibitted = Lambda(function=shuntingInhibition)([conv10, conv10_inhibition])

        mp4 = MaxPooling2D(pool_size=(2, 2))(v_conv_inhibitted)
        drop4 = Dropout(0.25)(mp4)

        flatten = Flatten()(drop4)

        return flatten