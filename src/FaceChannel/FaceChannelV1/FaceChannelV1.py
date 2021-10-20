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


class FaceChannelV1:


    IMAGE_SIZE = (64, 64)
    BATCH_SIZE = 32

    downloadFrom = "https://github.com/pablovin/FaceChannel/blob/master/src/FaceChannel/trainedNetworks.tar.xz"

    def __init__(self, type="Cat", loadModel=True, numberClasses=7):


        folderName = Path(os.path.abspath(sys.modules[FaceChannel.__module__].__file__)).parent / "TrainedNetworks/"
        print("Class: " + str(folderName))
        input("here")


        if not os.path.exists(folderName):
            os.makedirs(folderName)

            getFrom = self.downloadFrom
            downloadName = folderName +"/trainedNetworks.tar.xz"
            urllib.request.urlretrieve(getFrom, downloadName)

            print ("Download: " +str(downloadName ))
            # with tarfile.open(downloadName) as f:
            #     f.extractall(folderName)
        print ("Folder:" +str(folderName))
        input("here!!!")

        # if type =="Cat":
        #     modelDirectory = "FaceChannel/TrainedNetworks/CategoricalFaceChannel.h5"
        # elif type =="Dim":
        #     modelDirectory = "FaceChannel/TrainedNetworks/DimensionalFaceChannel.h5"
        # else:
        #     raise("Model type not found!")
        #
        # if loadModel:
        #     model = self.loadModel(modelDirectory)
        # else:
        #     if type =="Cat":
        #         model = self.getCategoricalModel(numberClasses)
        #     else:
        #         model = self.getDimensionalModel(numberClasses)
        #
        #
        # self.model = model
        # print ("----- LOADED MODEL: Face Channel ----- ")
        # self.model.summary()
        #
        # self.imageProcessing = imageProcessingUtil.imageProcessingUtil()

    """ Predict """

    def predict(self, images, preprocess=True):

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

    """Load saved Models"""
    def loadModel(self, modelDirectory):
        return load_model(modelDirectory,
                                 custom_objects={'fbeta_score': metrics.fbeta_score, 'rmse': metrics.rmse,
                                                 'recall': metrics.recall, 'precision': metrics.precision,
                                                 'ccc': metrics.ccc})

    """Models from Scratch"""
    def getDimensionalModel(self):
        backbone = self.buildFaceChannel()

        dense = Dense(200, activation="relu", name="denseLayer")(backbone)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

        return Model(inputs=backbone.input, outputs=[arousal_output, valence_output])

    def getCategoricalModel(self, numberClasses):

        backbone = self.buildFaceChannel()

        dense = Dense(200, activation="relu", name="denseLayer")(backbone)

        categoricalOutput = Dense(units=numberClasses, activation='tanh', name='categoricalOutput')(dense)

        return Model(inputs=backbone.input, outputs=categoricalOutput)

    def buildFaceChannel(self):
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