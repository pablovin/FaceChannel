from keras.models import load_model
import numpy

from keras.models import load_model, Model, Input

from keras.models import Sequential, Input
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau, EarlyStopping

from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

import os

import tensorflow as tf
from keras.optimizers import Adam, Adamax, Adagrad, SGD, RMSprop

import keras

from Utils import metrics


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from keras import backend as K

class trainModel:

    IMAGE_SIZE = (64,64)
    BATCH_SIZE = 128
    EPOCHES = 100

    GPU = '/gpu:0' #'/cpu:0'

    @property
    def model(self):
        return self._model

    @property
    def dataLoader(self):
        return self._dataLoader



    def buildModel(self, inputShape):

        def shuntingInhibition(inputs):
            inhibitionDecay = 0.5

            v_c, v_c_inhibit = inputs

            output = (v_c / (inhibitionDecay
                             + v_c_inhibit))

            return output



        keras.backend.set_image_data_format("channels_first")

        nch = 256
        # h = 5
        # reg = keras.regularizers.L1L2(1e-7, 1e-7)
        #
        # model = Sequential()

        print ("Input shape:" + str(inputShape))

        inputShape = numpy.array((1,64,64)).astype(numpy.int32)
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

    def buildDimensionalModel(self, inputShape):

        backbone = self.buildModel(inputShape)

        dense = Dense(200, activation="relu", name="denseLayer")(backbone)

        arousal_output = Dense(units=1, activation='tanh', name='arousal_output')(dense)
        valence_output = Dense(units=1, activation='tanh', name='valence_output')(dense)

        self._model = Model(inputs=backbone.input, outputs=[arousal_output, valence_output])


    def buildCategoricalModel(self, inputShape, numberClasses):

        backbone = self.buildModel(inputShape)

        dense = Dense(200, activation="relu", name="denseLayer")(backbone)

        categoricalOutput = Dense(units=numberClasses, activation='tanh', name='categoricalOutput')(dense)

        self._model = Model(inputs=backbone.input, outputs=categoricalOutput)



    def trainModelCategorical(self, images, categoriesTrain, saveDirectory):


        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss={'mean_squared_error'},
                           optimizer=optimizer,
                           metrics=[metrics.ccc])

        print ("Training:")

        self._model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5,
                                      min_lr=0.0001, verbose=1)


        self.model.fit([images],  [categoriesTrain], batch_size=self.BATCH_SIZE, epochs=self.EPOCHES, callbacks=[reduce_lr])

        self.model.save(saveDirectory + "/FaceChannelCategorical.h5")


    def trainModelDimensional(self, images, arousalTrain, valenceTrain, saveDirectory):


        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        self.model.compile(loss={'arousal_output':'mean_squared_error', 'valence_output':'mean_squared_error'},
                           optimizer=optimizer,
                           metrics=[metrics.ccc])

        print ("Training:")

        self._model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5,
                                      min_lr=0.0001, verbose=1)


        self.model.fit([images],  [arousalTrain, valenceTrain], batch_size=self.BATCH_SIZE, epochs=self.EPOCHES, callbacks=[reduce_lr])

        self.model.save(saveDirectory + "/FaceChannelDimensional.h5")
