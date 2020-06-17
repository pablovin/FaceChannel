from keras.models import load_model
import numpy

from keras.models import load_model, Model, Input

import imageProcessingUtil

import os

import tensorflow as tf

import metrics

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#from keras import backend as K

class modelLoader:

    IMAGE_SIZE = (64,64)
    BATCH_SIZE = 32

    GPU = '/gpu:0' #'/cpu:0'

    @property
    def modelDictionary(self):
        return self._modelDictionary

    @property
    def model(self):
        return self._model

    @property
    def dataLoader(self):
        return self._dataLoader


    def __init__(self, modelDictionary):

        self._modelDictionary = modelDictionary
        self._dataLoader = imageProcessingUtil.imageProcessingUtil()

        self.loadModel()


    def loadModel(self):

        self._model = load_model(self.modelDictionary.modelDirectory, custom_objects={'fbeta_score': metrics.fbeta_score, 'rmse': metrics.rmse,'recall': metrics.recall, 'precision': metrics.precision, 'ccc': metrics.ccc})
        self._model.summary()


    def classify(self, image):

        classification = self.model.predict(numpy.array([image]),batch_size=self.BATCH_SIZE, verbose=0)

        return classification



    def getDense(self, image):
        denseLayerOutput = self.model.get_layer(name="denseLayer").output

        classifier = Model(inputs=self.model.inputs, outputs=[denseLayerOutput])

        denseRepresentation = classifier.predict(numpy.array([image]), batch_size=self.BATCH_SIZE)

        return denseRepresentation


