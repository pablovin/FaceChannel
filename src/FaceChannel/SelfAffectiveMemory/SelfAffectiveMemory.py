"""
SelfAffectiveMemory.py
====================================
Self-Affective memory model.
"""


# -*- coding: utf-8 -*-
import numpy

from FaceChannel.SelfAffectiveMemory import Standard_GWR

from FaceChannel.FaceChannelV1.FaceChannelV1 import FaceChannelV1

class SelfAffectiveMemory():

    numberOfEpochs = 5
    """Number of traning epoches for the GWR"""

    insertionThreshold = 0.9
    """Activation threshold for node insertion"""

    learningRateBMU = 0.35
    """Learning rate of the best-matching unit (BMU)"""

    learningRateNeighbors = 0.76  #
    """Learning rate of the BMU's topological neighbors"""

    def __init__(self, numberOfEpochs=5,insertionThreshold=0.9,learningRateBMU=0.35,learningRateNeighbors=0.76):
        """Constructor of the SelfAffectiveMemory

                :param numberOfEpochs: Number of traning epoches for the GWR.
                :param insertionThreshold: Activation threshold for node insertion.
                :param learningRateBMU: Learning rate of the best-matching unit (BMU)
                :param learningRateNeighbors: Learning rate of the BMU's topological neighbors.
        """


        self.faceChannelV1 = FaceChannelV1("Dim", loadModel=True)
        self._isBuilt = False
        self.numberOfEpochs, self.insertionThreshold, self.learningRateBMU, self.learningRateNeighbors = numberOfEpochs,insertionThreshold,learningRateBMU,learningRateNeighbors


    def buildAffectiveMemory(self, dataTrain):
        """Method that activey builds the current affective memory

                :param dataTrain: initial training data as an ndarray.

        """

        dataTrain = numpy.array([dataTrain, dataTrain, dataTrain])
        standardGWR = Standard_GWR.AssociativeGWR()
        standardGWR.initNetwork(dataTrain,1)
        self._model = standardGWR
        self._isBuilt = True

    def train(self, dataPointsTrain):
        """Method that trains the affective memory online
                :param dataTrain: initial training data as an ndarray.

        """
        dataPointsTrain = numpy.array([dataPointsTrain])
        self._model.trainAGWR(dataPointsTrain, self.numberOfEpochs,self.insertionThreshold,self.learningRateBMU,self.learningRateNeighbors)

    def getNodes(self):
        """Method that returns all the current nodes of the affective memory
                :return: a tuple of nodes and ages of each node.
                :rtype: ndarray tuple
         """
        neuronAge = numpy.copy(self._model.habn)
        return self._model.weights, neuronAge

    def predict(self, images, preprocess=False):

        """Method that predicts the current arousal and valence of a given image or set of images.
                  as the affective memory is an online learning method, every given frame must be temporaly subsequent to the
                  previous ones. It relies on the FaceChannelV1 for feature extraction.

                :param images: The images as one or a list of ndarray.
                :param preprocess: If the image is already pre-processed or not.
                                   a pre-processed image has a format of (64,64,1).

                :return: The prediction of the given image(s) as a ndarray
                :rtype: ndarray
         """


        outputFaceChannel = self.faceChannelV1.predict(images, preprocess)
        # print ("Output:" + str(numpy.array(outputFaceChannel)[:,0,0]))
        affectiveMemoryInput = numpy.array(outputFaceChannel)[:,0,0].flatten()

        if self._isBuilt:
            self.train(affectiveMemoryInput)
        else:
            self.buildAffectiveMemory(affectiveMemoryInput)

        affectiveMemoryNodes = self.getNodes()[0]


        arousal = numpy.array(affectiveMemoryNodes)[:, 0]
        valence = numpy.array(affectiveMemoryNodes)[:, 1]
        averageArousal = numpy.mean(arousal)
        averageValence = numpy.mean(valence)

        return [averageArousal, averageValence]