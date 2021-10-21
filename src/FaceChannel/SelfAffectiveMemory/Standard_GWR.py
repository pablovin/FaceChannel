"""
Associative GWR based on (Marsland et al. 2002)'s Grow-When-Required (Python 3)
@last-modified: 8 September 2018
@author: German I. Parisi (german.parisi@gmail.com)
Please cite this paper: Parisi, G.I., Weber, C., Wermter, S. (2015) Self-Organizing Neural Integration of Pose-Motion Features for Human Action Recognition. Frontiers in Neurorobotics, 9(3).
"""

import scipy.spatial
import numpy as np
import math


class AssociativeGWR:


    def initNetwork(self, dataSet, initMethod, logManager=None):
        self.numNodes = 2
        self.dimension = dataSet.shape[1]
        self.weights = np.zeros((self.numNodes, self.dimension))
        self.edges = np.ones((self.numNodes, self.numNodes))
        self.ages = np.zeros((self.numNodes, self.numNodes))
        self.habn = np.ones(self.numNodes)

        self.logManager = logManager

        if (initMethod):
            self.weights[0] = dataSet[0]
            self.weights[1] = dataSet[1]
        else:
            randomIndex = np.random.randint(0, dataSet.shape[0], 2)
            self.weights[0] = dataSet[randomIndex[0]]
            self.weights[1] = dataSet[randomIndex[1]]

    def computeDistance(self, x, y, m):
        if m:
            return np.linalg.norm(x - y)  # np.sqrt(np.sum((x-y)**2))
        else:
            return scipy.spatial.distance.cosine(x, y)

    def habituateNeuron(self, index, tau):
        self.habn[index] += (tau * 1.05 * (1. - self.habn[index]) - tau)

    def updateNeuralWeight(self, input, index, epsilon):
        delta = np.array([np.dot((input - self.weights[index]), epsilon)]) * self.habn[index]
        self.weights[index] = self.weights[index] + delta

    def updateEdges(self, fi, si):
        neighboursFirst = np.nonzero(self.edges[fi])
        if (len(neighboursFirst[0]) >= self.maxNeighbours):
            remIndex = -1
            maxAgeNeighbour = 0
            for u in range(0, len(neighboursFirst[0])):
                if (self.ages[fi, neighboursFirst[0][u]] > maxAgeNeighbour):
                    maxAgeNeighbour = self.ages[fi, neighboursFirst[0][u]]
                    remIndex = neighboursFirst[0][u]
            self.edges[fi, remIndex] = 0
            self.edges[remIndex, fi] = 0
        self.edges[fi, si] = 1

    def removeOldEdges(self):
        for i in range(0, self.numNodes):
            neighbours = np.nonzero(self.edges[i])
            for j in range(0, len(neighbours[0])):
                if (self.ages[i, j] >= self.maxAge):
                    self.edges[i, j] = 0
                    self.edges[j, i] = 0

    def removeIsolatedNeurons(self):

        indCount = 0
        while (indCount < self.numNodes):
            neighbours = np.nonzero(self.edges[indCount])
            if (len(neighbours[0]) < 1):
                self.weights = np.delete(self.weights, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=0)
                self.edges = np.delete(self.edges, indCount, axis=1)
                self.ages = np.delete(self.ages, indCount, axis=0)
                self.ages = np.delete(self.ages, indCount, axis=1)
                self.habn = np.delete(self.habn, indCount)
                self.numNodes = self.weights.shape[0]
            else:
                indCount += 1

    def saveWeights(self, saveLocal):

        np.save(saveLocal,self.weights)

    def loadWeights(self, loadLocal):



        self.weights = np.load(loadLocal)


    def trainAGWR(self, dataSet, mE, iT, eeB, eeN):
        # print ("Dataset Shape:", dataSet.shape)
        self.samples, self.dimension = dataSet.shape
        self.maxEpochs = mE
        self.insertionThreshold = iT
        self.epsilon_b = eeB
        self.epsilon_n = eeN

        self.distanceMetric = 1
        self.habThreshold = 0.3
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.maxNodes = 5000  # OK for batch, bad for incremental
        self.maxNeighbours = 6
        self.maxAge = 100
        self.newNodeValue = 0.5
        self.aIncreaseFactor = 1
        self.aDecreaseFactor = 0.1

        # Start training
        epochs = 0
        errorCounter = np.zeros(self.maxEpochs)
        while (epochs < self.maxEpochs):
            epochs += 1
            for iteration in range(0, self.samples):
                # Generate input sample
                input = dataSet[iteration]

                # Find the best and second-best matching neurons
                distances = np.zeros(self.numNodes)
                for i in range(0, self.numNodes):
                    distances[i] = self.computeDistance(self.weights[i], input, self.distanceMetric)

                firstIndex = np.argmin(distances)
                firstDistance = distances[firstIndex]
                distances[firstIndex] = 99999
                secondIndex = np.argmin(distances)

                errorCounter[epochs - 1] += firstDistance

                # Compute network activity
                a = math.exp(-firstDistance)

                if ((a < self.insertionThreshold) and (self.habn[firstIndex] < self.habThreshold) and (
                        self.numNodes < self.maxNodes)):
                    # Add new neuron
                    newWeight = np.array([np.dot(self.weights[firstIndex] + input, self.newNodeValue)])
                    self.weights = np.concatenate((self.weights, newWeight), axis=0)
                    newIndex = self.numNodes
                    self.numNodes += 1
                    self.habn.resize(self.numNodes)
                    self.habn[newIndex] = 1

                    # Update edges
                    self.edges.resize((self.numNodes, self.numNodes))
                    self.edges[firstIndex, secondIndex] = 0
                    self.edges[secondIndex, firstIndex] = 0
                    self.edges[firstIndex, newIndex] = 1
                    self.edges[newIndex, firstIndex] = 1
                    self.edges[newIndex, secondIndex] = 1
                    self.edges[secondIndex, newIndex] = 1

                    # Update ages
                    self.ages.resize((self.numNodes, self.numNodes))
                    self.ages += 1
                    self.ages[firstIndex, newIndex] = 0
                    self.ages[newIndex, firstIndex] = 0
                    self.ages[newIndex, secondIndex] = 0
                    self.ages[secondIndex, newIndex] = 0

                    #print "(++", str(self.numNodes), ')'
                else:
                    # Adapt weights
                    self.updateNeuralWeight(input, firstIndex, self.epsilon_b)

                    # Habituate BMU
                    self.habituateNeuron(firstIndex, self.tau_b)

                    # Update ages
                    self.ages += 1
                    self.ages[firstIndex, secondIndex] = 0
                    self.ages[secondIndex, firstIndex] = 0

                    # Update edges
                    self.updateEdges(firstIndex, secondIndex)
                    self.updateEdges(secondIndex, firstIndex)

                    # Update topological neighbours
                    neighboursFirst = np.nonzero(self.edges[firstIndex])
                    for z in range(0, len(neighboursFirst[0])):
                        neIndex = neighboursFirst[0][z]
                        self.updateNeuralWeight(input, neIndex, self.epsilon_n)
                        self.habituateNeuron(neIndex, self.tau_n)

            # Remove old edges
            self.removeOldEdges()

            # Compute metrics
            errorCounter[epochs - 1] /= self.samples


        # Remove isolated neurons
        self.removeIsolatedNeurons()



    # Test GWR ################################################################

    def getBMU(self, dataSet):
        samples = dataSet.shape[0]
        bmus = -np.ones(samples)
        nNodes = len(self.weights)
        distance = np.zeros(nNodes)
        activations = np.zeros(samples)

        for iterat in range(0, samples):
            input = dataSet[iterat]

            for i in range(0, nNodes):
                distance[i] = self.computeDistance(self.weights[i], input, 1)

            firstIndex = distance.argmin()
            firstDistance = distance.min()
            activations[iterat] = math.exp(-firstDistance)
            bmus[iterat] = firstIndex

        weights = []
        for bmu in bmus:
            weights.append(self.weights[int(bmu)])

        return bmus, np.array(weights)

    def computeAccuracy(self, labelSet, blabels):
        goodCounter = 0

        for iterat in range(0, len(labelSet)):
            if (labelSet[iterat] == blabels[iterat]):
                goodCounter += 1

        return (100 * goodCounter) / len(labelSet)