"""
Emotion Recognition - Frame-Based Face Channel

__author__ = "Pablo Barros"

__version__ = "0.1"
__maintainer__ = "Pablo Barros"
__email__ = "barros@informatik.uni-hamburg.de"

More information about the implementation of the model:

Barros, P., Churamani, N., & Sciutti, A. (2020). The FaceChannel: A Light-weight Deep Neural Network for Facial Expression Recognition. arXiv preprint arXiv:2004.08195.

Barros, P., & Wermter, S. (2016). Developing crossmodal expression recognition based on a deep neural model. Adaptive behavior, 24(5), 373-396.
http://journals.sagepub.com/doi/full/10.1177/1059712316664017

"""

import numpy
import cv2
from Utils import imageProcessingUtil, modelDictionary, modelLoader, GUIController
import os
import time

import csv

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)



loadFramesFrom = "/home/pablo/Documents/Datasets/testFC/frames" #Folde where the videos are
saveCSVFiles = "/home/pablo/Documents/Datasets/testFC/frames" #Folder that will hold the .csv files

modelDimensional = modelLoader.modelLoader(modelDictionary.DimensionalModel) #Load neural network

imageProcessing = imageProcessingUtil.imageProcessingUtil()



"""
Opens the .csv file 
"""
with open(saveCSVFiles+".csv", mode='a') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    employee_writer.writerow(['Frame', 'Arousal', 'Valence'])

    for frame in os.listdir(loadFramesFrom):  # for each frame inside this folder
        print ("Started Frame:" + str(loadFramesFrom + "/" + frame))

        frameDirectory = str(loadFramesFrom + "/" + frame)
        frame = cv2.imread(loadFramesFrom + "/" + frame)

        facePoints, face = imageProcessing.detectFace(frame) #detect a face

        if not len(face) == 0:   # If a face is detected

            face = imageProcessing.preProcess(face,imageSize=(64,64))     # pre-process the face

            dimensionalRecognition = numpy.array(modelDimensional.classify(face))    # Obtain dimensional classification
        else: #if there is no face
            dimensionalRecognition = [-99,-99]
        # print ("DImensional: " + str(dimensionalRecognition))
        employee_writer.writerow([frameDirectory, dimensionalRecognition[0][0][0], dimensionalRecognition[1][0][0]])
        # print("-- Frame: " + str(frameCount))
        # print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


    cv2.destroyAllWindows()