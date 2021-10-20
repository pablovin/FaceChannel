"""
Emotion Recognition - Vision-Frame-Based Face Channel

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
from Utils import imageProcessingUtil, modelDictionary, modelLoader
import os
import time

import csv

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


loadVideosFrom = "/home/pablo/Documents/Datasets/OMG-Emotion/SonyVIdeo/original/all" #Folde where the videos are
saveCSVFiles = "//home/pablo/Documents/Datasets/OMG-Emotion/SonyVIdeo/original/" #Folder that will hold the .csv files

modelDimensional = modelLoader.modelLoader(modelDictionary.DimensionalModel) #Load neural network

imageProcessing = imageProcessingUtil.imageProcessingUtil()


for videoDirectory in os.listdir(loadVideosFrom): #for each video inside this folder

    videoTime = time.time()  # start time of the loop

    cap = cv2.VideoCapture(loadVideosFrom+"/"+videoDirectory) #open the video

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    """
    Opens the .csv file 
    """
    with open(saveCSVFiles+"/"+videoDirectory+".csv", mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        employee_writer.writerow(['Frame', 'Arousal', 'Valence'])

        frameCount = 0

        fpsCounter = []
        print ("Started Video:" + str(videoDirectory) +" - Total Frames:" + str(total))
        while(cap.isOpened() and not frameCount == total): #for each frame in this video
            start_time = time.time()  # start time of the loop
            ret, frame = cap.read()
            frameCount = frameCount + 1
            # print ("Frame count:" + str(frameCount))
            if type(frame) is numpy.ndarray:
                facePoints, face = imageProcessing.detectFace(frame) #detect a face

                if not len(face) == 0:   # If a face is detected

                    face = imageProcessing.preProcess(face,imageSize=(64,64))     # pre-process the face

                    dimensionalRecognition = numpy.array(modelDimensional.classify(face))    # Obtain dimensional classification
                else: #if there is no face
                    dimensionalRecognition = [-99,-99]

                # print ("DImensional: " + str(dimensionalRecognition))
                employee_writer.writerow([int(frameCount), dimensionalRecognition[0][0][0], dimensionalRecognition[1][0][0]])
                fpsCounter.append(1.0 / (time.time() - start_time))
                # print("-- Frame: " + str(frameCount))
                # print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop


    fpsAvg = numpy.array(fpsCounter).mean()
    videoTime =  (time.time() - videoTime)

    print("Finished Video: " + str(videoDirectory) +"- FPS:" + str(fpsAvg) + " - Time:" + str(videoTime) +" seconds")

    cap.release()
    cv2.destroyAllWindows()