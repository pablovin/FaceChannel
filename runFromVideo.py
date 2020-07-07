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



loadVideosFrom = "/home/pablo/Documents/Datasets/wristbot/videos" #Folde where the videos are
saveCSVFiles = "/home/pablo/Documents/Datasets/wristbot/csv" #Folder that will hold the .csv files

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