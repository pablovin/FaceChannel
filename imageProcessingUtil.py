import cv2
import numpy
import dlib


class imageProcessingUtil:



    faceDetectionMaximumFrequency = 10


    @property
    def faceDetector(self):
        return self._faceDetector

    def __init__(self, faceDetectionMaximumFrequency=10):

        self._faceDetector = dlib.get_frontal_face_detector()
        self.faceDetectionMaximumFrequency = faceDetectionMaximumFrequency


    def preProcess(self, image, imageSize= (64,64)):

        image = numpy.array(image)


        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = numpy.array(cv2.resize(image, imageSize))


        image = numpy.expand_dims(image, axis=0)


        image = image.astype('float32')

        image /= 255

        return image


    previouslyDetectedface = None
    currentFaceDetectionFrequency = -1

    def detectFace(self, image):

        #print "CurrentFace:", self.currentFaceDetectionFrequency
        if self.currentFaceDetectionFrequency == self.faceDetectionMaximumFrequency or self.currentFaceDetectionFrequency == -1 :
            dets = self.faceDetector(image, 1)
            self.previouslyDetectedFace = dets
            self.currentFaceDetectionFrequency = 0
        else:
            dets = self.previouslyDetectedFace


        self.currentFaceDetectionFrequency = self.currentFaceDetectionFrequency+1

        face = []

        for k, d in enumerate(dets):
            face = image[d.top():d.bottom(), d.left():d.right()]
            break


        return dets, face


