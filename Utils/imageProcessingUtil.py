import cv2
import numpy
import dlib


class imageProcessingUtil:



    faceDetectionMaximumFrequency = 10


    @property
    def faceDetector(self):
        return self._faceDetector

    def __init__(self):

        prototxtPath = "TrainedNetworks/faceDetector/deploy.prototxt"
        weightsPath = "TrainedNetworks/faceDetector/res10_300x300_ssd_iter_140000.caffemodel"

        self._faceDetector=  cv2.dnn.readNet(prototxtPath, weightsPath)


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

        (h, w) = image.shape[:2]

        # print ("Image shape:" + str((h,w)))
        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))

        self._faceDetector.setInput(blob)
        detections = self._faceDetector.forward()

        face = image

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it

                face = image[startY:endY, startX:endX]
                dets = [[startX,startY,  endX, endY]]

                # print("--shape Image:" + str(image.shape))
                # print("--shape Face:" + str(face.shape))
                #
                # print("--Detected XY: (" + str(startX) + "),(" + str(startY) + "),(" + str(startX) + "+" + str(
                #     endX) + ") - " + str(startY) + "+" + str(endY))


                self.previouslyDetectedface = dets

        dets = self.previouslyDetectedface



        return dets, face


