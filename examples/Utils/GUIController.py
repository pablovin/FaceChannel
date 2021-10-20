import numpy
import cv2
import matplotlib.pyplot as plt


class GUIController:

    def createDetectedFacGUI(self, frame, detectedFace, faceFrameColor=None, categoricalClassificationReport=[]):


        faceColor = (0,0,0)
        if not len(categoricalClassificationReport) == 0:

            mainClassification = numpy.argmax(categoricalClassificationReport)
            faceColor =faceFrameColor[mainClassification]
            # print("mainClassification:" + str(mainClassification))
            # print("Face color:" + str(faceColor))

        # Draw Detected Face
        for (x, y, w, h) in detectedFace:
            # image[startY:endY, startX:endX]
            # [startX, startY, endX, endY]
            cv2.rectangle(frame, (x, y), (w, h), faceColor, 2)


        return frame


    def createDimensionalPlotGUI(self, arousals, valences, frame):
        plt.style.use('dark_background')
        fig, axs = plt.subplots(2, figsize=(10,5))
        axs[0].set_title('Arousal')
        axs[1].set_title('Valence')

        axs[0].set_xlim([0,100])
        axs[1].set_xlim([0, 100])

        axs[0].set_ylim([-1,1])
        axs[1].set_ylim([-1, 1])

        axs[0].plot(range(len(arousals)), arousals)
        axs[1].plot(range(len(valences)), valences)

        plt.savefig("plot.png", )
        plt.close(fig)

        plot = cv2.imread("plot.png")

        plot = cv2.resize(plot, (1000, 250))

        frame[500:750, 0:1000] = plot

        return frame

    def createDimensionalEmotionGUI(self, classificationReport, frame, categoricalReport=[], categoricalDictionary=None):


        if not len(categoricalReport) == 0:

            mainClassification = numpy.argmax(categoricalReport)
            pointColor = categoricalDictionary.classesColor[mainClassification]
        else:
            pointColor = (0, 0, 255)


        #Dimensional Report

        cv2.line(frame, (640+170, 120), (640+170, 320), (255, 255, 255), 4)
        cv2.line(frame, (640+85, 210), (640+285, 210), (255, 255, 255), 4)

        cv2.putText(frame, "Calm", (640+150, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Excited", (640+150,335), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "Negative", (640+15, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Positive", (640+295, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        arousal = float(float(classificationReport[0][0][0]) * 100)
        valence = float(float(classificationReport[1][0][0]) * 100)

        #print "Arousal:", arousal
        #print "Valence:", valence

        #arousal,valence
        cv2.circle(frame, (640+185+int(valence), 210+int(arousal)), 5, pointColor, -1)


        return frame


    def createCategoricalEmotionGUI(self, classificationReport, frame, classesColor, classesOrder, initialPosition=0):


        classificationReport = classificationReport*100


        for index,emotion in enumerate(classesOrder):

            emotionClassification = int(classificationReport[int(index)])

            cv2.putText(frame, emotion, (640+5, initialPosition+15+int(index)*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  classesColor[index], 1)

            cv2.rectangle(frame, (640+100, initialPosition+5+int(index)*20), (int(640+100 + emotionClassification), initialPosition+20+int(index)*20), classesColor[index], -1)
            cv2.putText(frame, str(emotionClassification) + "%", (int(640+105 + emotionClassification + 10), initialPosition+20+int(index)*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, classesColor[index], 1)

        return frame
