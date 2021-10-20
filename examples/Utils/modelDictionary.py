
class CategoricaModel:
    modelname = "Categorical Trained on FER+"
    modelDirectory = "TrainedNetworks/CategoricalFaceChannel.h5"
    modelType = "Categorical"
    classsesOrder = ["Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt"]
    classesColor = [(255, 255, 255), (0, 255, 0),  (0, 222, 255), (255, 0, 0), (0, 0, 255), (255, 0, 144), (0, 144, 255), (75, 75, 96)]



class DimensionalModel:
    modelname = "Arousal and Valence Trained on AffectNet"
    modelDirectory = "TrainedNetworks/DimensionalFaceChannel.h5"
    modelType = "Dimensional"
    classsesOrder = ["Arousal", "Valence"]
    classesColor = [(0, 255, 0), (255, 0, 0)]