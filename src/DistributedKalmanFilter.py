import numpy as np


class DistributedKalmanFilter:
    def __init__(self):
        self.__predictions = []
        return

    def predict(self):
        prediction = np.array([1, 1])
        
        self.__predictions.append(prediction)

    def update(self, z):
        return

    def getPredictions(self):
        return self.__predictions

    def getLastPrediction(self):
        return self.__predictions[:-1]