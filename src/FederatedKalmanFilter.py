import numpy as np


class FederatedKalmanFilter:
    def __init__(self):
        self.__initalized = False
        self.__results = []
        self.__prior = []
        self.__priors = []
        self.__priorCov = []
        self.__posterior = []
        self.__posteriorCov = []
        self.__previousPosterior = []
        self.__previousPosteriorCov = []
        return

    def predict(self):
        prediction = np.array([1, 1])

        self.__results.append(prediction)

    def update(self, z):
        return

    def getPosteriorCov(self):
        return self.__posteriorCov

    def getResults(self):
        return self.__results

    def getLastResult(self):
        return self.__results[-1]
