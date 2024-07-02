import numpy as np


class KalmanFilter:
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

    def initialize(self, measurement):
        self.__previousPosterior = measurement
        self.__previousPosteriorCov = np.array([[100, 0], [0, 100]])
        self.__initalized = True

    def isInitialized(self):
        return self.__initalized

    def predict(self):
        F = np.array([[1, 0.002], [0.0008, 1]])
        Q = np.array([[10, 0.02], [0.02, 10]])

        self.__prior = np.dot(F, self.__previousPosterior)
        self.__priors.append(self.__prior)
        self.__priorCov = np.dot(F, np.dot(self.__previousPosteriorCov, F.T)) + Q

    def update(self, measurement):
        R = np.array([[10, 0], [0, 10]])
        H = np.array([[1, 0], [0, 1]])
        v = measurement - np.dot(H, self.__prior)
        S = np.dot(H, np.dot(self.__priorCov, H.T)) + R
        W = np.dot(self.__priorCov, np.dot(H.T, np.linalg.inv(S)))

        self.__posterior = self.__prior + np.dot(W, v)
        self.__posteriorCov = self.__priorCov - np.dot(W, np.dot(S, W.T))
        self.__results.append(self.__posterior)

        self.__previousPosterior = self.__posterior
        self.__previousPosteriorCov = self.__posteriorCov

    def getPriors(self):
        return self.__priors

    def getResults(self):
        return self.__results

    def getLastResult(self):
        return self.__results[:-1]
