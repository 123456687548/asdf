import numpy as np
from Modus import FilterModus

class DistributedKalmanFilter:
    def __init__(self):
        self.__initalized = False
        self.__S = 0
        self.__results = []
        self.__prior = []
        self.__priors = []
        self.__priorCov = []
        self.__posterior = []
        self.__posteriorCov = []
        self.__globalizedPosterior = []
        self.__globalizedPosteriorCov = []
        self.__I = np.array([[1, 0], [0, 1]])
        self.__O = np.array([[0, 0], [0, 0]])
        return

    def initialize(self, measurement, S):
        self.__S = S
        self.__globalizedPosterior = np.array(np.append(measurement, [0,0]))
        self.__globalizedPosteriorCov = np.block([[self.__I, self.__O], [self.__O, self.__I]])
        self.__initalized = True

    def isInitialized(self):
        return self.__initalized

    def globalization(self, sensors):
        globalizedPosteriorCov = np.zeros((4,4))

        for sensor in sensors:
            globalizedPosteriorCov += np.linalg.inv(sensor.getLastPosteriorCov(FilterModus.DISTRIBUTED_KALMAN_FILTER)) 
        self.__globalizedPosteriorCov = self.__S * np.linalg.inv(globalizedPosteriorCov)

        self.__globalizedPosterior = np.dot(globalizedPosteriorCov, np.dot(np.linalg.inv(self.__posteriorCov), self.__posterior))

    def predict(self, dt):
        S = self.__S
        I = self.__I
        O = self.__O
        F = np.block([[I, dt*I], [O, I]])
        Q = np.block([[0.25*dt**4*I, 0.3*dt**2*I], [0.3*dt**2*I, dt*I]])

        self.__prior = np.matmul(F, self.__globalizedPosterior)
        self.__priors.append(self.__prior)
        self.__priorCov = np.dot(F, np.dot(self.__globalizedPosteriorCov, F.T)) + S * Q

    def update(self, measurement):
        R = np.array([[10, 0], [0, 10]])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        v = measurement - np.dot(H, self.__prior)
        S = np.dot(H, np.dot(self.__priorCov, H.T)) + R
        W = np.dot(self.__priorCov, np.dot(H.T, np.linalg.inv(S)))

        self.__posterior = self.__prior + np.dot(W, v)
        self.__posteriorCov = self.__priorCov - np.dot(W, np.dot(S, W.T))
        self.__results.append(self.__posterior)

    def getPriors(self):
        return self.__priors

    def getPosteriorCov(self):
        return self.__posteriorCov

    def getResults(self):
        return self.__results

    def getLastResult(self):
        return np.array(self.__results[-1])