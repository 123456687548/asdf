import numpy as np
from KalmanFilter import KalmanFilter


class Sensor:
    kalmanFilter = KalmanFilter()
    measurements = []
    predictions = []

    def __init__(self, target):
        self.__target = target

    def noise(self):
        # todo
        return 0

    def H(self):
        """Measurement function"""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def measure(self):
        measurement = np.matmul(self.H(), self.__target.state()) + self.noise()
        self.measurements.append(measurement)
        return measurement

    def filter(self, measurement):
        self.predictions.append(self.kalmanFilter.predict())
        self.kalmanFilter.update(0)

    def getLastPrediction(self):
        return self.predictions[:-1]
