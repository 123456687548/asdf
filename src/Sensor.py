import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter


class Sensor:

    def __init__(self, target, name):
        self.__name = name
        self.__kalmanFilter = KalmanFilter()
        self.__measurements = []
        self.__predictions = []
        self.__target = target

    def noise(self):
        # todo
        return 0

    def H(self):
        """Measurement function"""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def measure(self):
        measurement = np.matmul(self.H(), self.__target.state()) + self.noise()
        self.__measurements.append(measurement)
        return measurement

    def filter(self, measurement):
        self.__predictions.append(self.__kalmanFilter.predict())
        self.__kalmanFilter.update(0)

    def getLastPrediction(self):
        return self.__predictions[:-1]

    def plot(self):
        plt.plot(range(len(self.__target.positions())), np.array(self.__target.positions()),
                 label='True target positions')
        plt.plot(range(len(self.__measurements)), np.array(self.__measurements), label='Sensor measurements')
        plt.plot(range(len(self.__predictions)), np.array(self.__predictions), label='Kalman filter prediction')
        plt.legend()
        plt.title(self.__name)
        plt.show()
