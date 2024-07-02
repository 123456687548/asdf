import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter
from FederatedKalmanFilter import FederatedKalmanFilter
from DistributedKalmanFilter import DistributedKalmanFilter


class Sensor:

    def __init__(self, target, name):
        self.__name = name
        self.__kalmanFilter = KalmanFilter()
        self.__federatedKalmanFilter = FederatedKalmanFilter()
        self.__distributedKalmanFilter = DistributedKalmanFilter()
        self.__measurements = []
        self.__target = target

    def noise(self):
        return np.random.multivariate_normal([0, 0], np.array([[10, 0], [0, 10]]))

    def H(self):
        """Measurement function"""
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    def measure(self):
        noise = self.noise()
        measurement = np.matmul(self.H(), self.__target.state()) + noise
        self.__measurements.append(measurement)
        return measurement

    def kalmanFilter(self, measurement):
        if not self.__kalmanFilter.isInitialized():
            self.__kalmanFilter.initialize(measurement)
        self.__kalmanFilter.predict()
        self.__kalmanFilter.update(measurement)

    def plot(self):
        plt.plot([position[0] for position in self.__target.positions()[:-1]],
                 [position[1] for position in self.__target.positions()[:-1]],
                 label='True target positions')
        plt.plot([measurement[0] for measurement in self.__measurements],
                 [measurement[1] for measurement in self.__measurements], label='Sensor measurements')
        plt.plot([result[0] for result in self.__kalmanFilter.getResults()],
                 [result[1] for result in self.__kalmanFilter.getResults()], label='Kalman filter')
        plt.plot([prior[0] for prior in self.__kalmanFilter.getPriors()],
                 [prior[1] for prior in self.__kalmanFilter.getPriors()], label='Kalman prediction')

        plt.legend()
        plt.title(self.__name)
        plt.show()
        plt.savefig(f'{self.__name}.png')
