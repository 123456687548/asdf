import numpy as np
import matplotlib.pyplot as plt

from Modus import FilterModus


class FusionCenter:
    def __init__(self):
        self.__sensors = []
        self.__kalmanFusions = []
        self.__federatedKalmanFusions = []
        self.__distributedKalmanFusions = []

    def convexCombination(self, modus):
        fused_x = 0
        fused_P = 0
        for sensor in self.__sensors:
            cov = sensor.getLastPosteriorCov(modus)
            fused_P += np.linalg.inv(cov)

        fused_P = np.linalg.inv(fused_P)

        for sensor in self.__sensors:
            x = sensor.getLastPosterior(modus)
            fused_x += np.dot(np.linalg.inv(sensor.getLastPosteriorCov(modus)), x)

        fused_x = np.dot(fused_P, fused_x)

        match modus:
            case FilterModus.KALMAN_FILTER:
                self.__kalmanFusions.append(fused_x)
            case FilterModus.DISTRIBUTED_KALMAN_FILTER:
                self.__distributedKalmanFusions.append(fused_x)
            case FilterModus.FEDERATED_KALMAN_FILTER:
                self.__federatedKalmanFusions.append(fused_x)

    def addSensor(self, sensor):
        self.__sensors.append(sensor)

    def plot(self, target, modus, fileName, title):
        fig = plt.figure(figsize=(5, 5))
        plt.plot([position[0] for position in target.positions()[:-1]],
                 [position[1] for position in target.positions()[:-1]],
                 label='True target positions')
        data = []
        match modus:
            case FilterModus.KALMAN_FILTER:
                data = self.__kalmanFusions
            case FilterModus.DISTRIBUTED_KALMAN_FILTER:
                data = self.__distributedKalmanFusions
            case FilterModus.FEDERATED_KALMAN_FILTER:
                data = self.__federatedKalmanFusions
        plt.plot([position[0] for position in data],
                 [position[1] for position in data],
                 label='Fusion (Convex Combination)')
        plt.title(title)
        plt.legend()
        plt.savefig(fileName, dpi=fig.dpi)
        plt.show()
        plt.clf()
