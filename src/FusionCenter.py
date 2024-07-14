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

    def plot(self, target, modus):
        plt.plot([position[0] for position in target.positions()[:-1]],
                 [position[1] for position in target.positions()[:-1]],
                 label='True target positions')
        match modus:
            case FilterModus.KALMAN_FILTER:
                plt.plot([position[0] for position in self.__kalmanFusions],
                         [position[1] for position in self.__kalmanFusions],
                         label='Fusion (Convex Combination)')
                plt.title('Kalman Filter in Sensors \n and Convex Combination in FC')
                plt.savefig('../plots/KF_Fusion.png')
            case FilterModus.DISTRIBUTED_KALMAN_FILTER:
                plt.plot([position[0] for position in self.__distributedKalmanFusions],
                         [position[1] for position in self.__distributedKalmanFusions],
                         label='Fusion (Convex Combination)')
                plt.title('Distributed Kalman Filter in Sensors \n and Convex Combination in FC')
                plt.savefig('../plots/DKF_Fusion.png')
            case FilterModus.FEDERATED_KALMAN_FILTER:
                plt.plot([position[0] for position in self.__federatedKalmanFusions],
                         [position[1] for position in self.__federatedKalmanFusions],
                         label='Fusion (Convex Combination)')
                plt.title('Federated Kalman Filter in Sensors \n and Convex Combination in FC')
                plt.savefig('../plots/FKF_Fusion.png')
        plt.legend()
        plt.show()
        plt.clf()
