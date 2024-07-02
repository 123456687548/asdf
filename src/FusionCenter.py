import numpy as np


class FusionCenter:
    def __init__(self):
        self.__sensors = []

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

        return fused_x

    def addSensor(self, sensor):
        self.__sensors.append(sensor)
