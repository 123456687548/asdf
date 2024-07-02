import numpy as np

class FusionCenter:
    def __init__(self):
        self.__sensors = []
        self.__

    def convexCombination(self):
        fused_x = 0
        fused_P = 0
        for sensor in self.__sensors:
            fused_P += np.linalg.inv(sensor.getKalmanFilter().getLastPosterior())

        fused_P = np.linalg.inv(fused_P)

        for sensor in self.__sensors:
            x = sensor.getKalmanFilter().getLastResult()
            fused_x += np.dot(np.linalg.inv(sensor.getKalmanFilter().getLastPosterior(), x))

        fused_x = np.dot(fused_P, fused_x)

        return fused_x

    def addSensor(self, sensor):
        self.__sensors.append(sensor)
