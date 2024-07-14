import math
import os

from Target import Target
from Sensor import Sensor
from FusionCenter import FusionCenter
from Modus import FilterModus
import matplotlib.pyplot as plt

SENSOR_AMOUNT = 4
SAMPLES = 100


def main():
    try:
        os.mkdir('../plots')
    except OSError:
        pass

    target = Target([50, 100], [2, 10])
    fusionCenter = FusionCenter()
    sensors = []

    # create Sensors
    for i in range(0, SENSOR_AMOUNT):
        sensor = Sensor(target, f'Sensor {i + 1}')
        sensors.append(sensor)
        fusionCenter.addSensor(sensor)

    for _ in range(0, SAMPLES):
        for sensor in sensors:
            measurement = sensor.measure()
            sensor.kalmanFilter(measurement)
            sensor.federatedKalmanFilter(measurement, len(sensors))
            sensor.distributedKalmanFilter(measurement, len(sensors))

        for sensor in sensors:
            sensor.distributedKalmanFilterGlobalization(sensors)

        fusionCenter.convexCombination(FilterModus.KALMAN_FILTER)
        fusionCenter.convexCombination(FilterModus.FEDERATED_KALMAN_FILTER)
        fusionCenter.convexCombination(FilterModus.DISTRIBUTED_KALMAN_FILTER)

        target.move()

    # Plot measurement, prediction and filtering for Kalman Filter
    subplot(sensors, FilterModus.KALMAN_FILTER, '../plots/KalmanFilter.png')

    # Plot KF_Fusion
    fusionCenter.plot(target, FilterModus.KALMAN_FILTER)

    # Plot measurement, prediction and filtering for Federated Kalman Filter
    subplot(sensors, FilterModus.FEDERATED_KALMAN_FILTER, '../plots/FederatedKalmanFilter.png')

    # Plot FKF_Fusion
    fusionCenter.plot(target, FilterModus.FEDERATED_KALMAN_FILTER)

    # Plot measurement, prediction and filtering for Distributed Kalman Filter
    subplot(sensors, FilterModus.DISTRIBUTED_KALMAN_FILTER, '../plots/DistributedKalmanFilter.png')

    # Plot DKF_Fusion
    fusionCenter.plot(target, FilterModus.DISTRIBUTED_KALMAN_FILTER)


def subplot(sensors, modus, fileName):
    for i, _ in enumerate(sensors):
        plt.subplot(math.floor(len(sensors) / 2), math.ceil(len(sensors) / 2), i + 1)
        match modus:
            case FilterModus.KALMAN_FILTER:
                sensors[i].plotKalmanFilter()
            case FilterModus.DISTRIBUTED_KALMAN_FILTER:
                sensors[i].plotDistributedKalmanFilter()
            case FilterModus.FEDERATED_KALMAN_FILTER:
                sensors[i].plotFederatedKalmanFilter()
    plt.legend()  # todo
    plt.savefig(fileName)
    plt.tight_layout()
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
