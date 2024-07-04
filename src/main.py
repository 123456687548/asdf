import math

from Target import Target
from Sensor import Sensor
from FusionCenter import FusionCenter
from Modus import FilterModus
import matplotlib.pyplot as plt

SENSOR_AMOUNT = 4
SAMPLES = 100


def main():
    target = Target([50, 100], [2, 10])
    fusionCenter = FusionCenter()
    sensors = []
    kalmanFusions = []
    federatedKalmanFusions = []
    distributedKalmanFusions = []

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

        kalmanFusion = fusionCenter.convexCombination(FilterModus.KALMAN_FILTER)
        kalmanFusions.append(kalmanFusion)

        federatedKalmanFusion = fusionCenter.convexCombination(FilterModus.FEDERATED_KALMAN_FILTER)
        federatedKalmanFusions.append(federatedKalmanFusion)

        distributedKalmanFusion = fusionCenter.convexCombination(FilterModus.DISTRIBUTED_KALMAN_FILTER)
        distributedKalmanFusions.append(distributedKalmanFusion)

        target.move()

    # Plot measurement, prediction and filtering for Kalman Filter
    for i, _ in enumerate(sensors):
        plt.subplot(math.floor(len(sensors) / 2), math.ceil(len(sensors) / 2), i + 1)
        sensors[i].plotKalmanFilter()
    # plt.legend()# todo
    plt.savefig('plots/KalmanFilter.png')
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Plot KF_Fusion
    plt.plot([position[0] for position in target.positions()[:-1]],
             [position[1] for position in target.positions()[:-1]],
             label='True target positions')
    plt.plot([position[0] for position in kalmanFusions],
             [position[1] for position in kalmanFusions],
             label='Fusion (Convex Combination)')
    plt.legend()
    plt.title('Kalman Filter in Sensors \n and Convex Combination in FC')
    plt.savefig('plots/KF_Fusion.png')
    plt.show()
    plt.clf()

    # Plot measurement, prediction and filtering for Federated Kalman Filter
    for i, _ in enumerate(sensors):
        plt.subplot(math.floor(len(sensors) / 2), math.ceil(len(sensors) / 2), i + 1)
        sensors[i].plotFederatedKalmanFilter()
    # plt.legend() todo
    plt.savefig('plots/FederatedKalmanFilter.png')
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Plot FKF_Fusion
    plt.plot([position[0] for position in target.positions()[:-1]],
             [position[1] for position in target.positions()[:-1]],
             label='True target positions')
    plt.plot([position[0] for position in federatedKalmanFusions],
             [position[1] for position in federatedKalmanFusions],
             label='Fusion (Convex Combination)')
    plt.legend()
    plt.title('Federated Kalman Filter in Sensors \n and Convex Combination in FC')
    plt.savefig('plots/FKF_Fusion.png')
    plt.show()
    plt.clf()

    # Plot measurement, prediction and filtering for Distributed Kalman Filter
    for i, _ in enumerate(sensors):
        plt.subplot(math.floor(len(sensors) / 2), math.ceil(len(sensors) / 2), i + 1)
        sensors[i].plotDistributedKalmanFilter()
    plt.legend()  # todo
    plt.savefig('plots/DistributedKalmanFilter.png')
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Plot DKF_Fusion
    plt.plot([position[0] for position in target.positions()[:-1]],
             [position[1] for position in target.positions()[:-1]],
             label='True target positions')
    plt.plot([position[0] for position in distributedKalmanFusions],
             [position[1] for position in distributedKalmanFusions],
             label='Fusion (Convex Combination)')
    plt.legend()
    plt.title('Distributed Kalman Filter in Sensors\n and Convex Combination in FC')
    plt.savefig('plots/DKF_Fusion.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
