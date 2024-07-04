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

    # create Sensors
    for i in range(0, SENSOR_AMOUNT):
        sensor = Sensor(target, f'Sensor {i + 1}')
        sensors.append(sensor)
        fusionCenter.addSensor(sensor)

    for _ in range(0, SAMPLES):
        for sensor in sensors:
            measurement = sensor.measure()
            sensor.kalmanFilter(measurement)

        kalmanFusion = fusionCenter.convexCombination(FilterModus.KALMAN_FILTER)
        kalmanFusions.append(kalmanFusion)

        target.move()

    # Plot measurement, prediction and filtering
    sensors[0].plot()
    sensors[1].plot()
    sensors[2].plot()
    sensors[3].plot()
    
    # Plot naive fusion
    plt.plot([position[0] for position in target.positions()[:-1]],
             [position[1] for position in target.positions()[:-1]],
             label='True target positions')
    plt.plot([position[0] for position in kalmanFusions],
             [position[1] for position in kalmanFusions],
             label='Fusion')
    plt.legend()
    plt.title('KF_Fusion')
    plt.savefig('KF_Fusion.png')
    plt.show()
    plt.clf()
    

if __name__ == '__main__':
    main()
