from Target import Target
from Sensor import Sensor
from FusionCenter import FusionCenter

SENSOR_AMOUNT = 4
SAMPLES = 100


def main():
    target = Target([1, 2], [1, 1])
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
            sensor.filter(measurement)

        fusionCenter.trackletFusion()
        fusionCenter.convexCombination()

        target.move()

    sensors[0].plot()


if __name__ == '__main__':
    main()
