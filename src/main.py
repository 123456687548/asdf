from Target import Target
from Sensor import Sensor
from FusionCenter import FusionCenter


def main():
    target = Target([1, 2], [1, 1])
    fusionCenter = FusionCenter()
    sensors = []

    # create Sensors
    for _ in range(1, 10):
        sensor = Sensor(target)
        sensors.append(sensor)
        fusionCenter.addSensor(sensor)

    for _ in range(1, 100):
        for sensor in sensors:
            measurement = sensor.measure()
            sensor.filter(measurement)

        fusionCenter.trackletFusion()
        fusionCenter.convexCombination()

        target.move()


if __name__ == '__main__':
    main()
