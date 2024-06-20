class FusionCenter:
    def __init__(self):
        self.__sensors = []

    def convexCombination(self):
        for sensor in self.__sensors:
            prediction = sensor.getLastPrediction()

        return

    def trackletFusion(self):
        for sensor in self.__sensors:
            prediction = sensor.getLastPrediction()

        return

    def addSensor(self, sensor):
        self.__sensors.append(sensor)
