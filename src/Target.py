import numpy as np


class Target:

    def __init__(self, position, velocity):
        self.__position = np.array(position)
        self.__positions = []
        self.__positions.append(np.array(self.__position))
        self.__velocity = np.array(velocity)

    def positions(self):
        return self.__positions

    def state(self):
        return np.append(self.__position, self.__velocity)

    def move(self):
        self.__position += self.__velocity
        self.__positions.append(np.array(self.__position))

    def updateVelocity(self, velocity):
        self.__velocity = velocity

    def print(self):
        print(f'TARGET:\nPosition: {self.__position}\nVelocity: {self.__velocity}\nState: {self.state()}')
