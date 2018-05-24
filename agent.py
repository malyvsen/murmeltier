import config
from layer import *

class Agent:
    def __init__(self, stddev = 1):
        self.layers = []
        self.layers.append(Layer.randomize(in_dim = config.observation_dim, out_dim = 8, stddev = stddev))
        self.layers.append(Layer.randomize(in_dim = 8, out_dim = 6, stddev = stddev))
        self.layers.append(Layer.randomize(in_dim = 6, out_dim = 4, stddev = stddev))
        self.layers.append(Layer.randomize(in_dim = 4, out_dim = config.num_possible_actions, stddev = stddev))

    def get_action(self, observation):
        val = observation
        for layer in self.layers:
            val = layer.get_output(val)
        return np.argmax(val)

    def mutate(self, factor):
        for layer in self.layers:
            layer.mutate(factor, factor, factor)

    def __add__(self, other):
        result = Agent()
        for i in range(len(self.layers)):
            result.layers[i] = self.layers[i] + other.layers[i]
        return result

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, scalar):
        result = Agent()
        for i in range(len(self.layers)):
            result.layers[i] = self.layers[i] * scalar
        return result

    def __truediv__(self, scalar):
        result = Agent()
        for i in range(len(self.layers)):
            result.layers[i] = self.layers[i] / scalar
        return result
