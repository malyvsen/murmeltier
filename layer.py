import config
import numpy as np

class Layer:
    def __init__(self, weights, biases, alpha):
        self.weights = weights
        self.biases = biases
        self.alpha = alpha

    @classmethod
    def randomize(cls, in_dim, out_dim, stddev = 1):
        weights = np.random.normal(size = (out_dim, in_dim), scale = stddev)
        biases = np.random.normal(size = out_dim, scale = stddev)
        alpha = np.random.normal(scale = stddev)
        return cls(weights, biases, alpha)

    def get_output(self, input):
        pre_activation = np.matmul(self.weights, input) + self.biases
        return np.maximum(pre_activation, self.alpha * pre_activation)

    def mutate(self, weights_factor, biases_factor, alpha_factor):
        self.weights += np.random.normal(size = np.shape(self.weights)) * weights_factor
        self.biases += np.random.normal(size = np.shape(self.biases)) * biases_factor
        self.alpha += np.random.normal() * alpha_factor

    def __add__(self, other):
        return Layer(self.weights + other.weights, self.biases + other.biases, self.alpha + other.alpha)

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, scalar):
        return Layer(self.weights * scalar, self.biases * scalar, self.alpha * scalar)

    def __truediv__(self, scalar):
        return Layer(self.weights / scalar, self.biases / scalar, self.alpha / scalar)
