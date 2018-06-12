import numpy as np


def normal(shape = None, stddev = None, loc = None):
    '''
    Sample from the normal distribution
    '''
    def initializer(shape = shape, stddev = stddev, loc = loc):
        return np.random.normal(size = () if shape is None else shape, scale = 1.0 if stddev is None else stddev, loc = 0.0 if loc is None else loc)
    return initializer
