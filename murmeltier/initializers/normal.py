import numpy as np


def normal(shape = None, stddev = None, loc = None):
    '''
    Returns a initializer function which, int turn, returns an np.array with given shape
    The initializer function takes one obligatory argument: params - the standard deviation
    '''
    def initializer(shape = shape, stddev = stddev, loc = loc):
        return np.random.normal(size = () if shape is None else shape, scale = 1.0 if stddev is None else stddev, loc = 0.0 if loc is None else loc)
    return initializer
