import numpy as np


def constant(shape = None, low = None, high = None):
    '''
    Sample from a uniform probability distribution between min and max
    '''
    def initializer(shape = shape, low = None, high = None):
        return np.random.uniform(size = () if shape is None else shape, low = 0.0 if low is None else low, high = 1.0 if high is None else high)
    return initializer
