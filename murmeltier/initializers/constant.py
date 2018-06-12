import numpy as np


def constant(shape = None, value = None):
    '''
    Just a constant value for all entries
    '''
    def initializer(shape = shape, value = value):
        return np.ones(shape = () if shape is None else shape) * (0.0 if value is None else value)
    return initializer
