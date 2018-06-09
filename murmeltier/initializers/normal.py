import numpy as np


def normal(shape = None):
    '''
    Returns a initializer function which, int turn, returns an np.array with given shape
    The initializer function takes one obligatory argument: params - the standard deviation
    '''
    def initializer(shape = shape, params):
        return np.random.normal(size = shape, scale = params)
    return initializer
