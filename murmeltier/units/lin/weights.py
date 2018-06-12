import numpy as np
from ..unit import Unit
from murmeltier.initializers import normal


class Weights(Unit):
    '''
    Matrix multiplication
    '''
    def __init__(self, in_specs, out_specs, initializer = normal, **kwargs):
        initializers = {}
        initializers['weights'] = initializer(shape = (in_specs, out_specs))
        self.config(in_specs = in_specs, out_specs = out_specs, initializers = initializers)
        self.initialize(**kwargs)


    def get_output(self, input):
        return np.matmul(input, self.params['weights'])
